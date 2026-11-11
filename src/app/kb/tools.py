"""The two knowledge-base tools. Docstrings are prompts: they decide when the model picks the tool.

TODO(task 2): search_knowledge_base is a sync function that blocks (embedding + Qdrant): make it `async def`, call kb.search
  through asyncio.to_thread.
TODO(task 6): the knowledge base is keyed by the chat's session id. Key it by the USER (runtime.context.user_id, add it to
  Ctx in context.py): ingest and search must use the user's id, so every chat of a user shares one knowledge base and nobody else sees it.
"""
from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import KB_SEARCHES

LOW_CONFIDENCE = 0.30  # best cosine below this = nothing really matches


def build_kb_tools(kb: KnowledgeBase) -> list:
    @tool
    async def ingest_document(source: str, runtime: ToolRuntime[Ctx]) -> str:
        """Add a document to this session's knowledge base so it can be searched later.
        Reads PDF, DOCX, TXT, MD and images (scanned pages are OCR'd), splits it into
        sections, and indexes them. Call it whenever the user uploads a file or gives a
        document link, then tell the user in one sentence what the document contains.

        Args:
            source: the file_id from the upload message (e.g. "a1b2c3d4e5"), or a public http(s) URL to a document
        """
        session_id = runtime.context.session_id
        try:
            if source.startswith(("http://", "https://")):
                _, path = await files.download(session_id, source)
                filename = path.name.split("__", 1)[1]
            else:
                path, filename = files.resolve(session_id, source.strip())
            result, existing = await ingest_file(kb, session_id, path, filename, runtime.config)
        except Exception as exc:  # any failure -> ERROR[CODE] text the model can explain (no stack traces)
            raise to_tool_exception(exc) from exc
        if existing:
            return f"'{existing['filename']}' is already in the knowledge base ({existing['n_sections']} sections)."
        p = result.profile
        heads = "; ".join(s.heading for s in result.sections[:8])
        ocr = f", {result.ocr_pages} page(s) read with OCR" if result.ocr_pages else ""
        return (f"Ingested '{filename}': {p.title} [{p.doc_type}, {p.language}]. {p.summary} "
                f"{len(result.sections)} sections, {result.n_chunks} chunks{ocr}. Sections: {heads}. "
                f"Keywords: {', '.join(p.keywords)}.")

    @tool
    def search_knowledge_base(query: str, runtime: ToolRuntime[Ctx], k: int = 4, doc_name: str = "") -> str:
        """Search the documents the user uploaded in this session. Use it BEFORE other tools or your
        own knowledge whenever the KNOWLEDGE BASE overview/keywords in the system prompt suggest the
        question can be answered from those documents. Returns numbered passages, each with a
        bracketed source (document › section, pages) that you must copy exactly when citing.

        Args:
            query: what to look for, phrased as a full question or topic (not a single keyword)
            k: number of passages to return (1-6)
            doc_name: file name from the KNOWLEDGE BASE document list. ALWAYS pass it when the user names or hints at a
                specific document ("in the policy pdf", "according to report.docx"); leave empty to search all documents
        """
        sid = runtime.context.session_id
        exact = None
        if doc_name:
            exact = kb.resolve_doc_name(sid, doc_name)
            if exact is None:
                names = ", ".join(f'"{d["filename"]}"' for d in kb.documents(sid)) or "(none)"
                raise ToolException(f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}'. Documents in this session: {names}. "
                                    "Next step: retry with one of these names, or without doc_name.")
        try:
            hits = kb.search(sid, query, k=max(1, min(k, 6)), doc_name=exact)
        except Exception as exc:
            raise to_tool_exception(exc) from exc
        best = max((h.relevance for h in hits), default=0.0)
        KB_SEARCHES.add(1, {"found": str(best >= LOW_CONFIDENCE).lower()})
        if not hits:
            return "No passages found: the knowledge base is empty or the document name does not exist."
        head = f"{len(hits)} passages{f' from {exact}' if exact else ''}, best relevance {best:.2f}."
        if best < LOW_CONFIDENCE:
            head += " LOW CONFIDENCE: nothing clearly matches. Say the documents do not cover this unless a passage below answers it."
        else:
            head += " Answer only from these passages. After each claim, copy the bracketed source of the passage you used, exactly."
        body = "\n\n".join(f"Passage {i} [{h.cite}] relevance {h.relevance:.2f}\n{h.text}" for i, h in enumerate(hits, start=1))
        return f"{head}\n\n{body}"

    for t in (ingest_document, search_knowledge_base):
        t.handle_tool_error = True  # ToolException text goes back to the model as a status=error tool result
    return [ingest_document, search_knowledge_base]
