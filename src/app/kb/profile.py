"""Maintain the session's knowledge-base profile. Updated at ingestion; the agent reads it every turn."""
from app.kb.models import DocProfile, KBProfile
from app.kb.sections import call_model
from app.kb.store import KnowledgeBase

MERGE_PROMPT = """You maintain a short description of a knowledge base so an assistant can decide whether a question can be answered from it.
Current overview: {overview}
Current keywords: {keywords}

A new document was just added:
- file: {filename}
- type: {doc_type}
- summary: {summary}
- keywords: {doc_keywords}

Return the updated overview (1-2 sentences, what KINDS of information are in the knowledge base) and up to 15 short keywords covering all documents."""


async def refresh_profile(kb: KnowledgeBase, session_id: str, filename: str, p: DocProfile, llm, config=None) -> None:
    prior = kb.get_profile(session_id) or {"overview": "(empty)", "keywords": []}
    prompt = MERGE_PROMPT.format(overview=prior["overview"], keywords=", ".join(prior["keywords"]) or "(none)",
                                 filename=filename, doc_type=p.doc_type, summary=p.summary, doc_keywords=", ".join(p.keywords))
    try:
        new: KBProfile = await call_model(llm, KBProfile, prompt, config)
        overview, keywords = new.overview, new.keywords[:15]
    except Exception:  # any failure: the document is already stored, so the profile still updates deterministically
        keywords = list(dict.fromkeys([*prior["keywords"], *p.keywords]))[:15]
        overview = "Documents including: " + "; ".join(d["filename"] for d in kb.documents(session_id))
    kb.save_profile(session_id, overview, keywords)
