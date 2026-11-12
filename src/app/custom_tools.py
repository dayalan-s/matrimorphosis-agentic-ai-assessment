
"""Tasks 7 and 8: custom knowledge-base tools with observability and document management."""

import asyncio
import time

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool
from opentelemetry import metrics

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import log


_meter = metrics.get_meter("agent")

KBSTATS_CALLS = _meter.create_counter(
    "kbstats.calls",
    unit="1",
    description="Knowledge-base statistics tool calls",
)

OUTLINE_CALLS = _meter.create_counter(
    "outline.calls",
    unit="1",
    description="Document outline tool calls",
)


def _get_user_id(runtime: ToolRuntime[Ctx]) -> str:
    """Get the authenticated user's knowledge-base ID."""
    user_id = runtime.context.user_id

    if not user_id:
        raise ToolException(
            "ERROR[AUTH_REQUIRED]: A logged-in user is required. "
            "Next step: Sign in and try again."
        )

    return user_id


async def _find_document(
    kb: KnowledgeBase,
    user_id: str,
    doc_name: str,
) -> tuple[dict | None, list[dict]]:
    """Resolve a document name and return its stored row and all user documents."""
    exact_name = await asyncio.to_thread(
        kb.resolve_doc_name,
        user_id,
        doc_name,
    )

    documents = await asyncio.to_thread(kb.documents, user_id)

    if exact_name is None:
        return None, documents

    document = next(
        (doc for doc in documents if doc["filename"] == exact_name),
        None,
    )

    return document, documents


def _unknown_document(doc_name: str, documents: list[dict]) -> ToolException:
    names = ", ".join(
        f'"{doc["filename"]}"' for doc in documents
    ) or "(none)"

    return ToolException(
        f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}'. "
        f"Documents in your knowledge base: {names}. "
        "Next step: Retry with one of these filenames."
    )


def build_custom_tools(kb: KnowledgeBase) -> list:

    @tool
    async def kb_stats(
        runtime: ToolRuntime[Ctx],
        top_keywords: int = 5,
    ) -> str:
        """Show statistics about the logged-in user's knowledge base.

        Use this when the user asks how many documents, sections or chunks
        are stored, or asks which topics appear most often.

        Args:
            top_keywords: Number of frequent keywords to include (1-20).
        """
        started = time.perf_counter()
        status = "error"

        try:
            if not isinstance(top_keywords, int) or not 1 <= top_keywords <= 20:
                raise ToolException(
                    "ERROR[INVALID_INPUT]: top_keywords must be between 1 and 20. "
                    "Next step: Choose a value from 1 to 20."
                )

            user_id = _get_user_id(runtime)

            stats = await asyncio.to_thread(kb.stats, user_id)

            if stats["documents"] == 0:
                status = "ok"
                return "The knowledge base is empty."

            keywords = stats.get("keywords", [])[:top_keywords]
            keyword_text = ", ".join(keywords) if keywords else "none"

            status = "ok"

            return (
                f'{stats["documents"]} documents, '
                f'{stats["sections"]} sections, '
                f'{stats["chunks"]} chunks. '
                f"Top keywords: {keyword_text}"
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)

            KBSTATS_CALLS.add(1, {"status": status})

            log.info(
                "kb_stats_done",
                tool="kb_stats",
                status=status,
                duration_ms=duration_ms,
            )

    @tool
    async def document_outline(
        doc_name: str,
        runtime: ToolRuntime[Ctx],
    ) -> str:
        """Show the section-by-section outline of a document in the user's knowledge base.

        Use this when the user asks for a document's structure, headings,
        sections, page locations, or a summary of what the document covers.
        Use it before searching when the user wants to understand the document's
        overall structure.

        Args:
            doc_name: The document's filename as shown in the knowledge-base document list.
        """
        started = time.perf_counter()
        status = "error"
        sections_count = 0

        try:
            user_id = _get_user_id(runtime)

            document, documents = await _find_document(
                kb,
                user_id,
                doc_name,
            )

            if document is None:
                raise _unknown_document(doc_name, documents)

            outline = await asyncio.to_thread(
                kb.outline,
                user_id,
                document["doc_id"],
            )

            sections_count = len(outline)

            lines = [
                f"Outline of '{document['filename']}' ({sections_count} sections):"
            ]

            for i, section in enumerate(outline, start=1):
                heading = section.get("heading") or "Untitled section"
                level = section.get("level", 1)
                summary = section.get("summary") or ""

                page_start = section.get("page_start")
                page_end = section.get("page_end")

                if page_start is None:
                    page_text = ""
                elif page_start == page_end:
                    page_text = f" (p.{page_start})"
                else:
                    page_text = f" (p.{page_start}-{page_end})"

                lines.append(
                    f"{i}. [L{level}] {heading}{page_text}: {summary}"
                )

            status = "ok"
            return "\n".join(lines)

        except Exception as exc:
            raise to_tool_exception(exc) from exc

        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)

            OUTLINE_CALLS.add(1, {"status": status})

            log.info(
                "document_outline_done",
                tool="document_outline",
                status=status,
                duration_ms=duration_ms,
                sections=sections_count,
            )

    @tool
    async def delete_document(
        doc_name: str,
        runtime: ToolRuntime[Ctx],
    ) -> str:
        """Delete a document from the logged-in user's knowledge base.

        Use this when the user explicitly asks to remove or delete a document.

        Args:
            doc_name: The exact or recognizable filename of the document to delete.
        """
        try:
            user_id = _get_user_id(runtime)

            document, documents = await _find_document(
                kb,
                user_id,
                doc_name,
            )

            if document is None:
                raise _unknown_document(doc_name, documents)

            deleted = await asyncio.to_thread(
                kb.delete_document,
                user_id,
                document["doc_id"],
            )

            if not deleted:
                raise _unknown_document(doc_name, documents)

            remaining = await asyncio.to_thread(kb.documents, user_id)

            return (
                f"Deleted '{document['filename']}'. "
                f"{len(remaining)} documents left."
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

    @tool
    async def update_document(
        doc_name: str,
        source: str,
        runtime: ToolRuntime[Ctx],
    ) -> str:
        """Replace an existing knowledge-base document with a newly uploaded file.

        Use this when the user asks to update, replace, or provide a newer
        version of a document already in the knowledge base. The new file is
        indexed first; the old version is removed only after indexing succeeds.

        Args:
            doc_name: The filename of the existing document to replace.
            source: The file_id of the newly uploaded replacement file.
        """
        try:
            user_id = _get_user_id(runtime)

            old_document, documents = await _find_document(
                kb,
                user_id,
                doc_name,
            )

            if old_document is None:
                raise _unknown_document(doc_name, documents)

            # Resolve the new upload. If this fails, the old document remains.
            new_path, _ = files.resolve(user_id, source.strip())

            # Ingest the replacement using the old document's filename.
            # ingest_file returns an existing row when the content is identical.
            result, existing = await ingest_file(
                kb,
                user_id,
                new_path,
                old_document["filename"],
                runtime.config,
            )

            if existing:
                return f"'{old_document['filename']}' is already up to date."

            # Only delete the old version after the new version was ingested.
            deleted = await asyncio.to_thread(
                kb.delete_document,
                user_id,
                old_document["doc_id"],
            )

            if not deleted:
                raise RuntimeError("The previous document could not be replaced.")

            return (
                f"Updated '{old_document['filename']}': "
                f"{len(result.sections)} sections, "
                f"{result.n_chunks} chunks."
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

    tools = [
        kb_stats,
        document_outline,
        delete_document,
        update_document,
    ]

    for custom_tool in tools:
        custom_tool.handle_tool_error = True

    return tools
