"""Ingestion: file -> text blocks (OCR if needed) -> sections (Groq, structured) -> profile -> chunks -> 3 vectors -> Qdrant."""
import asyncio
import hashlib
import time
import uuid
from pathlib import Path

from app.kb.extract import extract
from app.kb.models import IngestResult
from app.kb.profile import refresh_profile
from app.kb.sections import doc_profile, make_chunks, segment
from app.kb.store import KnowledgeBase
from app.llm import fast_llm, llm
from app.telemetry import INGEST_DOCS, INGEST_DURATION, OCR_PAGES, log


async def ingest_file(kb: KnowledgeBase, session_id: str, path: Path, filename: str, config=None) -> tuple[IngestResult | None, dict | None]:
    """Returns (result, None) for a new document, or (None, existing_row) if this session already has it."""
    t0 = time.perf_counter()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if existing := kb.find_by_sha(session_id, sha):
        return None, existing
    kind = path.suffix.lower().lstrip(".")
    try:
        extracted = await asyncio.to_thread(extract, path)          # PyMuPDF / python-docx / RapidOCR
        sections = await segment(extracted.blocks, filename, llm, config)  # Groq structured output (windows)
        profile = await doc_profile(filename, sections, llm, config)       # Groq structured output (reduce)
        chunks = make_chunks(sections)
        result = IngestResult(uuid.uuid4().hex[:12], filename, profile, extracted.n_pages, extracted.ocr_pages, sections, len(chunks))
        await asyncio.to_thread(kb.add, session_id, sha, result, chunks)   # bekko x3 vectors + Qdrant + registry
        await refresh_profile(kb, session_id, filename, profile, fast_llm, config)
    except Exception as exc:
        INGEST_DOCS.add(1, {"file_type": kind, "status": "error"})
        log.error("ingest_failed", filename=filename, file_type=kind, error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    elapsed = time.perf_counter() - t0
    INGEST_DOCS.add(1, {"file_type": kind, "status": "ok"})
    INGEST_DURATION.record(elapsed, {"file_type": kind})
    if extracted.ocr_pages:
        OCR_PAGES.add(extracted.ocr_pages, {"file_type": kind})
    log.info("ingest_done", filename=filename, file_type=kind, pages=extracted.n_pages, ocr_pages=extracted.ocr_pages,
             sections=len(sections), chunks=len(chunks), duration_ms=round(elapsed * 1000))
    return result, None
