"""Structured extraction with Groq: sections as block ranges, then a document profile.

Why ranges? Asking the model to retype every section would burn tokens and risk altering the text.
It returns `start_block..end_block`; our code slices the exact original text. Long documents are
processed in windows (map), then one profile call summarises the headings (reduce).
"""
import asyncio
import re

from app.errors import fallback_ok
from app.telemetry import log
from app.kb.models import Block, Chunk, DocProfile, Section, SectionSpan, WindowSections

WINDOW_CHARS = 9000  # preview characters per Groq call
PREVIEW = 220        # characters of each block the model sees
CHUNK_CHARS = 900    # max characters per embedded chunk (small passages keep the agent's context within per-request limits)

SEGMENT_PROMPT = """You split a document into sections. Below are numbered text blocks (long blocks are cut with '...').
Return the sections in order as ranges of block ids.
- Cover every block exactly once, in order; sections must not overlap.
- Use the document's own headings (lines starting with '#', or short title-like lines). If a part has no heading, write a short descriptive one.
- Aim for 3-12 sections per window. Never make one section per block.
{carry}
File: {filename}
Blocks:
{blocks}"""

PROFILE_PROMPT = """Describe this document for a search index.
File: {filename}
Sections (heading: summary):
{sections}

Start of the text:
{head}"""


_sleep = asyncio.sleep  # patched in tests
RETRY_WAIT_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)", re.I)


async def call_model(model, schema, prompt, config=None, attempts: int = 3):
    """Structured Groq call. Rate limits wait exactly as long as the API says (capped), then retry; other errors propagate."""
    for attempt in range(1, attempts + 1):
        try:
            # json_schema mode, not tool-calling: on Groq the tool route intermittently fails with 400 tool_use_failed
            return await model.with_structured_output(schema, method="json_schema").ainvoke(prompt, config=config)
        except Exception as exc:
            if attempt == attempts or not (type(exc).__name__ == "RateLimitError" or fallback_ok(exc)):
                raise  # out of attempts, or an outage/auth problem that retrying cannot fix
            if type(exc).__name__ != "RateLimitError":  # bad model output: one more sample, no waiting
                log.warning("structured_output_retry", schema=schema.__name__, attempt=attempt, error_type=type(exc).__name__, error=str(exc)[:1000])
                continue
            m = RETRY_WAIT_RE.search(str(exc))
            wait = float(m.group(1)) / (1000 if m and m.group(2).lower() == "ms" else 1) if m else 5.0
            await _sleep(min(wait + 0.5, 30))


def windows(blocks: list[Block]) -> list[list[Block]]:
    out, cur, size = [], [], 0
    for b in blocks:
        n = min(len(b.text), PREVIEW) + 8
        if cur and size + n > WINDOW_CHARS:
            out, cur, size = [*out, cur], [], 0
        cur.append(b)
        size += n
    return [*out, cur] if cur else out


def _preview(b: Block) -> str:
    t = b.text if len(b.text) <= PREVIEW else b.text[:PREVIEW] + "..."
    return f"[{b.id}]{f' (p{b.page})' if b.page else ''} {t}"


def repair(spans: list[SectionSpan], lo: int, hi: int, n: int) -> list[SectionSpan]:
    """Make the model's ranges a clean partition of [lo, hi]: clamp, de-overlap, fill gaps."""
    spans = sorted((s for s in spans if s.end_block >= s.start_block), key=lambda s: s.start_block)
    fixed: list[SectionSpan] = []
    for s in spans:
        start = max(lo, s.start_block, fixed[-1].end_block + 1 if fixed else lo)
        end = min(hi, s.end_block)
        if end >= start:
            fixed.append(s.model_copy(update={"start_block": start, "end_block": end}))
    if not fixed:
        return [SectionSpan(heading=f"Part {n}", level=1, start_block=lo, end_block=hi, keywords=[], summary="")]
    fixed[0].start_block = lo  # gaps are absorbed by the previous section
    for prev, nxt in zip(fixed, fixed[1:]):
        prev.end_block = nxt.start_block - 1
    fixed[-1].end_block = hi
    return fixed


async def segment(blocks: list[Block], filename: str, llm, config=None) -> list[Section]:
    by_id = {b.id: b for b in blocks}
    spans: list[SectionSpan] = []
    carry = ""
    for n, win in enumerate(windows(blocks), start=1):
        prompt = SEGMENT_PROMPT.format(
            carry=carry, filename=filename, blocks="\n".join(_preview(b) for b in win)
        )
        try:
            got = (await call_model(llm, WindowSections, prompt, config)).sections
        except Exception as exc:
            if not fallback_ok(exc):  # outage / rate limit / auth: fail the ingestion, write nothing
                raise
            log.warning("segment_window_fallback", window=n, error_type=type(exc).__name__, error=str(exc)[:1000])
            got = []  # the model answered badly for one window: keep the document as a single part
        part = repair(got, win[0].id, win[-1].id, n)
        spans += part
        carry = f"The previous window ended inside the section titled '{part[-1].heading}'. If the first blocks continue it, reuse that exact heading."
    merged: list[SectionSpan] = []
    for s in spans:  # a section split across windows comes back with the same heading
        if merged and merged[-1].heading.strip().lower() == s.heading.strip().lower() and merged[-1].end_block + 1 == s.start_block:
            merged[-1].end_block = s.end_block
            merged[-1].keywords = list(dict.fromkeys([*merged[-1].keywords, *s.keywords]))[:6]
        else:
            merged.append(s)
    sections = []
    for i, s in enumerate(merged):
        bs = [by_id[j] for j in range(s.start_block, s.end_block + 1) if j in by_id]
        pages = [b.page for b in bs if b.page]
        sections.append(Section(
            index=i, heading=s.heading.strip().lstrip("# ") or f"Section {i + 1}", level=s.level,
            text="\n\n".join(b.text for b in bs), page_start=min(pages, default=None), page_end=max(pages, default=None),
            keywords=[k.strip() for k in s.keywords][:6], summary=s.summary.strip(),
        ))
    return sections


async def doc_profile(filename: str, sections: list[Section], llm, config=None) -> DocProfile:
    listing = "\n".join(f"- {s.heading}: {s.summary}" for s in sections)[:6000]
    prompt = PROFILE_PROMPT.format(filename=filename, sections=listing, head=sections[0].text[:600])
    try:
        return await call_model(llm, DocProfile, prompt, config)
    except Exception as exc:
        if not fallback_ok(exc):
            raise
        kws = list(dict.fromkeys(k for s in sections for k in s.keywords))[:8]
        return DocProfile(title=filename.rsplit(".", 1)[0], doc_type="document", language="unknown",
                          summary=f"{len(sections)} sections: " + ", ".join(s.heading for s in sections[:5]), keywords=kws)


def make_chunks(sections: list[Section], max_chars: int = CHUNK_CHARS) -> list[Chunk]:
    """Pack paragraphs of a section into chunks; a chunk never crosses a section boundary."""
    chunks: list[Chunk] = []
    for sec in sections:
        parts, cur = [], ""
        for para in sec.text.split("\n\n"):
            while len(para) > max_chars:  # very long paragraph: split at a sentence end if possible
                cut = max(para.rfind(". ", 0, max_chars), max_chars // 2) + 1
                parts.append(para[:cut].strip())
                para = para[cut:].strip()
            if cur and len(cur) + len(para) + 2 > max_chars:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n\n{para}".strip()
        if cur:
            parts.append(cur)
        chunks += [Chunk(sec, i, p) for i, p in enumerate(p for p in parts if p)]
    return chunks
