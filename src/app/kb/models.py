"""Pydantic models: what the LLM must return, and what the pipeline stores."""
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


# ---- LLM output (structured extraction) ----
class SectionSpan(BaseModel):
    """One section, expressed as a range of numbered text blocks (the model never retypes the text)."""

    heading: str = Field(description="Short section title; if the text has none, write a descriptive one")
    level: int = Field(1, description="1 = top-level section, 2 = subsection, ...")
    start_block: int = Field(description="First block id of the section")
    end_block: int = Field(description="Last block id of the section (inclusive)")
    keywords: list[str] = Field(description="3-6 short keywords for this section")
    summary: str = Field(description="One sentence: what this section says")


class WindowSections(BaseModel):
    sections: list[SectionSpan]


class DocProfile(BaseModel):
    """Document-level metadata, derived from the section headings and summaries."""

    title: str = Field(description="The document's title")
    doc_type: str = Field(description="e.g. report, contract, invoice, manual, paper, slides, notes, policy")
    language: str = Field(description="Main language, e.g. English")
    summary: str = Field(description="Two sentences: what kind of information this document contains")
    keywords: list[str] = Field(description="4-8 short keywords describing the document")


class KBProfile(BaseModel):
    """Session-level knowledge-base summary. The agent reads it to decide whether to search."""

    overview: str = Field(description="1-2 sentences: what kind of information the knowledge base holds")
    keywords: list[str] = Field(description="Up to 15 short keywords covering all documents")


# ---- pipeline data ----
@dataclass
class Block:
    id: int
    text: str
    page: int | None = None  # None for formats without pages (docx, txt)
    ocr: bool = False


@dataclass
class Extracted:
    blocks: list[Block]
    file_type: str
    n_pages: int = 0
    ocr_pages: int = 0


@dataclass
class Section:
    index: int
    heading: str
    level: int
    text: str
    page_start: int | None
    page_end: int | None
    keywords: list[str]
    summary: str


@dataclass
class Chunk:
    section: Section
    part: int
    text: str


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    profile: DocProfile
    n_pages: int
    ocr_pages: int
    sections: list[Section] = field(default_factory=list)
    n_chunks: int = 0
