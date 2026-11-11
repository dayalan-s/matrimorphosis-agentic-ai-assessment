"""Any document -> numbered text blocks. Text layer first; RapidOCR (PP-OCRv6) when there is none.

PDF  : PyMuPDF text blocks per page; pages with (almost) no text layer are rendered and OCR'd
DOCX : paragraphs and tables in order; heading styles kept as '#' prefixes
Image: OCR
TXT/MD: paragraphs
"""
import threading
from pathlib import Path

import numpy as np

from app.errors import DocError
from app.kb.models import Block, Extracted

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
MAX_PAGES = 60          # demo guard: bounds Groq tokens and OCR time
MIN_TEXT_CHARS = 40     # a PDF page with less text than this is treated as scanned
OCR_DPI = 200

_ocr = None
_ocr_lock = threading.Lock()


def ocr_engine():
    """One shared RapidOCR engine, PP-OCRv6 detection + recognition (models download on first use)."""
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import EngineType, OCRVersion

            # PP-OCRv6 small models ship inside the wheel as .onnx; pin ONNX Runtime so no other backend (torch, paddle) is ever used
            _ocr = RapidOCR(params={
                "Det.ocr_version": OCRVersion.PPOCRV6, "Det.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
            })
        return _ocr


def ocr_paragraphs(image: np.ndarray) -> list[str]:
    """OCR one image and merge lines into paragraphs (a bigger vertical gap starts a new paragraph)."""
    out = ocr_engine()(image)
    if out is None or out.txts is None or len(out.txts) == 0:
        return []
    lines = sorted(
        ((float(b[:, 1].min()), float(b[:, 1].max()), t) for b, t in zip(out.boxes, out.txts) if t.strip()),
        key=lambda x: x[0],
    )
    paragraphs, current, prev_bottom = [], [], None
    height = np.median([bot - top for top, bot, _ in lines]) or 1.0
    for top, bottom, text in lines:
        if prev_bottom is not None and top - prev_bottom > 0.7 * height and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text.strip())
        prev_bottom = bottom
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _pdf(path: Path) -> Extracted:
    import pymupdf as fitz  # PyMuPDF

    blocks: list[Block] = []
    ocr_pages = 0
    try:
        doc = fitz.open(path)
    except Exception as exc:  # fitz.FileDataError and friends
        raise DocError("CORRUPT_FILE", "This PDF is damaged or not a valid PDF.", "Ask the user to re-export it and upload again.") from exc
    with doc:
        if doc.needs_pass:
            raise DocError("ENCRYPTED_FILE", "This PDF is password-protected.", "Ask the user for an unprotected copy.")
        if doc.page_count > MAX_PAGES:
            raise DocError("TOO_LARGE", f"This PDF has {doc.page_count} pages; the limit is {MAX_PAGES}.", "Ask the user to upload a shorter excerpt.")
        for pno, page in enumerate(doc, start=1):
            texts = [b[4].strip() for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
            ocr = sum(len(t) for t in texts) < MIN_TEXT_CHARS
            if ocr:
                pix = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csRGB)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                texts, ocr_pages = ocr_paragraphs(img), ocr_pages + 1
            blocks += [Block(len(blocks) + i, t.replace("\n", " "), pno, ocr) for i, t in enumerate(texts)]
        n_pages = doc.page_count
    return Extracted(blocks, "pdf", n_pages, ocr_pages)


def _docx(path: Path) -> Extracted:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    texts: list[str] = []
    for item in docx.Document(path).iter_inner_content():
        if isinstance(item, Paragraph) and item.text.strip():
            style = item.style.name if item.style is not None else ""
            if style.startswith("Heading") and style.split()[-1].isdigit():
                texts.append("#" * int(style.split()[-1]) + " " + item.text.strip())
            elif style == "Title":
                texts.append("# " + item.text.strip())
            else:
                texts.append(item.text.strip())
        elif isinstance(item, Table):
            for row in item.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    texts.append(" | ".join(cells))
    return Extracted([Block(i, t) for i, t in enumerate(texts)], "docx")


def _text(path: Path) -> Extracted:
    import re

    parts = [p.strip() for p in re.split(r"\n\s*\n", path.read_text(errors="replace")) if p.strip()]
    return Extracted([Block(i, p.replace("\n", " ")) for i, p in enumerate(parts)], path.suffix.lstrip("."))


def _image(path: Path) -> Extracted:
    from PIL import Image

    paragraphs = ocr_paragraphs(np.array(Image.open(path).convert("RGB")))
    return Extracted([Block(i, t, 1, True) for i, t in enumerate(paragraphs)], "image", 1, 1)


def extract(path: Path) -> Extracted:
    ext = path.suffix.lower()
    if ext == ".doc":
        raise DocError("UNSUPPORTED_FILE", "Legacy .doc files are not supported.", "Ask the user to save it as .docx or PDF.")
    if ext not in SUPPORTED:
        raise DocError("UNSUPPORTED_FILE", f"'{ext}' files are not supported (supported: {', '.join(sorted(SUPPORTED))}).",
                       "Ask the user for a PDF, DOCX, TXT, MD or image.")
    try:
        result = {".pdf": _pdf, ".docx": _docx, ".txt": _text, ".md": _text}.get(ext, _image)(path)
    except DocError:
        raise
    except Exception as exc:  # e.g. a corrupt DOCX or unreadable image
        raise DocError("CORRUPT_FILE", f"The {ext} file could not be read.", "Ask the user to re-save it and upload again.") from exc
    if not result.blocks:
        raise DocError("EMPTY_DOCUMENT", "No readable text was found, even with OCR.", "Ask for a clearer scan or a text-based version.")
    return result
