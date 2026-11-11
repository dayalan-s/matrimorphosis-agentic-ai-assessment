"""Make an image-only ("scanned") PDF from the first pages of a text PDF, to demo and test the OCR path.

    python scripts/make_scanned_pdf.py samples/react-reasoning-and-acting.pdf [pages]
"""
import sys
from pathlib import Path

import pymupdf

src = Path(sys.argv[1])
pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2
out = src.with_name(src.stem + "-scanned.pdf")
with pymupdf.open(src) as doc, pymupdf.open() as scan:
    for page in list(doc)[:pages]:
        png = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY).tobytes("png")  # rasterise: the text layer is gone
        scan.new_page(width=page.rect.width, height=page.rect.height).insert_image(page.rect, stream=png)
    scan.save(out)
print(out, f"({pages} image-only pages)")
