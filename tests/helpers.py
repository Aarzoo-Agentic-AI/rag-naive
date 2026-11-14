"""Reusable test helpers (importable as ``tests.helpers``)."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.embeddings import Embeddings


# --------------------------------------------------------------------------- #
# Minimal PDF writer (no extra dependency) so tests exercise real pypdf parsing.
# --------------------------------------------------------------------------- #
def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a PDF with one line of Helvetica text per page ("" = blank page)."""
    objects: list[str | None] = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        None,  # Pages object, filled in once the page ids are known
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 50 750 Td ({_escape_pdf_text(text)}) Tj ET" if text else ""
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objects)
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path


# --------------------------------------------------------------------------- #
# Predictable embeddings: one dimension per vocabulary word.
# --------------------------------------------------------------------------- #
VOCAB = ["pto", "vacation", "bonus", "salary", "401k", "match", "insurance", "deductible"]


class KeywordEmbeddings(Embeddings):
    """Bag-of-words vectors so nearest-neighbour results are predictable.

    Records every query it encodes so tests can assert on query encoding.
    """

    def __init__(self) -> None:
        self.query_calls: list[str] = []

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        vec = [float(words.count(v)) for v in VOCAB]
        return vec if any(vec) else [1e-6] * len(VOCAB)  # avoid zero vector (cosine)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)
