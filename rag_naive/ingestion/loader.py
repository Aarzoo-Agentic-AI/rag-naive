"""Load PDF files into LangChain Documents (one Document per page)."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def find_pdfs(source: Path) -> list[Path]:
    """Return a single PDF path, or every PDF under a directory (recursively)."""
    source = Path(source)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() == ".pdf")
    else:
        raise FileNotFoundError(f"PDF source not found: {source}")

    if not files:
        raise FileNotFoundError(f"No PDF files found in: {source}")
    return files


def load_pdfs(source: Path) -> list[Document]:
    """Load PDF pages with ``source`` and ``page`` metadata, skipping blank pages."""
    documents: list[Document] = []
    for path in find_pdfs(source):
        logger.info("Loading %s", path)
        pages = PyPDFLoader(str(path)).load()
        documents.extend(p for p in pages if p.page_content.strip())
    return documents
