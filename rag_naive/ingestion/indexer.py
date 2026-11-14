"""Offline indexing: load PDFs -> split into chunks -> embed -> store in ChromaDB."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config import Settings
from ..vectorstore import get_vectorstore
from .loader import load_pdfs
from .splitter import split_documents

logger = logging.getLogger(__name__)


def chunk_id(chunk: Document) -> str:
    """Deterministic ID so re-ingesting the same PDFs upserts instead of duplicating."""
    meta = chunk.metadata
    key = f"{meta.get('source')}|{meta.get('page')}|{meta.get('start_index')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ingest(
    settings: Settings,
    source: Path | None = None,
    embeddings: Embeddings | None = None,
    reset: bool = False,
) -> int:
    """Index PDFs into ChromaDB. Returns the number of chunks stored."""
    documents = load_pdfs(source or settings.pdf_dir)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    logger.info("Loaded %d pages -> %d chunks", len(documents), len(chunks))

    store = get_vectorstore(settings, embeddings=embeddings)
    if reset:
        store.reset_collection()

    if chunks:
        store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return len(chunks)
