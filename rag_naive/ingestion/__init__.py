"""Ingestion stage: PDF loading, chunking and indexing into ChromaDB."""

from .indexer import chunk_id, ingest
from .loader import find_pdfs, load_pdfs
from .splitter import split_documents

__all__ = ["chunk_id", "ingest", "find_pdfs", "load_pdfs", "split_documents"]
