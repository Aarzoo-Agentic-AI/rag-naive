"""Naive (standard) RAG: PDF -> ChromaDB -> Groq, orchestrated with LangChain.

Package layout:
    config.py       settings from environment / .env
    vectorstore.py  embedding model + ChromaDB (shared by ingestion and retrieval)
    ingestion/      load PDFs, split into chunks, index into ChromaDB
    retrieval/      query encoding + top-k similarity search
    generation/     prompt + Groq LLM
    pipeline.py     NaiveRAG: wires retrieval -> generation
"""

from .config import Settings
from .ingestion import ingest, load_pdfs, split_documents
from .pipeline import NaiveRAG, RAGResponse
from .retrieval import format_docs

__all__ = [
    "Settings",
    "ingest",
    "load_pdfs",
    "split_documents",
    "NaiveRAG",
    "RAGResponse",
    "format_docs",
]
