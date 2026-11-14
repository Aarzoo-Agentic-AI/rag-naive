"""Retrieval stage: encode the query and fetch the top-k chunks from ChromaDB."""

from .retriever import build_retriever, format_docs

__all__ = ["build_retriever", "format_docs"]
