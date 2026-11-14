"""Retrieval stage: encode the query and fetch the top-k chunks from ChromaDB."""

from .retriever import build_retriever, citation_label, format_docs

__all__ = ["build_retriever", "citation_label", "format_docs"]
