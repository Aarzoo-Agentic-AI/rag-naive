"""Retrieval stage: query encoding + plain top-k similarity search.

No metadata filtering, no MMR, no re-ranking, no query rewriting (naive RAG).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def build_retriever(vectorstore: VectorStore, top_k: int) -> BaseRetriever:
    """The retriever embeds the query with the store's embedding model, then
    returns the ``top_k`` nearest chunks."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into a single context string for the prompt."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, page {page + 1}" if isinstance(page, int) else source
        parts.append(f"[{i}] ({label})\n{doc.page_content}")
    return "\n\n".join(parts)
