"""Retrieval stage: query encoding + plain top-k similarity search.

No metadata filtering, no MMR, no re-ranking, no query rewriting (naive RAG).
"""

from __future__ import annotations

from pathlib import PureWindowsPath

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def build_retriever(vectorstore: VectorStore, top_k: int) -> BaseRetriever:
    """The retriever embeds the query with the store's embedding model, then
    returns the ``top_k`` nearest chunks."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def citation_label(doc: Document) -> str:
    """Citation tag for a chunk, e.g. ``[HR_Policy_Handbook_2027.pdf, p. 4]``.

    Uses the file name only (not the full path) and a 1-based page number.
    """
    source = doc.metadata.get("source")
    name = PureWindowsPath(source).name if source else "unknown"
    page = doc.metadata.get("page")
    return f"[{name}, p. {page + 1}]" if isinstance(page, int) else f"[{name}]"


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into a single context string for the prompt.

    Each chunk is prefixed with its citation label so the LLM can copy it verbatim.
    """
    return "\n\n".join(f"{citation_label(doc)}\n{doc.page_content}" for doc in docs)
