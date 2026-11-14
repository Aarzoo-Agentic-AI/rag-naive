"""Embedding model and ChromaDB vector store factories."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .config import Settings


@lru_cache(maxsize=4)
def _hf_embeddings(model_name: str) -> Embeddings:
    # Imported lazily: sentence-transformers/torch is slow to import.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(settings: Settings) -> Embeddings:
    """Return the encoder used for BOTH documents and queries."""
    return _hf_embeddings(settings.embedding_model)


def get_vectorstore(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
