"""Tests for rag_naive.vectorstore."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_naive import vectorstore
from rag_naive.vectorstore import get_embeddings, get_vectorstore

# Captured at import time, before the autouse fixture replaces it with a guard.
REAL_HF_EMBEDDINGS = vectorstore._hf_embeddings


class FakeHFEmbeddings:
    """Stands in for langchain_huggingface.HuggingFaceEmbeddings (no download)."""

    instances: list[FakeHFEmbeddings] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        FakeHFEmbeddings.instances.append(self)


@pytest.fixture
def fake_hf(monkeypatch):
    import langchain_huggingface

    FakeHFEmbeddings.instances = []
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeHFEmbeddings)
    REAL_HF_EMBEDDINGS.cache_clear()
    yield FakeHFEmbeddings
    REAL_HF_EMBEDDINGS.cache_clear()


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
def test_hf_embeddings_uses_model_name_and_normalized_vectors(fake_hf):
    emb = REAL_HF_EMBEDDINGS("sentence-transformers/all-MiniLM-L6-v2")

    assert emb.kwargs == {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "encode_kwargs": {"normalize_embeddings": True},
    }


def test_hf_embeddings_model_is_loaded_once_per_name(fake_hf):
    a = REAL_HF_EMBEDDINGS("model-a")
    again = REAL_HF_EMBEDDINGS("model-a")
    b = REAL_HF_EMBEDDINGS("model-b")

    assert a is again
    assert a is not b
    assert len(fake_hf.instances) == 2


def test_get_embeddings_uses_model_from_settings(monkeypatch, make_settings):
    requested = []
    monkeypatch.setattr(vectorstore, "_hf_embeddings", lambda name: requested.append(name) or name)

    result = get_embeddings(make_settings(embedding_model="custom/model"))

    assert requested == ["custom/model"]
    assert result == "custom/model"


# --------------------------------------------------------------------------- #
# Chroma vector store
# --------------------------------------------------------------------------- #
def test_get_vectorstore_creates_persist_directory(make_settings, fake_embeddings):
    s = make_settings()
    assert not s.chroma_dir.exists()

    get_vectorstore(s, embeddings=fake_embeddings)

    assert s.chroma_dir.is_dir()


def test_get_vectorstore_uses_collection_name_and_cosine_distance(make_settings, fake_embeddings):
    store = get_vectorstore(make_settings(collection_name="hr_docs"), embeddings=fake_embeddings)

    assert store._collection.name == "hr_docs"
    assert store._collection.metadata["hnsw:space"] == "cosine"


def test_get_vectorstore_uses_injected_embeddings(make_settings, fake_embeddings):
    store = get_vectorstore(make_settings(), embeddings=fake_embeddings)

    assert store.embeddings is fake_embeddings


def test_get_vectorstore_falls_back_to_configured_embeddings(monkeypatch, make_settings, fake_embeddings):
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda settings: fake_embeddings)

    store = get_vectorstore(make_settings())

    assert store.embeddings is fake_embeddings


def test_data_persists_across_store_instances(make_settings, fake_embeddings):
    s = make_settings()
    get_vectorstore(s, embeddings=fake_embeddings).add_documents(
        [Document(page_content="persisted chunk", metadata={"page": 0})], ids=["c1"]
    )

    reopened = get_vectorstore(s, embeddings=fake_embeddings).get()

    assert reopened["ids"] == ["c1"]
    assert reopened["documents"] == ["persisted chunk"]


def test_collections_are_isolated_by_name(make_settings, fake_embeddings):
    a = get_vectorstore(make_settings(collection_name="col_a"), embeddings=fake_embeddings)
    b = get_vectorstore(make_settings(collection_name="col_b"), embeddings=fake_embeddings)
    a.add_documents([Document(page_content="only in a")], ids=["x"])

    assert b.get()["ids"] == []


def test_invalid_collection_name_is_rejected(make_settings, fake_embeddings):
    # Chroma requires 3-512 chars of [a-zA-Z0-9._-]; a bad CHROMA_COLLECTION fails loudly.
    with pytest.raises(Exception, match="name"):
        get_vectorstore(make_settings(collection_name="x"), embeddings=fake_embeddings)
