"""Shared pytest fixtures.

Unit tests must never hit the network or depend on the developer's real .env,
so Settings are always built explicitly via the ``make_settings`` fixture, and
loading the real HuggingFace embedding model is blocked. Tests marked ``e2e``
are exempt: they deliberately use the real embedding model and Groq.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_naive.config import Settings
from tests.helpers import KeywordEmbeddings, write_pdf


@pytest.fixture(autouse=True)
def _block_real_embedding_model(request, monkeypatch):
    """Fail fast if a unit test forgets to inject fake embeddings."""
    if request.node.get_closest_marker("e2e"):
        return

    def _refuse(model_name: str):
        raise AssertionError(
            f"Unit test tried to load the real embedding model '{model_name}'. "
            "Pass `embeddings=fake_embeddings` instead."
        )

    monkeypatch.setattr("rag_naive.vectorstore._hf_embeddings", _refuse)


@pytest.fixture
def make_settings(tmp_path: Path):
    """Factory for Settings with isolated paths and a dummy API key."""

    def _make(**overrides) -> Settings:
        defaults = dict(
            groq_api_key="test-key",
            groq_model="test-model",
            groq_temperature=0.0,
            pdf_dir=tmp_path / "pdfs",
            chroma_dir=tmp_path / "chroma",
            collection_name="test_collection",
        )
        defaults.update(overrides)
        return Settings(**defaults)

    return _make


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Same text -> same vector, no model download."""
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def keyword_embeddings() -> KeywordEmbeddings:
    """Embeddings with predictable similarity (see tests/helpers.py)."""
    return KeywordEmbeddings()


@pytest.fixture
def make_pdf() -> Callable[[Path, list[str]], Path]:
    return write_pdf
