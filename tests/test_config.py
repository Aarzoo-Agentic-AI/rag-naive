"""Tests for rag_naive.config.

The developer's real .env is loaded into os.environ when rag_naive.config is
imported, so every test starts by clearing all variables Settings reads.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from rag_naive.config import PROJECT_ROOT, Settings

ENV_VARS = [
    "GROQ_API_KEY", "GROQ_MODEL", "GROQ_TEMPERATURE", "EMBEDDING_MODEL",
    "PDF_DIR", "CHROMA_DIR", "CHROMA_COLLECTION", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_project_root_is_the_repository_root():
    assert (PROJECT_ROOT / "rag_naive" / "config.py").is_file()
    assert (PROJECT_ROOT / "main.py").is_file()


def test_defaults_when_no_environment_variables_are_set():
    s = Settings()

    assert s.groq_api_key is None
    assert s.groq_model == "openai/gpt-oss-120b"
    assert s.groq_temperature == 0.0
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.pdf_dir == PROJECT_ROOT / "data" / "pdfs"
    assert s.chroma_dir == PROJECT_ROOT / "chroma_db"
    assert s.collection_name == "naive_rag"
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (1000, 200, 4)


def test_environment_variables_override_defaults(clean_env, tmp_path):
    values = {
        "GROQ_API_KEY": "gsk_test",
        "GROQ_MODEL": "qwen/qwen3.8-27b",
        "GROQ_TEMPERATURE": "0.7",
        "EMBEDDING_MODEL": "some/other-model",
        "PDF_DIR": str(tmp_path / "pdfs"),
        "CHROMA_DIR": str(tmp_path / "db"),
        "CHROMA_COLLECTION": "custom",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "50",
        "TOP_K": "8",
    }
    for name, value in values.items():
        clean_env.setenv(name, value)

    s = Settings()

    assert s.groq_api_key == "gsk_test"
    assert s.groq_model == "qwen/qwen3.8-27b"
    assert s.groq_temperature == 0.7
    assert s.embedding_model == "some/other-model"
    assert s.pdf_dir == tmp_path / "pdfs"
    assert s.chroma_dir == tmp_path / "db"
    assert s.collection_name == "custom"
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (500, 50, 8)


def test_numeric_settings_are_converted_to_numbers(clean_env):
    clean_env.setenv("TOP_K", "6")
    clean_env.setenv("GROQ_TEMPERATURE", "0.25")

    s = Settings()

    assert isinstance(s.top_k, int) and s.top_k == 6
    assert isinstance(s.groq_temperature, float) and s.groq_temperature == 0.25


@pytest.mark.parametrize("name", ["GROQ_MODEL", "CHUNK_SIZE", "TOP_K", "CHROMA_COLLECTION"])
def test_empty_environment_variable_falls_back_to_default(clean_env, name):
    clean_env.setenv(name, "")

    assert getattr(Settings(), _field_for(name)) == getattr(_defaults(), _field_for(name))


def test_invalid_integer_raises_value_error(clean_env):
    clean_env.setenv("TOP_K", "four")

    with pytest.raises(ValueError):
        Settings()


def test_relative_paths_are_resolved_against_project_root(clean_env):
    clean_env.setenv("PDF_DIR", "my/pdfs")
    clean_env.setenv("CHROMA_DIR", "index")

    s = Settings()

    assert s.pdf_dir == PROJECT_ROOT / "my" / "pdfs"
    assert s.chroma_dir == PROJECT_ROOT / "index"


def test_paths_do_not_depend_on_current_working_directory(clean_env, tmp_path):
    clean_env.chdir(tmp_path)

    s = Settings()

    assert s.pdf_dir == PROJECT_ROOT / "data" / "pdfs"
    assert s.chroma_dir == PROJECT_ROOT / "chroma_db"


def test_explicit_arguments_override_environment(clean_env):
    clean_env.setenv("TOP_K", "8")

    assert Settings(top_k=2).top_k == 2


def test_settings_are_read_at_instantiation_not_import(clean_env):
    clean_env.setenv("TOP_K", "3")
    first = Settings()
    clean_env.setenv("TOP_K", "9")

    assert (first.top_k, Settings().top_k) == (3, 9)


def test_settings_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().top_k = 10  # type: ignore[misc]


def test_require_groq_key_returns_key_when_set():
    assert Settings(groq_api_key="gsk_abc").require_groq_key() == "gsk_abc"


@pytest.mark.parametrize("missing", [None, ""])
def test_require_groq_key_raises_helpful_error(missing):
    with pytest.raises(RuntimeError, match=r"GROQ_API_KEY is not set.*\.env"):
        Settings(groq_api_key=missing).require_groq_key()


# --------------------------------------------------------------------------- #
def _field_for(env_name: str) -> str:
    return {
        "GROQ_MODEL": "groq_model",
        "CHUNK_SIZE": "chunk_size",
        "TOP_K": "top_k",
        "CHROMA_COLLECTION": "collection_name",
    }[env_name]


def _defaults() -> Settings:
    return Settings()  # env is clean thanks to the autouse fixture
