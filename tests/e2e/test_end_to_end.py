"""End-to-end tests: real PDFs -> real embeddings -> real ChromaDB -> real Groq.

Opt-in (they download the embedding model on first run, call the Groq API and
cost a few tokens):

    pytest -m e2e

Skipped automatically when GROQ_API_KEY is not configured.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from rag_naive import NaiveRAG, Settings, ingest
from rag_naive.config import PROJECT_ROOT
from rag_naive.vectorstore import get_embeddings

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not Settings().groq_api_key, reason="GROQ_API_KEY not configured"),
]

# Made-up facts the LLM cannot know from pre-training, so a correct answer
# proves the information came from retrieval.
PAGES = {
    "facilities.pdf": [
        "Facilities overview. The Orion campus has three buildings.",
        "Parking: a Zephyr parking permit costs 37 dollars per month.",
    ],
    "travel.pdf": [
        "Travel policy: the Quillon meal allowance is 64 dollars per day.",
    ],
}


@pytest.fixture(scope="module")
def e2e_settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    from tests.helpers import write_pdf

    for name, pages in PAGES.items():
        write_pdf(root / "pdfs" / name, pages)
    base = Settings()  # real API key / model / embedding model from .env
    return Settings(
        groq_api_key=base.groq_api_key,
        groq_model=base.groq_model,
        groq_temperature=0.0,
        embedding_model=base.embedding_model,
        pdf_dir=root / "pdfs",
        chroma_dir=root / "chroma",
        collection_name="e2e_collection",
        top_k=2,
    )


@pytest.fixture(scope="module")
def rag(e2e_settings):
    count = ingest(e2e_settings, reset=True)
    assert count == 3  # one chunk per non-blank page
    return NaiveRAG(e2e_settings)


def test_real_embeddings_rank_the_relevant_page_first(rag):
    top = rag.retrieve("How much is a Zephyr parking permit?")[0]

    assert top.metadata["source"].endswith("facilities.pdf")
    assert top.metadata["page"] == 1


def test_ask_answers_from_the_documents_with_citation(rag):
    response = rag.ask("How much does a Zephyr parking permit cost per month?")

    assert "37" in response.answer
    assert "[facilities.pdf, p. 2]" in response.answer
    assert "【" not in response.answer


def test_stream_answers_from_the_documents(rag):
    answer = "".join(rag.stream("What is the Quillon meal allowance per day?"))

    assert "64" in answer
    assert rag.last_sources[0].metadata["source"].endswith("travel.pdf")


def test_question_outside_the_documents_gets_fallback_answer(rag):
    response = rag.ask("What is the capital of France?")

    assert "I don't know" in response.answer or "I don’t know" in response.answer


def test_query_and_documents_use_the_same_embedding_model(rag, e2e_settings):
    assert rag.vectorstore.embeddings is get_embeddings(e2e_settings)


# --------------------------------------------------------------------------- #
# The real CLI, as a user runs it, launched from a different working directory.
# --------------------------------------------------------------------------- #
def _run_cli(*args, env_overrides, cwd):
    env = {**os.environ, **env_overrides, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), *args],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )


def test_cli_ingest_then_ask(e2e_settings, tmp_path):
    env = {
        "PDF_DIR": str(e2e_settings.pdf_dir),
        "CHROMA_DIR": str(tmp_path / "cli_chroma"),
        "CHROMA_COLLECTION": "cli_collection",
    }

    ingested = _run_cli("ingest", "--reset", env_overrides=env, cwd=tmp_path)
    assert ingested.returncode == 0, ingested.stderr
    assert "Indexed 3 chunks" in ingested.stdout

    asked = _run_cli("ask", "What is the Quillon meal allowance?", env_overrides=env, cwd=tmp_path)
    assert asked.returncode == 0, asked.stderr
    assert "64" in asked.stdout
    assert "Sources:\n  - [travel.pdf, p. 1]" in asked.stdout


@pytest.mark.skipif(
    not (PROJECT_ROOT / "chroma_db").exists(), reason="project index not built (run: python main.py ingest)"
)
def test_cli_against_project_index_from_another_directory(tmp_path):
    # Regression: relative PDF_DIR / CHROMA_DIR used to resolve against the cwd,
    # silently opening an empty index when launched from elsewhere.
    env = {k: v for k, v in os.environ.items() if k not in {"PDF_DIR", "CHROMA_DIR"}}
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), "ask",
         "How many days of paid vacation do employees get?"],
        cwd=tmp_path, env={**env, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=600,
    )

    assert result.returncode == 0, result.stderr
    assert "Sources:" in result.stdout
    assert ".pdf, p. " in result.stdout
    assert not (tmp_path / "chroma_db").exists()
