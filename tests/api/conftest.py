"""Fixtures for the REST API tests.

The real FastAPI app runs in-process via ``TestClient`` against a real temporary
ChromaDB. Only the two external dependencies are replaced:

* the HuggingFace embedding model -> ``KeywordEmbeddings`` (predictable ranking)
* ChatGroq -> ``FakeLLM`` (scripted tokens, optional errors, prompt capture)
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIError as GroqAPIError
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableGenerator

from rag_naive.api.app import create_app

# One line of text per page; words come from tests.helpers.VOCAB.
BENEFITS_PAGES = ["PTO vacation is 15 days per year.", "The 401k match is 5 percent of salary."]
BONUS_PAGES = ["Bonus equals salary times target percent."]


class FakeLLM:
    """Stand-in for ChatGroq, usable both for ``invoke`` and ``stream``.

    * ``chunks``   - content chunks to emit ("" simulates gpt-oss empty chunks)
    * ``error``    - exception to raise ...
    * ``raise_at`` - ... before emitting chunk number ``raise_at``
    * ``prompts``  - messages of every call, for asserting on the context sent
    """

    def __init__(self) -> None:
        self.chunks: list[str] = ["The answer."]
        self.error: Exception | None = None
        self.raise_at = 0
        self.prompts: list[list] = []
        self.runnable = RunnableGenerator(self._generate)

    def _generate(self, inputs):
        for prompt_value in inputs:
            self.prompts.append(prompt_value.to_messages())
        for i, text in enumerate(self.chunks):
            if self.error is not None and i == self.raise_at:
                raise self.error
            yield AIMessageChunk(content=text)
        if self.error is not None and self.raise_at >= len(self.chunks):
            raise self.error

    @property
    def last_context(self) -> str:
        return self.prompts[-1][0].content  # system message holds the context


def groq_error(message: str = "model not found") -> GroqAPIError:
    return GroqAPIError(message, httpx.Request("POST", "https://api.groq.com"), body=None)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def make_client(make_settings, keyword_embeddings, fake_llm, monkeypatch):
    """Factory: ``make_client(**settings_overrides)`` -> started ``TestClient``."""
    monkeypatch.setattr("rag_naive.vectorstore._hf_embeddings", lambda _name: keyword_embeddings)

    def _fake_get_llm(settings):
        settings.require_groq_key()  # keep the real "missing key" behaviour
        return fake_llm.runnable

    monkeypatch.setattr("rag_naive.pipeline.get_llm", _fake_get_llm)

    clients: list[TestClient] = []

    def _make(warm_up: bool = False, **overrides) -> TestClient:
        client = TestClient(create_app(make_settings(**overrides), warm_up=warm_up))
        client.__enter__()  # runs the lifespan (warm-up)
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client(top_k=2)


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """PDF_DIR used by ``make_settings`` (created on demand by the tests)."""
    return tmp_path / "pdfs"


@pytest.fixture
def seed_pdfs(pdf_dir, make_pdf) -> Path:
    """benefits.pdf (2 pages) and bonus.pdf (1 page) in PDF_DIR, not yet indexed."""
    make_pdf(pdf_dir / "benefits.pdf", BENEFITS_PAGES)
    make_pdf(pdf_dir / "bonus.pdf", BONUS_PAGES)
    return pdf_dir


@pytest.fixture
def indexed(client, seed_pdfs) -> TestClient:
    """Client whose collection already contains the seed PDFs (3 chunks)."""
    assert client.post("/ingest", json={}).status_code == 200
    return client


@pytest.fixture
def pdf_bytes(tmp_path, make_pdf):
    """Factory: PDF file content for uploads."""

    def _make(pages: list[str], name: str = "upload.pdf") -> bytes:
        return make_pdf(tmp_path / "upload_src" / name, pages).read_bytes()

    return _make
