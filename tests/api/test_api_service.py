"""Tests for the meta endpoints, app wiring and service-layer building blocks:
GET /, GET /health, OpenAPI, lifespan warm-up, ReadWriteLock, SSE formatting,
SourceChunk and concurrency between queries and ingestion."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.documents import Document

from rag_naive.api.errors import (
    BadRequestError,
    LLMNotConfiguredError,
    NotFoundError,
    ServiceError,
)
from rag_naive.api.schemas import SourceChunk
from rag_naive.api.service import ReadWriteLock, _sse

WAIT = 2.0  # seconds; generous upper bound for thread hand-offs


# --------------------------------------------------------------------------- #
# GET / and GET /health
# --------------------------------------------------------------------------- #
class TestMeta:
    def test_root_redirects_to_swagger_ui(self, client):
        r = client.get("/", follow_redirects=False)

        assert r.status_code == 307
        assert r.headers["location"] == "/docs"

    def test_health_reports_configuration(self, client):
        assert client.get("/health").json() == {
            "status": "ok",
            "llm_configured": True,
            "groq_model": "test-model",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "collection": "test_collection",
            "indexed_chunks": 0,
        }

    def test_health_counts_indexed_chunks(self, indexed):
        assert indexed.get("/health").json()["indexed_chunks"] == 3

    def test_health_without_api_key(self, make_client):
        r = make_client(groq_api_key=None).get("/health")

        assert r.status_code == 200
        assert r.json()["llm_configured"] is False

    def test_unknown_route_is_404(self, client):
        assert client.get("/nope").status_code == 404

    def test_wrong_method_is_405(self, client):
        assert client.get("/query").status_code == 405


class TestOpenAPI:
    @pytest.fixture
    def spec(self, client):
        return client.get("/openapi.json").json()

    def test_lists_every_endpoint(self, spec):
        paths = {path: sorted(ops) for path, ops in spec["paths"].items()}

        assert paths == {
            "/health": ["get"],
            "/ingest": ["post"],
            "/ingest/upload": ["post"],
            "/documents": ["get"],
            "/retrieve": ["post"],
            "/query": ["post"],
            "/query/stream": ["post"],
        }

    def test_upload_is_multipart(self, spec):
        content = spec["paths"]["/ingest/upload"]["post"]["requestBody"]["content"]

        assert list(content) == ["multipart/form-data"]

    def test_stream_documents_event_stream_and_errors(self, spec):
        responses = spec["paths"]["/query/stream"]["post"]["responses"]

        assert "text/event-stream" in responses["200"]["content"]
        assert {"400", "404", "422", "502", "503"} <= set(responses)

    def test_docs_pages_are_served(self, client):
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200


# --------------------------------------------------------------------------- #
# Lifespan warm-up
# --------------------------------------------------------------------------- #
class TestWarmUp:
    def test_builds_pipeline_at_startup(self, make_client):
        client = make_client(warm_up=True)

        assert client.app.state.service._rag is not None

    def test_skipped_when_disabled(self, make_client):
        client = make_client(warm_up=False)

        assert client.app.state.service._rag is None

    def test_missing_api_key_is_logged_not_fatal(self, make_client, caplog):
        with caplog.at_level(logging.WARNING, logger="rag_naive.api.service"):
            client = make_client(warm_up=True, groq_api_key=None)

        assert client.get("/health").status_code == 200
        assert "Question answering disabled" in caplog.text
        assert "GROQ_API_KEY is not set" in caplog.text

    def test_pipeline_is_discarded_after_ingest(self, make_client, seed_pdfs):
        client = make_client(warm_up=True)

        client.post("/ingest", json={})

        assert client.app.state.service._rag is None


# --------------------------------------------------------------------------- #
# Concurrency: queries during ingestion
# --------------------------------------------------------------------------- #
def test_concurrent_queries_during_reset_all_succeed(indexed):
    question = {"question": "What is the 401k match?"}

    def query(_):
        return indexed.post("/query", json=question).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(query, i) for i in range(12)]
        resets = [indexed.post("/ingest", json={"reset": True}).status_code for _ in range(3)]
        statuses = [f.result() for f in futures]

    assert resets == [200, 200, 200]
    assert statuses == [200] * 12
    assert indexed.get("/health").json()["indexed_chunks"] == 3


# --------------------------------------------------------------------------- #
# ReadWriteLock
# --------------------------------------------------------------------------- #
def _hold(lock_cm, entered: threading.Event, release: threading.Event) -> threading.Thread:
    """Start a thread that acquires ``lock_cm`` and holds it until ``release`` is set."""

    def run():
        with lock_cm:
            entered.set()
            release.wait(WAIT)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class TestReadWriteLock:
    def test_readers_share_the_lock(self):
        lock = ReadWriteLock()
        first, second, release = threading.Event(), threading.Event(), threading.Event()

        _hold(lock.read(), first, release)
        assert first.wait(WAIT)
        _hold(lock.read(), second, release)

        assert second.wait(WAIT)  # entered while the first reader still holds it
        release.set()

    def test_writer_waits_for_readers(self):
        lock = ReadWriteLock()
        reader, writer = threading.Event(), threading.Event()
        release_reader, release_writer = threading.Event(), threading.Event()

        _hold(lock.read(), reader, release_reader)
        assert reader.wait(WAIT)
        _hold(lock.write(), writer, release_writer)

        assert not writer.wait(0.2)
        release_reader.set()
        assert writer.wait(WAIT)
        release_writer.set()

    def test_readers_wait_for_writer(self):
        lock = ReadWriteLock()
        writer, reader = threading.Event(), threading.Event()
        release_writer, release_reader = threading.Event(), threading.Event()

        _hold(lock.write(), writer, release_writer)
        assert writer.wait(WAIT)
        _hold(lock.read(), reader, release_reader)

        assert not reader.wait(0.2)
        release_writer.set()
        assert reader.wait(WAIT)
        release_reader.set()

    def test_writers_are_exclusive(self):
        lock = ReadWriteLock()
        first, second = threading.Event(), threading.Event()
        release_first, release_second = threading.Event(), threading.Event()

        _hold(lock.write(), first, release_first)
        assert first.wait(WAIT)
        _hold(lock.write(), second, release_second)

        assert not second.wait(0.2)
        release_first.set()
        assert second.wait(WAIT)
        release_second.set()

    def test_waiting_writer_blocks_new_readers(self):
        """Prevents a steady stream of queries from starving ingestion."""
        lock = ReadWriteLock()
        reader1, writer, reader2 = threading.Event(), threading.Event(), threading.Event()
        release_r1, release_w, release_r2 = threading.Event(), threading.Event(), threading.Event()

        _hold(lock.read(), reader1, release_r1)
        assert reader1.wait(WAIT)
        _hold(lock.write(), writer, release_w)
        deadline = time.monotonic() + WAIT
        while not lock._writers_waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        assert lock._writers_waiting == 1
        _hold(lock.read(), reader2, release_r2)

        assert not reader2.wait(0.2)  # queued behind the waiting writer
        release_r1.set()
        assert writer.wait(WAIT)
        assert not reader2.is_set()
        release_w.set()
        assert reader2.wait(WAIT)
        release_r2.set()

    def test_lock_is_released_when_body_raises(self):
        lock = ReadWriteLock()

        for cm in (lock.read, lock.write):
            with pytest.raises(ValueError):
                with cm():
                    raise ValueError

        with lock.write():  # would deadlock if a release had been skipped
            pass
        assert (lock._readers, lock._writer, lock._writers_waiting) == (0, False, 0)


# --------------------------------------------------------------------------- #
# SSE formatting, SourceChunk, errors
# --------------------------------------------------------------------------- #
class TestSSE:
    def test_format(self):
        assert _sse("token", {"text": "hi"}) == 'event: token\ndata: {"text": "hi"}\n\n'

    def test_keeps_unicode_and_escapes_newlines(self):
        assert _sse("token", {"text": "café\nnext"}) == (
            'event: token\ndata: {"text": "café\\nnext"}\n\n'
        )


class TestSourceChunk:
    def test_from_document(self):
        doc = Document(page_content="text", metadata={"source": r"C:\data\pdfs\a.pdf", "page": 0})

        assert SourceChunk.from_document(doc).model_dump() == {
            "label": "[a.pdf, p. 1]",
            "file": "a.pdf",
            "page": 1,
            "content": "text",
        }

    def test_posix_source_path(self):
        doc = Document(page_content="t", metadata={"source": "data/pdfs/b.pdf", "page": 3})

        assert SourceChunk.from_document(doc).file == "b.pdf"

    def test_without_page(self):
        chunk = SourceChunk.from_document(Document(page_content="t", metadata={"source": "a.pdf"}))

        assert (chunk.label, chunk.page) == ("[a.pdf]", None)

    def test_without_metadata(self):
        chunk = SourceChunk.from_document(Document(page_content="t"))

        assert (chunk.label, chunk.file, chunk.page) == ("[unknown]", "unknown", None)


@pytest.mark.parametrize(
    "error, status",
    [(BadRequestError, 400), (NotFoundError, 404), (LLMNotConfiguredError, 503), (ServiceError, 500)],
)
def test_service_error_status_codes(error, status):
    exc = error("details here")

    assert exc.status_code == status
    assert exc.detail == str(exc) == "details here"
