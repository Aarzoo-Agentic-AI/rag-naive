"""Tests for the query endpoints: POST /retrieve, POST /query, POST /query/stream."""

from __future__ import annotations

import json

import pytest

from tests.api.conftest import groq_error

QUESTION = "What is the 401k match?"


def _events(body: str) -> list[tuple[str, object]]:
    """Parse a Server-Sent Events body into (event, decoded data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def _stream(client, payload: dict):
    with client.stream("POST", "/query/stream", json=payload) as r:
        body = "".join(r.iter_text())
    return r, body


# --------------------------------------------------------------------------- #
# Request validation (shared QueryRequest model)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("endpoint", ["/retrieve", "/query", "/query/stream"])
class TestQueryRequestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": ""},
            {"question": "x" * 4001},
            {"question": 42},
            {"question": QUESTION, "top_k": 0},
            {"question": QUESTION, "top_k": 51},
            {"question": QUESTION, "top_k": "many"},
        ],
        ids=["missing", "empty", "too-long", "not-a-string", "top_k-0", "top_k-51", "top_k-text"],
    )
    def test_invalid_payload_is_422(self, indexed, endpoint, payload):
        r = indexed.post(endpoint, json=payload)

        assert r.status_code == 422
        assert isinstance(r.json()["detail"], list)  # FastAPI validation errors

    @pytest.mark.parametrize("content_type", ["text/event-stream", "text/plain"])
    def test_json_body_with_non_json_content_type_is_422(self, indexed, endpoint, content_type):
        """The body must be sent as application/json, also for /query/stream."""
        r = indexed.post(
            endpoint,
            content=json.dumps({"question": QUESTION}),
            headers={"Content-Type": content_type},
        )

        assert r.status_code == 422

    @pytest.mark.parametrize("top_k", [1, 50])
    def test_top_k_bounds_are_inclusive(self, indexed, endpoint, top_k):
        assert indexed.post(endpoint, json={"question": QUESTION, "top_k": top_k}).status_code == 200

    def test_max_length_question_is_accepted(self, indexed, endpoint):
        assert indexed.post(endpoint, json={"question": "x" * 4000}).status_code == 200


# --------------------------------------------------------------------------- #
# POST /retrieve
# --------------------------------------------------------------------------- #
class TestRetrieve:
    def test_returns_nearest_chunks_with_citation_fields(self, indexed):
        r = indexed.post("/retrieve", json={"question": QUESTION})

        assert r.status_code == 200
        body = r.json()
        assert body["question"] == QUESTION
        assert body["chunks"][0] == {
            "label": "[benefits.pdf, p. 2]",
            "file": "benefits.pdf",
            "page": 2,
            "content": "The 401k match is 5 percent of salary.",
        }

    def test_uses_top_k_from_settings_by_default(self, indexed):
        assert len(indexed.post("/retrieve", json={"question": QUESTION}).json()["chunks"]) == 2

    def test_top_k_overrides_settings(self, indexed):
        r = indexed.post("/retrieve", json={"question": QUESTION, "top_k": 1})

        assert len(r.json()["chunks"]) == 1

    def test_top_k_larger_than_index_returns_everything(self, indexed):
        r = indexed.post("/retrieve", json={"question": QUESTION, "top_k": 50})

        assert len(r.json()["chunks"]) == 3

    def test_does_not_call_the_llm(self, indexed, fake_llm):
        indexed.post("/retrieve", json={"question": QUESTION})

        assert fake_llm.prompts == []

    def test_encodes_the_question_with_the_embedding_model(self, indexed, keyword_embeddings):
        indexed.post("/retrieve", json={"question": QUESTION})

        assert keyword_embeddings.query_calls[-1] == QUESTION

    def test_works_without_groq_api_key(self, make_client, seed_pdfs):
        client = make_client(groq_api_key=None)
        client.post("/ingest", json={})

        r = client.post("/retrieve", json={"question": QUESTION})

        assert r.status_code == 200
        assert r.json()["chunks"]

    def test_empty_index_returns_no_chunks(self, client):
        assert client.post("/retrieve", json={"question": QUESTION}).json()["chunks"] == []


# --------------------------------------------------------------------------- #
# POST /query
# --------------------------------------------------------------------------- #
class TestQuery:
    def test_returns_answer_and_sources(self, indexed, fake_llm):
        fake_llm.chunks = ["The match is ", "5% [benefits.pdf, p. 2]."]

        r = indexed.post("/query", json={"question": QUESTION})

        assert r.status_code == 200
        body = r.json()
        assert body["question"] == QUESTION
        assert body["answer"] == "The match is 5% [benefits.pdf, p. 2]."
        assert [s["label"] for s in body["sources"]][0] == "[benefits.pdf, p. 2]"
        assert len(body["sources"]) == 2

    def test_retrieved_chunks_are_sent_to_the_llm_with_labels(self, indexed, fake_llm):
        indexed.post("/query", json={"question": QUESTION})

        context = fake_llm.last_context
        assert "[benefits.pdf, p. 2]\nThe 401k match is 5 percent of salary." in context
        assert fake_llm.prompts[-1][-1].content == QUESTION

    def test_sources_match_the_context_given_to_the_llm(self, indexed, fake_llm):
        sources = indexed.post("/query", json={"question": QUESTION}).json()["sources"]

        for source in sources:
            assert source["label"] in fake_llm.last_context

    def test_top_k_overrides_settings(self, indexed, fake_llm):
        r = indexed.post("/query", json={"question": QUESTION, "top_k": 1})

        assert len(r.json()["sources"]) == 1
        assert "bonus.pdf" not in fake_llm.last_context

    def test_normalizes_full_width_citation_brackets(self, indexed, fake_llm):
        fake_llm.chunks = ["5% 【benefits.pdf, p. 2】"]

        assert indexed.post("/query", json={"question": QUESTION}).json()["answer"] == (
            "5% [benefits.pdf, p. 2]"
        )

    def test_missing_api_key_is_503(self, make_client, seed_pdfs):
        client = make_client(groq_api_key=None)
        client.post("/ingest", json={})

        r = client.post("/query", json={"question": QUESTION})

        assert r.status_code == 503
        assert "GROQ_API_KEY is not set" in r.json()["detail"]

    def test_groq_api_error_is_502_with_groq_message(self, indexed, fake_llm):
        fake_llm.error = groq_error("model `x` does not exist")

        r = indexed.post("/query", json={"question": QUESTION})

        assert r.status_code == 502
        assert r.json() == {"detail": "Groq API error: model `x` does not exist"}

    def test_still_works_after_ingest_with_reset(self, indexed):
        """reset_collection() invalidates open Chroma handles; the service must rebuild."""
        assert indexed.post("/query", json={"question": QUESTION}).status_code == 200
        assert indexed.post("/ingest", json={"reset": True}).status_code == 200

        r = indexed.post("/query", json={"question": QUESTION})

        assert r.status_code == 200
        assert r.json()["sources"]

    def test_sees_documents_ingested_after_startup(self, client, seed_pdfs):
        assert client.post("/query", json={"question": QUESTION}).json()["sources"] == []

        client.post("/ingest", json={})

        assert client.post("/query", json={"question": QUESTION}).json()["sources"]


# --------------------------------------------------------------------------- #
# POST /query/stream
# --------------------------------------------------------------------------- #
class TestQueryStream:
    def test_event_order_sources_tokens_done(self, indexed, fake_llm):
        fake_llm.chunks = ["The match ", "is 5%."]

        r, body = _stream(indexed, {"question": QUESTION})

        events = _events(body)
        assert r.status_code == 200
        assert [name for name, _ in events] == ["sources", "token", "token", "done"]
        assert events[-1] == ("done", {})

    def test_sources_event_has_chunk_fields(self, indexed):
        _, body = _stream(indexed, {"question": QUESTION})

        name, sources = _events(body)[0]
        assert name == "sources"
        assert len(sources) == 2
        assert sources[0] == {
            "label": "[benefits.pdf, p. 2]",
            "file": "benefits.pdf",
            "page": 2,
            "content": "The 401k match is 5 percent of salary.",
        }

    def test_tokens_concatenate_to_the_answer(self, indexed, fake_llm):
        fake_llm.chunks = ["Employees ", "get a ", "5% match."]

        _, body = _stream(indexed, {"question": QUESTION})

        tokens = [data["text"] for name, data in _events(body) if name == "token"]
        assert "".join(tokens) == "Employees get a 5% match."

    def test_empty_tokens_are_not_sent(self, indexed, fake_llm):
        fake_llm.chunks = ["", "Hi", "", "", " there", ""]

        _, body = _stream(indexed, {"question": QUESTION})

        assert [d["text"] for n, d in _events(body) if n == "token"] == ["Hi", " there"]

    def test_citation_brackets_are_normalized_while_streaming(self, indexed, fake_llm):
        fake_llm.chunks = ["5% 【benefits.pdf", ", p. 2】"]

        _, body = _stream(indexed, {"question": QUESTION})

        assert "".join(d["text"] for n, d in _events(body) if n == "token") == "5% [benefits.pdf, p. 2]"

    def test_non_ascii_text_is_sent_unescaped(self, indexed, fake_llm):
        fake_llm.chunks = ["5 % — café"]

        _, body = _stream(indexed, {"question": QUESTION})

        assert "5 % — café" in body

    def test_response_headers(self, indexed):
        r, _ = _stream(indexed, {"question": QUESTION})

        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["x-accel-buffering"] == "no"

    def test_top_k_overrides_settings(self, indexed):
        _, body = _stream(indexed, {"question": QUESTION, "top_k": 1})

        assert len(_events(body)[0][1]) == 1

    def test_missing_api_key_is_503_before_streaming(self, make_client, seed_pdfs):
        client = make_client(groq_api_key=None)

        r, body = _stream(client, {"question": QUESTION})

        assert r.status_code == 503
        assert r.headers["content-type"] == "application/json"
        assert "GROQ_API_KEY is not set" in json.loads(body)["detail"]

    def test_groq_error_mid_stream_becomes_error_event(self, indexed, fake_llm):
        fake_llm.chunks = ["Partial ", "answer"]
        fake_llm.error, fake_llm.raise_at = groq_error("rate limit reached"), 1

        r, body = _stream(indexed, {"question": QUESTION})

        events = _events(body)
        assert r.status_code == 200  # headers were already sent
        assert [name for name, _ in events] == ["sources", "token", "error"]
        assert events[-1][1] == {"detail": "Groq API error: rate limit reached"}

    def test_groq_error_before_first_token(self, indexed, fake_llm):
        fake_llm.error = groq_error("model not found")

        _, body = _stream(indexed, {"question": QUESTION})

        assert [name for name, _ in _events(body)] == ["sources", "error"]

    def test_unexpected_error_becomes_error_event(self, indexed, fake_llm):
        fake_llm.error, fake_llm.raise_at = RuntimeError("boom"), 1

        _, body = _stream(indexed, {"question": QUESTION})

        assert _events(body)[-1] == ("error", {"detail": "boom"})

    def test_each_stream_reports_its_own_sources(self, indexed):
        """Sources come from the request itself, not from shared NaiveRAG.last_sources."""
        _, pto = _stream(indexed, {"question": "pto vacation", "top_k": 1})
        _, bonus = _stream(indexed, {"question": "bonus", "top_k": 1})

        assert _events(pto)[0][1][0]["label"] == "[benefits.pdf, p. 1]"
        assert _events(bonus)[0][1][0]["label"] == "[bonus.pdf, p. 1]"
