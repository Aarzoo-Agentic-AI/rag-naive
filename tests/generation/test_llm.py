"""Tests for rag_naive.generation.llm.

The real Groq model is never called: chains are exercised with LangChain's
fake chat models, and get_llm is only checked for how it configures ChatGroq.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_naive.generation import build_generation_chain, get_llm
from rag_naive.generation.llm import _normalize_stream, normalize_citations

INPUT = {"question": "How many PTO days?", "context": "[Guide.pdf, p. 6]\n15 days."}


# --------------------------------------------------------------------------- #
# normalize_citations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("15 days【Guide.pdf, p. 6】.", "15 days[Guide.pdf, p. 6]."),
        ("a【x】 b【y】", "a[x] b[y]"),
        ("【", "["),
        ("】", "]"),
    ],
)
def test_normalize_citations_converts_lenticular_brackets(raw, expected):
    assert normalize_citations(raw) == expected


@pytest.mark.parametrize(
    "text",
    ["", "plain text", "already [Guide.pdf, p. 6]", "unicode – 30 %‑50 % ✓"],
)
def test_normalize_citations_leaves_other_text_unchanged(text):
    assert normalize_citations(text) == text


def test_normalize_stream_converts_each_chunk_and_preserves_chunking():
    chunks = ["Bonus", " 5%", "【", "Policy.pdf", ", p. 5", "】", "."]

    out = list(_normalize_stream(iter(chunks)))

    assert out == ["Bonus", " 5%", "[", "Policy.pdf", ", p. 5", "]", "."]


def test_normalize_stream_is_lazy():
    def source():
        yield "first"
        raise AssertionError("second chunk must not be pulled yet")

    stream = _normalize_stream(source())

    assert next(stream) == "first"


# --------------------------------------------------------------------------- #
# build_generation_chain
# --------------------------------------------------------------------------- #
def test_chain_invoke_returns_llm_answer_as_string():
    chain = build_generation_chain(FakeListChatModel(responses=["15 days [Guide.pdf, p. 6]."]))

    assert chain.invoke(INPUT) == "15 days [Guide.pdf, p. 6]."


def test_chain_invoke_normalizes_citation_brackets():
    chain = build_generation_chain(FakeListChatModel(responses=["15 days【Guide.pdf, p. 6】."]))

    assert chain.invoke(INPUT) == "15 days[Guide.pdf, p. 6]."


def test_chain_stream_yields_multiple_chunks_and_normalizes_them():
    answer = "15 days【Guide.pdf, p. 6】."
    chain = build_generation_chain(FakeListChatModel(responses=[answer]))

    chunks = list(chain.stream(INPUT))

    # FakeListChatModel streams one character at a time; the final generator
    # step must not buffer the whole answer into a single chunk.
    assert len(chunks) == len(answer)
    assert "".join(chunks) == "15 days[Guide.pdf, p. 6]."


def test_chain_sends_rendered_prompt_to_llm():
    captured = {}

    def fake_llm(prompt_value):
        captured["messages"] = prompt_value.to_messages()
        return AIMessage(content="ok")

    chain = build_generation_chain(RunnableLambda(fake_llm))

    assert chain.invoke(INPUT) == "ok"
    system, human = captured["messages"]
    assert INPUT["context"] in system.content
    assert human.content == INPUT["question"]


def test_chain_requires_question_and_context():
    chain = build_generation_chain(FakeListChatModel(responses=["unused"]))

    with pytest.raises(KeyError):
        chain.invoke({"question": "missing context"})


# --------------------------------------------------------------------------- #
# get_llm
# --------------------------------------------------------------------------- #
def test_get_llm_configures_chatgroq_from_settings(make_settings):
    from langchain_groq import ChatGroq

    settings = make_settings(groq_model="openai/gpt-oss-120b", groq_temperature=0.3)

    llm = get_llm(settings)

    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "openai/gpt-oss-120b"
    assert llm.temperature == 0.3
    assert llm.groq_api_key.get_secret_value() == "test-key"


@pytest.mark.parametrize("missing_key", [None, ""])
def test_get_llm_raises_clear_error_without_api_key(make_settings, missing_key):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        get_llm(make_settings(groq_api_key=missing_key))
