"""Tests for rag_naive.pipeline.NaiveRAG (retrieval -> generation wiring).

Uses a real temporary ChromaDB with keyword embeddings for predictable
retrieval, and fake / capturing chat models instead of Groq.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_naive import pipeline
from rag_naive.pipeline import NaiveRAG, RAGResponse
from rag_naive.vectorstore import get_vectorstore

CHUNKS = [
    Document(page_content="PTO vacation: 15 days per year.",
             metadata={"source": r"data\pdfs\benefits.pdf", "page": 5}),
    Document(page_content="Bonus equals salary times target percent.",
             metadata={"source": r"data\pdfs\bonus.pdf", "page": 4}),
    Document(page_content="401k match is 5% of salary.",
             metadata={"source": r"data\pdfs\benefits.pdf", "page": 4}),
]


@pytest.fixture
def settings(make_settings):
    return make_settings(top_k=2)


@pytest.fixture
def store(settings, keyword_embeddings):
    vs = get_vectorstore(settings, embeddings=keyword_embeddings)
    vs.add_documents(CHUNKS, ids=[f"id-{i}" for i in range(len(CHUNKS))])
    return vs


class CapturingLLM:
    """Records the messages sent to the 'LLM' and returns a fixed answer."""

    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer
        self.calls: list[list] = []
        self.runnable = RunnableLambda(self._call)

    def _call(self, prompt_value):
        self.calls.append(prompt_value.to_messages())
        return AIMessage(content=self.answer)


def _rag(settings, store, llm=None) -> NaiveRAG:
    return NaiveRAG(settings, vectorstore=store, llm=llm or FakeListChatModel(responses=["ok"] * 5))


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #
def test_ask_returns_answer_and_retrieved_sources(settings, store):
    rag = _rag(settings, store, FakeListChatModel(responses=["15 days [benefits.pdf, p. 6]."]))

    response = rag.ask("How many PTO vacation days?")

    assert isinstance(response, RAGResponse)
    assert response.answer == "15 days [benefits.pdf, p. 6]."
    assert response.sources[0].page_content == CHUNKS[0].page_content
    assert len(response.sources) == settings.top_k


def test_ask_sends_question_and_labelled_context_to_llm(settings, store):
    llm = CapturingLLM()
    rag = _rag(settings, store, llm.runnable)

    rag.ask("What is the 401k match?")

    (messages,) = llm.calls
    system, human = messages
    assert human.content == "What is the 401k match?"
    assert "[benefits.pdf, p. 5]\n401k match is 5% of salary." in system.content
    assert "data\\pdfs" not in system.content  # file names only, no paths


def test_ask_context_contains_exactly_the_retrieved_chunks(settings, store):
    llm = CapturingLLM()
    rag = _rag(settings, store, llm.runnable)

    response = rag.ask("bonus salary")

    system = llm.calls[0][0].content
    for doc in response.sources:
        assert doc.page_content in system
    not_retrieved = [c for c in CHUNKS if c.page_content not in {d.page_content for d in response.sources}]
    for doc in not_retrieved:
        assert doc.page_content not in system


def test_ask_calls_llm_exactly_once(settings, store):
    llm = CapturingLLM()

    _rag(settings, store, llm.runnable).ask("pto")

    assert len(llm.calls) == 1


def test_ask_normalizes_citation_brackets(settings, store):
    rag = _rag(settings, store, FakeListChatModel(responses=["5%【benefits.pdf, p. 5】"]))

    assert rag.ask("401k match").answer == "5%[benefits.pdf, p. 5]"


def test_ask_on_empty_store_still_answers_with_no_sources(make_settings, keyword_embeddings):
    s = make_settings(collection_name="empty_col")
    empty = get_vectorstore(s, embeddings=keyword_embeddings)
    llm = CapturingLLM("I don't know based on the provided documents.")

    response = NaiveRAG(s, vectorstore=empty, llm=llm.runnable).ask("pto")

    assert response.sources == []
    assert response.answer == "I don't know based on the provided documents."


# --------------------------------------------------------------------------- #
# retrieve
# --------------------------------------------------------------------------- #
def test_retrieve_returns_top_k_from_settings(make_settings, store):
    for k in (1, 3):
        rag = _rag(make_settings(top_k=k), store)
        assert len(rag.retrieve("pto bonus 401k")) == k


def test_retrieve_ranks_most_relevant_chunk_first(settings, store):
    top = _rag(settings, store).retrieve("How is the bonus calculated from salary?")[0]

    assert top.metadata["source"].endswith("bonus.pdf")


# --------------------------------------------------------------------------- #
# stream
# --------------------------------------------------------------------------- #
def test_stream_yields_tokens_that_join_to_the_answer(settings, store):
    answer = "15 days【benefits.pdf, p. 6】"
    rag = _rag(settings, store, FakeListChatModel(responses=[answer]))

    tokens = list(rag.stream("pto"))

    assert len(tokens) > 1
    assert "".join(tokens) == "15 days[benefits.pdf, p. 6]"


def test_stream_exposes_retrieved_sources_via_last_sources(settings, store):
    rag = _rag(settings, store)
    assert rag.last_sources == []

    list(rag.stream("How many PTO vacation days?"))

    assert rag.last_sources[0].page_content == CHUNKS[0].page_content
    assert len(rag.last_sources) == settings.top_k


def test_stream_last_sources_updates_per_question(settings, store):
    rag = _rag(settings, store)

    list(rag.stream("pto vacation"))
    first = rag.last_sources[0].page_content
    list(rag.stream("401k match"))

    assert first == CHUNKS[0].page_content
    assert rag.last_sources[0].page_content == CHUNKS[2].page_content


def test_stream_and_ask_use_the_same_context(settings, store):
    ask_llm, stream_llm = CapturingLLM(), CapturingLLM()

    _rag(settings, store, ask_llm.runnable).ask("bonus salary")
    list(_rag(settings, store, stream_llm.runnable).stream("bonus salary"))

    assert ask_llm.calls == stream_llm.calls


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_injected_components_are_used(settings, store):
    llm = FakeListChatModel(responses=["x"])
    rag = NaiveRAG(settings, vectorstore=store, llm=llm)

    assert rag.vectorstore is store
    assert rag.llm is llm
    assert rag.retriever.search_kwargs == {"k": settings.top_k}


def test_default_llm_is_built_from_settings(monkeypatch, settings, store):
    built = []
    fake = FakeListChatModel(responses=["x"])
    monkeypatch.setattr(pipeline, "get_llm", lambda s: built.append(s) or fake)

    rag = NaiveRAG(settings, vectorstore=store)

    assert built == [settings]
    assert rag.llm is fake


def test_default_vectorstore_is_opened_with_given_embeddings(settings, store, keyword_embeddings):
    # `store` already populated this settings' collection on disk.
    rag = NaiveRAG(settings, llm=FakeListChatModel(responses=["x"]), embeddings=keyword_embeddings)

    assert rag.vectorstore.embeddings is keyword_embeddings
    assert len(rag.vectorstore.get()["ids"]) == len(CHUNKS)


def test_missing_api_key_fails_at_construction(make_settings, store):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        NaiveRAG(make_settings(groq_api_key=None), vectorstore=store)
