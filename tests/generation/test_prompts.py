"""Tests for rag_naive.generation.prompts."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from rag_naive.generation.prompts import RAG_PROMPT, SYSTEM_PROMPT


def test_prompt_declares_exactly_context_and_question_variables():
    assert set(RAG_PROMPT.input_variables) == {"context", "question"}


def test_prompt_renders_system_then_human_message():
    messages = RAG_PROMPT.invoke(
        {"context": "CTX-BODY", "question": "What is PTO?"}
    ).to_messages()

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_context_goes_into_system_message_and_question_into_human_message():
    messages = RAG_PROMPT.invoke(
        {"context": "CTX-BODY", "question": "What is PTO?"}
    ).to_messages()

    assert "CTX-BODY" in messages[0].content
    assert "What is PTO?" not in messages[0].content
    assert messages[1].content == "What is PTO?"


def test_system_prompt_restricts_answers_to_context():
    assert "ONLY the provided context" in SYSTEM_PROMPT


def test_system_prompt_defines_the_fallback_answer():
    assert "I don't know based on the provided documents." in SYSTEM_PROMPT


def test_system_prompt_specifies_citation_label_format():
    # The example label must match what retrieval.format_docs produces.
    assert "[Employee_Benefits_Guide_2027.pdf, p. 6]" in SYSTEM_PROMPT
    assert "copying that label exactly" in SYSTEM_PROMPT


def test_context_with_curly_braces_is_inserted_literally():
    # PDF text can contain "{...}"; it must not be treated as a template variable.
    messages = RAG_PROMPT.invoke(
        {"context": "formula {salary} x {target}", "question": "q"}
    ).to_messages()

    assert "formula {salary} x {target}" in messages[0].content
