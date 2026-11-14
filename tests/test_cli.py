"""Tests for the command-line interface in main.py.

The pipeline is replaced with fakes; these tests cover argument parsing,
output formatting, the chat loop and error handling.
"""

from __future__ import annotations

from pathlib import Path

import groq
import httpx
import pytest
from langchain_core.documents import Document

import main
from rag_naive.pipeline import RAGResponse

SOURCES = [
    Document(page_content="a", metadata={"source": r"data\pdfs\benefits.pdf", "page": 5}),
    Document(page_content="b", metadata={"source": r"data\pdfs\benefits.pdf", "page": 5}),  # dup
    Document(page_content="c", metadata={"source": r"data\pdfs\bonus.pdf", "page": 0}),
]


class FakeRAG:
    """Stands in for NaiveRAG; records questions."""

    instances: list[FakeRAG] = []

    def __init__(self, settings) -> None:
        self.settings = settings
        self.questions: list[str] = []
        self.last_sources: list[Document] = []
        FakeRAG.instances.append(self)

    def ask(self, question: str) -> RAGResponse:
        self.questions.append(question)
        return RAGResponse(answer=f"Answer to: {question}", sources=SOURCES)

    def stream(self, question: str):
        self.questions.append(question)
        self.last_sources = SOURCES
        yield from ["Streamed ", "answer."]


@pytest.fixture
def settings(make_settings, monkeypatch):
    s = make_settings()
    monkeypatch.setattr(main, "Settings", lambda: s)
    return s


@pytest.fixture
def fake_rag(monkeypatch, settings):
    FakeRAG.instances = []
    monkeypatch.setattr(main, "NaiveRAG", FakeRAG)
    return FakeRAG


@pytest.fixture
def feed_input(monkeypatch):
    """Simulate interactive input; raises EOFError once lines run out."""

    def _feed(*lines: str):
        it = iter(lines)

        def fake_input(prompt=""):
            print(prompt, end="")
            try:
                return next(it)
            except StopIteration:
                raise EOFError from None

        monkeypatch.setattr("builtins.input", fake_input)

    return _feed


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def test_command_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        main.main([])

    assert exc.value.code == 2


def test_ask_requires_a_question():
    with pytest.raises(SystemExit) as exc:
        main.main(["ask"])

    assert exc.value.code == 2


def test_parser_ingest_defaults():
    args = main.build_parser().parse_args(["ingest"])

    assert args.path is None
    assert args.reset is False


def test_parser_ingest_with_path_and_reset():
    args = main.build_parser().parse_args(["ingest", "docs/file.pdf", "--reset"])

    assert args.path == Path("docs/file.pdf")
    assert args.reset is True


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def test_ingest_command_calls_ingest_and_reports_count(monkeypatch, settings, capsys):
    calls = []
    monkeypatch.setattr(main, "ingest", lambda s, source, reset: calls.append((s, source, reset)) or 42)

    code = main.main(["ingest", "some.pdf", "--reset"])

    assert code == 0
    assert calls == [(settings, Path("some.pdf"), True)]
    out = capsys.readouterr().out
    assert "Indexed 42 chunks" in out
    assert "test_collection" in out


def test_ingest_command_without_path_uses_default_source(monkeypatch, settings):
    calls = []
    monkeypatch.setattr(main, "ingest", lambda s, source, reset: calls.append(source) or 0)

    main.main(["ingest"])

    assert calls == [None]


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #
def test_ask_prints_answer_and_deduplicated_sources(fake_rag, capsys):
    code = main.main(["ask", "How many PTO days?"])

    assert code == 0
    assert fake_rag.instances[0].questions == ["How many PTO days?"]
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "Answer to: How many PTO days?",
        "",
        "Sources:",
        "  - [benefits.pdf, p. 6]",
        "  - [bonus.pdf, p. 1]",
    ]


def test_ask_passes_settings_to_pipeline(fake_rag, settings):
    main.main(["ask", "q"])

    assert fake_rag.instances[0].settings is settings


def test_print_sources_prints_nothing_without_sources(capsys):
    main._print_sources([])

    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
def test_chat_streams_answer_and_sources_then_exits(fake_rag, feed_input, capsys):
    feed_input("What is PTO?", "exit")

    code = main.main(["chat"])

    assert code == 0
    assert fake_rag.instances[0].questions == ["What is PTO?"]
    out = capsys.readouterr().out
    assert "Assistant: Streamed answer." in out
    assert "  - [benefits.pdf, p. 6]" in out


def test_chat_builds_pipeline_once_for_many_questions(fake_rag, feed_input):
    feed_input("q1", "q2", "q3", "quit")

    main.main(["chat"])

    assert len(fake_rag.instances) == 1
    assert fake_rag.instances[0].questions == ["q1", "q2", "q3"]


def test_chat_skips_blank_lines_and_strips_whitespace(fake_rag, feed_input):
    feed_input("", "   ", "  padded question  ", "exit")

    main.main(["chat"])

    assert fake_rag.instances[0].questions == ["padded question"]


@pytest.mark.parametrize("command", ["exit", "quit", "EXIT", "Quit"])
def test_chat_exit_commands_are_case_insensitive(fake_rag, feed_input, command):
    feed_input(command)

    assert main.main(["chat"]) == 0
    assert fake_rag.instances[0].questions == []


def test_chat_exits_cleanly_on_end_of_input(fake_rag, feed_input):
    feed_input("q1")  # then EOF (Ctrl+Z / Ctrl+D)

    assert main.main(["chat"]) == 0


def test_chat_exits_cleanly_on_ctrl_c(fake_rag, monkeypatch):
    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)

    assert main.main(["chat"]) == 0


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #
def _raise(exc):
    def _f(*args, **kwargs):
        raise exc

    return _f


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("No PDF files found in: data/pdfs"),
        RuntimeError("GROQ_API_KEY is not set."),
        groq.APIError("model not found", httpx.Request("POST", "https://api.groq.com"), body=None),
    ],
    ids=["missing-pdfs", "missing-key", "groq-api-error"],
)
def test_known_errors_print_message_and_exit_1(monkeypatch, settings, capsys, error):
    monkeypatch.setattr(main, "NaiveRAG", _raise(error))

    code = main.main(["ask", "q"])

    assert code == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == f"Error: {error}"
    assert "Traceback" not in captured.err


def test_ingest_missing_folder_is_reported_as_error(monkeypatch, settings, capsys):
    monkeypatch.setattr(main, "ingest", _raise(FileNotFoundError("PDF source not found: x")))

    assert main.main(["ingest"]) == 1
    assert "Error: PDF source not found: x" in capsys.readouterr().err


def test_unexpected_errors_are_not_swallowed(monkeypatch, settings):
    monkeypatch.setattr(main, "NaiveRAG", _raise(ValueError("bug")))

    with pytest.raises(ValueError, match="bug"):
        main.main(["ask", "q"])
