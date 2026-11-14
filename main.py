"""Command-line interface for the naive RAG pipeline.

Examples:
    python main.py ingest                      # index every PDF in PDF_DIR
    python main.py ingest path/to/file.pdf --reset
    python main.py ask "What is the refund policy?"
    python main.py chat                        # interactive Q&A loop
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from groq import APIError as GroqAPIError

from rag_naive import NaiveRAG, Settings, ingest
from rag_naive.pipeline import RAGResponse


def _print_sources(sources) -> None:
    if not sources:
        return
    print("\nSources:")
    seen = set()
    for doc in sources:
        page = doc.metadata.get("page")
        key = (doc.metadata.get("source", "unknown"), page)
        if key in seen:
            continue
        seen.add(key)
        page_label = f" (page {page + 1})" if isinstance(page, int) else ""
        print(f"  - {key[0]}{page_label}")


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    count = ingest(settings, source=args.path, reset=args.reset)
    print(f"Indexed {count} chunks into collection '{settings.collection_name}' "
          f"at {settings.chroma_dir}")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    rag = NaiveRAG(settings)
    response: RAGResponse = rag.ask(args.question)
    print(response.answer)
    _print_sources(response.sources)
    return 0


def cmd_chat(_: argparse.Namespace, settings: Settings) -> int:
    rag = NaiveRAG(settings)
    print("Naive RAG chat. Type 'exit' or press Ctrl+C to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0
        print("Assistant: ", end="", flush=True)
        for token in rag.stream(question):
            print(token, end="", flush=True)
        print()
        _print_sources(rag.last_sources)
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Naive RAG over PDF files (LangChain + ChromaDB + Groq)")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable info logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index PDF files into ChromaDB")
    p_ingest.add_argument("path", nargs="?", type=Path, help="PDF file or directory (default: PDF_DIR)")
    p_ingest.add_argument("--reset", action="store_true", help="clear the collection before indexing")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a single question")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive question loop")
    p_chat.set_defaults(func=cmd_chat)
    return parser


def main(argv: list[str] | None = None) -> int:
    # LLM output often contains Unicode (e.g. narrow no-break spaces) that the
    # default Windows console encoding (cp1252) cannot represent.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args, Settings())
    except (FileNotFoundError, RuntimeError, GroqAPIError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
