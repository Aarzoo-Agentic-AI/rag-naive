"""Tests for rag_naive.ingestion.splitter."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_naive.ingestion import split_documents

# 60 distinct words, ~420 characters, so chunk boundaries are easy to verify.
LONG_TEXT = " ".join(f"word{i:02d}" for i in range(60))


def _page(text: str, page: int = 0, source: str = "guide.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_short_document_becomes_a_single_chunk():
    chunks = split_documents([_page("Short page.")], chunk_size=100, chunk_overlap=10)

    assert [c.page_content for c in chunks] == ["Short page."]


def test_long_document_is_split_into_chunks_within_size_limit():
    chunks = split_documents([_page(LONG_TEXT)], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)


def test_chunks_cover_all_of_the_original_text():
    chunks = split_documents([_page(LONG_TEXT)], chunk_size=100, chunk_overlap=20)

    words_in_chunks = {w for c in chunks for w in c.page_content.split()}
    assert words_in_chunks == set(LONG_TEXT.split())


def test_consecutive_chunks_overlap():
    chunks = split_documents([_page(LONG_TEXT)], chunk_size=100, chunk_overlap=30)

    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.page_content.split()) & set(nxt.page_content.split())


def test_zero_overlap_produces_disjoint_chunks():
    chunks = split_documents([_page(LONG_TEXT)], chunk_size=100, chunk_overlap=0)

    for prev, nxt in zip(chunks, chunks[1:]):
        assert not set(prev.page_content.split()) & set(nxt.page_content.split())


def test_start_index_metadata_points_at_chunk_in_page_text():
    chunks = split_documents([_page(LONG_TEXT)], chunk_size=100, chunk_overlap=20)

    for c in chunks:
        start = c.metadata["start_index"]
        assert LONG_TEXT[start : start + len(c.page_content)] == c.page_content


def test_page_metadata_is_copied_to_every_chunk():
    chunks = split_documents(
        [_page(LONG_TEXT, page=4, source="policy.pdf")], chunk_size=100, chunk_overlap=20
    )

    assert all(c.metadata["source"] == "policy.pdf" for c in chunks)
    assert all(c.metadata["page"] == 4 for c in chunks)


def test_chunks_never_span_two_pages():
    pages = [_page("alpha " * 30, page=0), _page("beta " * 30, page=1)]

    chunks = split_documents(pages, chunk_size=80, chunk_overlap=10)

    for c in chunks:
        expected_word = "alpha" if c.metadata["page"] == 0 else "beta"
        assert set(c.page_content.split()) == {expected_word}


def test_empty_input_returns_empty_list():
    assert split_documents([], chunk_size=100, chunk_overlap=10) == []


def test_overlap_larger_than_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        split_documents([_page(LONG_TEXT)], chunk_size=50, chunk_overlap=100)
