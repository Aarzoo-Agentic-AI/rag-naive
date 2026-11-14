"""Tests for rag_naive.ingestion.loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_naive.ingestion import find_pdfs, load_pdfs


# --------------------------------------------------------------------------- #
# find_pdfs
# --------------------------------------------------------------------------- #
def test_find_pdfs_accepts_a_single_file(tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "a.pdf", ["x"])

    assert find_pdfs(pdf) == [pdf]


def test_find_pdfs_accepts_str_path(tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "a.pdf", ["x"])

    assert find_pdfs(str(pdf)) == [pdf]


def test_find_pdfs_returns_sorted_pdfs_from_directory(tmp_path, make_pdf):
    b = make_pdf(tmp_path / "b.pdf", ["x"])
    a = make_pdf(tmp_path / "a.pdf", ["x"])

    assert find_pdfs(tmp_path) == [a, b]


def test_find_pdfs_searches_subdirectories(tmp_path, make_pdf):
    top = make_pdf(tmp_path / "top.pdf", ["x"])
    nested = make_pdf(tmp_path / "sub" / "deeper" / "nested.pdf", ["x"])

    assert set(find_pdfs(tmp_path)) == {top, nested}


def test_find_pdfs_matches_extension_case_insensitively(tmp_path, make_pdf):
    upper = make_pdf(tmp_path / "REPORT.PDF", ["x"])

    assert find_pdfs(tmp_path) == [upper]


def test_find_pdfs_ignores_non_pdf_files(tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "a.pdf", ["x"])
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / "pdf").mkdir()  # directory named like the extension

    assert find_pdfs(tmp_path) == [pdf]


def test_find_pdfs_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="PDF source not found"):
        find_pdfs(tmp_path / "does-not-exist")


def test_find_pdfs_raises_for_directory_without_pdfs(tmp_path):
    (tmp_path / "notes.txt").write_text("not a pdf")

    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        find_pdfs(tmp_path)


# --------------------------------------------------------------------------- #
# load_pdfs
# --------------------------------------------------------------------------- #
def test_load_pdfs_returns_one_document_per_page(tmp_path, make_pdf):
    make_pdf(tmp_path / "guide.pdf", ["Page one text.", "Page two text."])

    docs = load_pdfs(tmp_path)

    assert [d.page_content.strip() for d in docs] == ["Page one text.", "Page two text."]


def test_load_pdfs_sets_source_and_zero_based_page_metadata(tmp_path, make_pdf):
    pdf = make_pdf(tmp_path / "guide.pdf", ["First.", "Second."])

    docs = load_pdfs(tmp_path)

    assert [d.metadata["page"] for d in docs] == [0, 1]
    assert all(Path(d.metadata["source"]) == pdf for d in docs)


def test_load_pdfs_skips_blank_pages(tmp_path, make_pdf):
    make_pdf(tmp_path / "guide.pdf", ["Intro.", "", "Appendix."])

    docs = load_pdfs(tmp_path)

    assert [d.page_content.strip() for d in docs] == ["Intro.", "Appendix."]
    # Page numbers still reflect the original PDF, not the filtered position.
    assert [d.metadata["page"] for d in docs] == [0, 2]


def test_load_pdfs_combines_multiple_files_in_sorted_order(tmp_path, make_pdf):
    make_pdf(tmp_path / "b.pdf", ["From B."])
    make_pdf(tmp_path / "a.pdf", ["From A."])

    docs = load_pdfs(tmp_path)

    assert [Path(d.metadata["source"]).name for d in docs] == ["a.pdf", "b.pdf"]


def test_load_pdfs_preserves_special_characters(tmp_path, make_pdf):
    make_pdf(tmp_path / "x.pdf", ["Match (100%) of first 4% \\ max 5%"])

    (doc,) = load_pdfs(tmp_path)

    assert "Match (100%) of first 4% \\ max 5%" in doc.page_content


def test_load_pdfs_returns_empty_list_when_every_page_is_blank(tmp_path, make_pdf):
    make_pdf(tmp_path / "empty.pdf", ["", ""])

    assert load_pdfs(tmp_path) == []


def test_load_pdfs_propagates_missing_source_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdfs(tmp_path / "missing")
