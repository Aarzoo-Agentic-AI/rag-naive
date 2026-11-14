"""Tests for rag_naive.ingestion.indexer.

``ingest`` runs against a real, temporary ChromaDB with deterministic fake
embeddings, so no model download or network access is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_naive.ingestion import chunk_id, ingest
from rag_naive.vectorstore import get_vectorstore


def _chunk(source="guide.pdf", page=0, start_index=0, text="text") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "start_index": start_index},
    )


def _stored(settings, embeddings) -> dict:
    """Return everything in the collection (ids, documents, metadatas)."""
    return get_vectorstore(settings, embeddings=embeddings).get()


# --------------------------------------------------------------------------- #
# chunk_id
# --------------------------------------------------------------------------- #
def test_chunk_id_is_a_sha256_hex_digest():
    assert re.fullmatch(r"[0-9a-f]{64}", chunk_id(_chunk()))


def test_chunk_id_is_deterministic():
    assert chunk_id(_chunk()) == chunk_id(_chunk())


def test_chunk_id_depends_only_on_location_not_content():
    # An edited chunk at the same location must overwrite the old one.
    assert chunk_id(_chunk(text="old wording")) == chunk_id(_chunk(text="new wording"))


@pytest.mark.parametrize(
    "changed",
    [
        {"source": "other.pdf"},
        {"page": 1},
        {"start_index": 500},
    ],
)
def test_chunk_id_differs_when_location_differs(changed):
    assert chunk_id(_chunk()) != chunk_id(_chunk(**changed))


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
@pytest.fixture
def settings(make_settings, make_pdf):
    s = make_settings(chunk_size=200, chunk_overlap=20)
    make_pdf(s.pdf_dir / "benefits.pdf", ["PTO is 15 days per year.", "401k match is 5%."])
    make_pdf(s.pdf_dir / "bonus.pdf", ["Bonus = salary x target x CPF x multiplier."])
    return s


def test_ingest_returns_number_of_chunks_and_stores_them(settings, fake_embeddings):
    count = ingest(settings, embeddings=fake_embeddings)

    stored = _stored(settings, fake_embeddings)
    assert count == 3
    assert len(stored["ids"]) == 3


def test_ingest_stores_text_and_citation_metadata(settings, fake_embeddings):
    ingest(settings, embeddings=fake_embeddings)

    stored = _stored(settings, fake_embeddings)
    by_text = {d.strip(): m for d, m in zip(stored["documents"], stored["metadatas"])}
    meta = by_text["401k match is 5%."]
    assert Path(meta["source"]).name == "benefits.pdf"
    assert meta["page"] == 1
    assert meta["start_index"] == 0


def test_ingest_uses_deterministic_chunk_ids(settings, fake_embeddings):
    ingest(settings, embeddings=fake_embeddings)

    stored = _stored(settings, fake_embeddings)
    for doc_id, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        assert doc_id == chunk_id(Document(page_content=text, metadata=meta))


def test_reingesting_same_pdfs_does_not_duplicate_chunks(settings, fake_embeddings):
    ingest(settings, embeddings=fake_embeddings)
    ingest(settings, embeddings=fake_embeddings)

    assert len(_stored(settings, fake_embeddings)["ids"]) == 3


def test_reingest_updates_text_of_changed_page(settings, fake_embeddings, make_pdf):
    ingest(settings, embeddings=fake_embeddings)
    make_pdf(settings.pdf_dir / "benefits.pdf", ["PTO is 20 days per year.", "401k match is 5%."])

    ingest(settings, embeddings=fake_embeddings)

    texts = [d.strip() for d in _stored(settings, fake_embeddings)["documents"]]
    assert "PTO is 20 days per year." in texts
    assert "PTO is 15 days per year." not in texts


def test_without_reset_chunks_of_removed_pdfs_remain(settings, fake_embeddings):
    # Documents current (naive) behaviour: ingest only upserts, it never deletes.
    ingest(settings, embeddings=fake_embeddings)
    (settings.pdf_dir / "bonus.pdf").unlink()

    count = ingest(settings, embeddings=fake_embeddings)

    assert count == 2
    assert len(_stored(settings, fake_embeddings)["ids"]) == 3


def test_reset_removes_chunks_of_removed_pdfs(settings, fake_embeddings):
    ingest(settings, embeddings=fake_embeddings)
    (settings.pdf_dir / "bonus.pdf").unlink()

    ingest(settings, embeddings=fake_embeddings, reset=True)

    stored = _stored(settings, fake_embeddings)
    assert len(stored["ids"]) == 2
    assert {Path(m["source"]).name for m in stored["metadatas"]} == {"benefits.pdf"}


def test_ingest_source_argument_overrides_pdf_dir(settings, fake_embeddings, make_pdf, tmp_path):
    other = make_pdf(tmp_path / "elsewhere" / "single.pdf", ["Only this page."])

    count = ingest(settings, source=other, embeddings=fake_embeddings)

    stored = _stored(settings, fake_embeddings)
    assert count == 1
    assert [d.strip() for d in stored["documents"]] == ["Only this page."]


def test_ingest_respects_chunk_size_setting(make_settings, make_pdf, fake_embeddings):
    s = make_settings(chunk_size=40, chunk_overlap=0)
    long_line = " ".join(f"word{i:02d}" for i in range(30))  # ~210 chars, one page
    make_pdf(s.pdf_dir / "long.pdf", [long_line])

    count = ingest(s, embeddings=fake_embeddings)

    stored = _stored(s, fake_embeddings)
    assert count == len(stored["ids"]) > 1
    assert all(len(d) <= 40 for d in stored["documents"])


def test_ingest_with_only_blank_pages_stores_nothing(make_settings, make_pdf, fake_embeddings):
    s = make_settings()
    make_pdf(s.pdf_dir / "blank.pdf", ["", ""])

    assert ingest(s, embeddings=fake_embeddings) == 0
    assert _stored(s, fake_embeddings)["ids"] == []


def test_ingest_missing_source_raises_before_creating_store(make_settings, fake_embeddings):
    s = make_settings()  # pdf_dir does not exist

    with pytest.raises(FileNotFoundError):
        ingest(s, embeddings=fake_embeddings)

    assert not s.chroma_dir.exists()
