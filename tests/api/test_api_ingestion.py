"""Tests for the ingestion endpoints: POST /ingest, POST /ingest/upload, GET /documents."""

from __future__ import annotations

import pytest

from tests.api.conftest import BONUS_PAGES


def _upload(client, *files, reset: bool | None = None):
    """files: (name, bytes) pairs, sent as the multipart field ``files``."""
    data = {} if reset is None else {"reset": str(reset).lower()}
    return client.post(
        "/ingest/upload",
        files=[("files", (name, content, "application/pdf")) for name, content in files],
        data=data,
    )


def _chunks_per_file(client) -> dict[str, int]:
    return {d["file"]: d["indexed_chunks"] for d in client.get("/documents").json()["documents"]}


# --------------------------------------------------------------------------- #
# POST /ingest
# --------------------------------------------------------------------------- #
class TestIngestFolder:
    def test_indexes_every_pdf_in_pdf_dir(self, client, seed_pdfs):
        r = client.post("/ingest", json={})

        assert r.status_code == 200
        assert r.json() == {
            "files": ["benefits.pdf", "bonus.pdf"],
            "chunks_indexed": 3,
            "total_chunks": 3,
            "collection": "test_collection",
            "reset": False,
        }

    def test_body_is_optional_fields_only(self, client, seed_pdfs):
        """An empty JSON object means: all of PDF_DIR, no reset."""
        assert client.post("/ingest", json={}).json()["reset"] is False

    def test_single_file_relative_to_pdf_dir(self, client, seed_pdfs):
        r = client.post("/ingest", json={"path": "bonus.pdf"})

        assert r.status_code == 200
        assert r.json()["files"] == ["bonus.pdf"]
        assert r.json()["chunks_indexed"] == 1

    def test_sub_folder_relative_to_pdf_dir(self, client, pdf_dir, make_pdf):
        make_pdf(pdf_dir / "hr" / "pto.pdf", ["PTO vacation"])
        make_pdf(pdf_dir / "other.pdf", ["bonus"])

        r = client.post("/ingest", json={"path": "hr"})

        assert r.json()["files"] == ["pto.pdf"]

    def test_reingest_is_an_upsert(self, indexed):
        r = indexed.post("/ingest", json={})

        assert r.json()["chunks_indexed"] == 3
        assert r.json()["total_chunks"] == 3  # not 6

    def test_reset_removes_chunks_of_deleted_pdfs(self, indexed, seed_pdfs):
        (seed_pdfs / "bonus.pdf").unlink()

        assert indexed.post("/ingest", json={}).json()["total_chunks"] == 3  # stale chunk stays
        r = indexed.post("/ingest", json={"reset": True})

        assert r.json()["reset"] is True
        assert r.json()["total_chunks"] == 2
        assert "bonus.pdf" not in _chunks_per_file(indexed)

    def test_works_without_groq_api_key(self, make_client, seed_pdfs):
        client = make_client(groq_api_key=None)

        assert client.post("/ingest", json={}).status_code == 200

    @pytest.mark.parametrize("path", ["../outside.pdf", "..", "sub/../../outside.pdf"])
    def test_rejects_relative_path_escaping_pdf_dir(self, client, seed_pdfs, path):
        r = client.post("/ingest", json={"path": path})

        assert r.status_code == 400
        assert "inside PDF_DIR" in r.json()["detail"]

    def test_rejects_absolute_path_outside_pdf_dir(self, client, seed_pdfs, tmp_path, make_pdf):
        outside = make_pdf(tmp_path / "elsewhere" / "secret.pdf", ["salary"])

        r = client.post("/ingest", json={"path": str(outside)})

        assert r.status_code == 400

    def test_missing_file_is_404(self, client, seed_pdfs):
        r = client.post("/ingest", json={"path": "nope.pdf"})

        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    def test_folder_without_pdfs_is_404(self, client, pdf_dir):
        pdf_dir.mkdir()

        r = client.post("/ingest", json={})

        assert r.status_code == 404
        assert "No PDF files found" in r.json()["detail"]

    def test_failed_ingest_writes_nothing(self, client, pdf_dir):
        pdf_dir.mkdir()
        client.post("/ingest", json={})

        assert client.get("/health").json()["indexed_chunks"] == 0

    def test_reset_must_be_boolean(self, client, seed_pdfs):
        assert client.post("/ingest", json={"reset": "maybe"}).status_code == 422


# --------------------------------------------------------------------------- #
# POST /ingest/upload
# --------------------------------------------------------------------------- #
class TestIngestUpload:
    def test_saves_and_indexes_uploaded_pdf(self, client, pdf_dir, pdf_bytes):
        r = _upload(client, ("bonus_plan.pdf", pdf_bytes(BONUS_PAGES)))

        assert r.status_code == 200
        assert r.json() == {
            "files": ["bonus_plan.pdf"],
            "chunks_indexed": 1,
            "total_chunks": 1,
            "collection": "test_collection",
            "reset": False,
        }
        assert (pdf_dir / "bonus_plan.pdf").read_bytes() == pdf_bytes(BONUS_PAGES)

    def test_multiple_files_in_one_request(self, client, pdf_bytes):
        r = _upload(client, ("a.pdf", pdf_bytes(["pto"])), ("b.pdf", pdf_bytes(["bonus", "salary"])))

        assert r.json()["files"] == ["a.pdf", "b.pdf"]
        assert r.json()["chunks_indexed"] == 3

    def test_only_uploaded_files_are_indexed(self, client, seed_pdfs, pdf_bytes):
        """Other PDFs already in PDF_DIR are not picked up by an upload."""
        _upload(client, ("new.pdf", pdf_bytes(["pto"])))

        assert _chunks_per_file(client) == {"benefits.pdf": 0, "bonus.pdf": 0, "new.pdf": 1}

    def test_uploaded_file_becomes_retrievable(self, client, pdf_bytes):
        _upload(client, ("bonus.pdf", pdf_bytes(BONUS_PAGES)))

        r = client.post("/retrieve", json={"question": "bonus"})

        assert r.json()["chunks"][0]["label"] == "[bonus.pdf, p. 1]"

    def test_same_name_replaces_file_and_upserts_chunks(self, client, pdf_dir, pdf_bytes):
        _upload(client, ("doc.pdf", pdf_bytes(["old bonus text"])))
        r = _upload(client, ("doc.pdf", pdf_bytes(["new bonus text"])))

        assert r.json()["total_chunks"] == 1
        hit = client.post("/retrieve", json={"question": "bonus"}).json()["chunks"][0]
        assert "new bonus text" in hit["content"]

    def test_reset_true_leaves_only_the_uploaded_files(self, indexed, pdf_bytes):
        r = _upload(indexed, ("a.pdf", pdf_bytes(["pto"])), ("b.pdf", pdf_bytes(["bonus"])), reset=True)

        assert r.json()["reset"] is True
        assert r.json()["total_chunks"] == 2  # reset applied once, not per file
        assert _chunks_per_file(indexed) == {"a.pdf": 1, "b.pdf": 1, "benefits.pdf": 0, "bonus.pdf": 0}

    def test_reset_defaults_to_false(self, indexed, pdf_bytes):
        r = _upload(indexed, ("new.pdf", pdf_bytes(["pto"])))

        assert r.json()["reset"] is False
        assert r.json()["total_chunks"] == 4

    @pytest.mark.parametrize(
        "client_name", ["../../evil.pdf", "nested/dir/evil.pdf", r"C:\Users\me\evil.pdf"]
    )
    def test_client_supplied_directories_are_stripped(self, client, tmp_path, pdf_dir, pdf_bytes,
                                                      client_name):
        r = _upload(client, (client_name, pdf_bytes(["pto"])))

        assert r.status_code == 200
        assert r.json()["files"] == ["evil.pdf"]
        assert (pdf_dir / "evil.pdf").is_file()
        assert not (tmp_path / "evil.pdf").exists()

    def test_upper_case_extension_is_accepted(self, client, pdf_bytes):
        assert _upload(client, ("REPORT.PDF", pdf_bytes(["pto"]))).status_code == 200

    @pytest.mark.parametrize("name", ["notes.txt", "pdf", "archive.pdf.zip"])
    def test_rejects_non_pdf_file_name(self, client, pdf_bytes, name):
        r = _upload(client, (name, pdf_bytes(["pto"])))

        assert r.status_code == 400
        assert "Only .pdf files are accepted" in r.json()["detail"]

    def test_rejects_content_that_is_not_a_pdf(self, client, pdf_dir):
        r = _upload(client, ("fake.pdf", b"just some text"))

        assert r.status_code == 400
        assert "'fake.pdf' is not a valid PDF file." == r.json()["detail"]

    def test_rejects_file_over_size_limit(self, make_client, pdf_dir):
        client = make_client(max_upload_mb=1)

        r = _upload(client, ("big.pdf", b"%PDF-" + b"0" * (1024 * 1024)))

        assert r.status_code == 400
        assert "exceeds the 1 MB upload limit" in r.json()["detail"]

    def test_file_under_size_limit_is_accepted(self, make_client, pdf_dir, pdf_bytes):
        client = make_client(max_upload_mb=1)

        assert _upload(client, ("small.pdf", pdf_bytes(["pto"]))).status_code == 200

    def test_rejects_duplicate_names_case_insensitively(self, client, pdf_bytes):
        r = _upload(client, ("a.pdf", pdf_bytes(["pto"])), ("A.PDF", pdf_bytes(["bonus"])))

        assert r.status_code == 400
        assert "Duplicate" in r.json()["detail"]

    def test_all_or_nothing_when_one_file_is_invalid(self, indexed, seed_pdfs, pdf_bytes):
        before = {p.name: p.read_bytes() for p in seed_pdfs.iterdir()}

        r = _upload(indexed, ("good.pdf", pdf_bytes(["pto"])), ("bad.pdf", b"not a pdf"), reset=True)

        assert r.status_code == 400
        after = {p.name: p.read_bytes() for p in seed_pdfs.iterdir()}
        assert after == before  # nothing saved, no .part files left behind
        assert indexed.get("/health").json()["indexed_chunks"] == 3  # reset not applied

    def test_no_temp_files_left_after_success(self, client, pdf_dir, pdf_bytes):
        _upload(client, ("a.pdf", pdf_bytes(["pto"])))

        assert sorted(p.name for p in pdf_dir.iterdir()) == ["a.pdf"]

    def test_creates_pdf_dir_if_missing(self, client, pdf_dir, pdf_bytes):
        assert not pdf_dir.exists()

        assert _upload(client, ("a.pdf", pdf_bytes(["pto"]))).status_code == 200
        assert (pdf_dir / "a.pdf").is_file()

    def test_files_field_is_required(self, client):
        assert client.post("/ingest/upload", data={"reset": "false"}).status_code == 422

    def test_works_without_groq_api_key(self, make_client, pdf_bytes):
        client = make_client(groq_api_key=None)

        assert _upload(client, ("a.pdf", pdf_bytes(["pto"]))).status_code == 200


# --------------------------------------------------------------------------- #
# GET /documents
# --------------------------------------------------------------------------- #
class TestDocuments:
    def test_empty_index_and_no_folder(self, client):
        assert client.get("/documents").json() == {
            "collection": "test_collection",
            "total_chunks": 0,
            "documents": [],
        }

    def test_lists_indexed_files_with_chunk_counts(self, indexed):
        assert indexed.get("/documents").json() == {
            "collection": "test_collection",
            "total_chunks": 3,
            "documents": [
                {"file": "benefits.pdf", "indexed_chunks": 2, "on_disk": True},
                {"file": "bonus.pdf", "indexed_chunks": 1, "on_disk": True},
            ],
        }

    def test_pdf_on_disk_but_not_indexed(self, client, seed_pdfs):
        docs = client.get("/documents").json()["documents"]

        assert {"file": "bonus.pdf", "indexed_chunks": 0, "on_disk": True} in docs

    def test_deleted_pdf_with_remaining_chunks(self, indexed, seed_pdfs):
        (seed_pdfs / "bonus.pdf").unlink()

        docs = indexed.get("/documents").json()["documents"]

        assert {"file": "bonus.pdf", "indexed_chunks": 1, "on_disk": False} in docs

    def test_includes_pdfs_in_sub_folders_and_ignores_other_files(self, client, pdf_dir, make_pdf):
        make_pdf(pdf_dir / "hr" / "nested.pdf", ["pto"])
        (pdf_dir / "readme.txt").write_text("not a pdf")

        files = [d["file"] for d in client.get("/documents").json()["documents"]]

        assert files == ["nested.pdf"]
