"""Thread-safe service layer between the HTTP routes and the RAG pipeline.

Why a service layer:

* **One shared pipeline.** The embedding model and the Groq client are created
  once and reused across requests.
* **Ingest vs. query safety.** ``ingest(reset=True)`` deletes and recreates the
  Chroma collection, which breaks any store instance opened before it
  ("Collection ... does not exist"). Ingestion therefore takes an exclusive
  lock and then discards the pipeline so it is rebuilt against the new
  collection. Queries share a read lock, so they still run concurrently.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from groq import APIError as GroqAPIError

from ..config import Settings
from ..ingestion import find_pdfs, ingest
from ..pipeline import NaiveRAG, RAGResponse
from ..retrieval import build_retriever, format_docs
from ..vectorstore import get_embeddings, get_vectorstore
from .errors import BadRequestError, LLMNotConfiguredError, NotFoundError
from .schemas import (
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    SourceChunk,
)

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"
_COPY_CHUNK = 1024 * 1024


class ReadWriteLock:
    """Many concurrent readers or one writer; waiting writers block new readers."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if not self._readers:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


def _sse(event: str, data) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class RAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._lock = ReadWriteLock()
        self._build_lock = threading.Lock()
        self._rag: NaiveRAG | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def warm_up(self) -> None:
        """Load the embedding model and build the pipeline at startup so the
        first request is fast. A missing API key is logged, not fatal: ingestion
        and /health still work without it."""
        get_embeddings(self.settings)
        try:
            self._pipeline()
        except LLMNotConfiguredError as exc:
            logger.warning("Question answering disabled: %s", exc.detail)

    def _pipeline(self) -> NaiveRAG:
        if self._rag is None:
            with self._build_lock:
                if self._rag is None:
                    try:
                        self._rag = NaiveRAG(self.settings)
                    except RuntimeError as exc:  # missing GROQ_API_KEY
                        raise LLMNotConfiguredError(str(exc)) from exc
        return self._rag

    @property
    def llm_configured(self) -> bool:
        return bool(self.settings.groq_api_key)

    def require_llm(self) -> None:
        """Raise ``LLMNotConfiguredError`` now, e.g. before a streaming response starts."""
        self._pipeline()

    # ------------------------------------------------------------------ #
    # Query side (shared lock)
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str, top_k: int | None = None) -> list[SourceChunk]:
        """Retrieval only. Uses the vector store directly, so it works without a
        GROQ_API_KEY (the full pipeline cannot be built without one)."""
        with self._lock.read():
            retriever = build_retriever(self._store(), top_k or self.settings.top_k)
            docs = retriever.invoke(question)
        return [SourceChunk.from_document(d) for d in docs]

    def ask(self, question: str, top_k: int | None = None) -> RAGResponse:
        with self._lock.read():
            return self._pipeline().ask(question, top_k)

    def stream_events(self, question: str, top_k: int | None = None) -> Iterator[str]:
        """SSE stream: one ``sources`` event, many ``token`` events, then ``done``.

        Retrieval and generation are done here (not via ``NaiveRAG.stream``)
        because ``NaiveRAG.last_sources`` is shared state and not safe across
        concurrent requests.
        """
        with self._lock.read():
            rag = self._pipeline()
            docs = rag.retrieve(question, top_k)
            yield _sse("sources", [SourceChunk.from_document(d).model_dump() for d in docs])
            try:
                for token in rag.generation_chain.stream(
                    {"question": question, "context": format_docs(docs)}
                ):
                    if token:  # reasoning models (gpt-oss) emit many empty content chunks
                        yield _sse("token", {"text": token})
            except Exception as exc:  # headers are already sent; report in-band
                if isinstance(exc, GroqAPIError):
                    logger.warning("Groq API error while streaming: %s", exc)
                    detail = f"Groq API error: {exc}"
                else:
                    logger.exception("Streaming generation failed")
                    detail = str(exc)
                yield _sse("error", {"detail": detail})
                return
            yield _sse("done", {})

    # ------------------------------------------------------------------ #
    # Ingestion side (exclusive lock)
    # ------------------------------------------------------------------ #
    def _resolve_in_pdf_dir(self, relative: str | None) -> Path:
        base = self.settings.pdf_dir.resolve()
        if not relative:
            return base
        target = (base / relative).resolve()
        if not target.is_relative_to(base):
            raise BadRequestError(f"'path' must be inside PDF_DIR ({base}).")
        return target

    def ingest_path(self, relative: str | None, reset: bool) -> IngestResponse:
        source = self._resolve_in_pdf_dir(relative)
        try:
            files = find_pdfs(source)
        except FileNotFoundError as exc:
            raise NotFoundError(str(exc)) from exc

        with self._lock.write():
            count = ingest(self.settings, source=source, reset=reset)
            self._rag = None  # rebuild against the (possibly recreated) collection
            total = self._count_chunks()
        return IngestResponse(
            files=[f.name for f in files],
            chunks_indexed=count,
            total_chunks=total,
            collection=self.settings.collection_name,
            reset=reset,
        )

    def ingest_uploads(self, uploads: list[tuple[str, BinaryIO]], reset: bool) -> IngestResponse:
        """Validate and save uploaded PDFs into PDF_DIR, then index only those files."""
        if not uploads:
            raise BadRequestError("No files uploaded.")
        names = [self._validate_filename(name) for name, _ in uploads]
        if len(set(n.lower() for n in names)) != len(names):
            raise BadRequestError("Duplicate file names in upload.")

        # All-or-nothing: validate every file into a temp copy before touching PDF_DIR.
        staged: list[Path] = []
        try:
            for name, (_, stream) in zip(names, uploads):
                staged.append(self._stage_pdf(name, stream))
        except BaseException:
            for tmp in staged:
                tmp.unlink(missing_ok=True)
            raise

        with self._lock.write():
            saved = []
            for name, tmp in zip(names, staged):
                target = self.settings.pdf_dir / name
                shutil.move(tmp, target)  # replaces an existing file of the same name
                saved.append(target)
            count = 0
            for i, path in enumerate(saved):
                count += ingest(self.settings, source=path, reset=reset and i == 0)
            self._rag = None
            total = self._count_chunks()
        return IngestResponse(
            files=[p.name for p in saved],
            chunks_indexed=count,
            total_chunks=total,
            collection=self.settings.collection_name,
            reset=reset,
        )

    @staticmethod
    def _validate_filename(filename: str | None) -> str:
        name = PureWindowsPath(filename or "").name  # strips any client-supplied directories
        if not name or not name.lower().endswith(".pdf"):
            raise BadRequestError(f"Only .pdf files are accepted (got '{filename}').")
        return name

    def _stage_pdf(self, name: str, stream: BinaryIO) -> Path:
        """Copy an upload to a temp file in PDF_DIR, checking the PDF signature and
        size limit. Returns the temp path; the caller moves it into place."""
        limit = self.settings.max_upload_mb * 1024 * 1024
        pdf_dir = self.settings.pdf_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=pdf_dir, suffix=".part", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                head = stream.read(len(_PDF_MAGIC))
                if head != _PDF_MAGIC:
                    raise BadRequestError(f"'{name}' is not a valid PDF file.")
                tmp.write(head)
                size = len(head)
                while chunk := stream.read(_COPY_CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise BadRequestError(
                            f"'{name}' exceeds the {self.settings.max_upload_mb} MB upload limit."
                        )
                    tmp.write(chunk)
            except BaseException:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise

        logger.info("Received upload %s (%d bytes)", name, size)
        return tmp_path

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def _store(self):
        return get_vectorstore(self.settings, embeddings=get_embeddings(self.settings))

    def _count_chunks(self) -> int:
        return len(self._store().get(include=[])["ids"])

    def documents(self) -> DocumentsResponse:
        with self._lock.read():
            metadatas = self._store().get(include=["metadatas"])["metadatas"]
        counts = Counter(
            PureWindowsPath(m.get("source", "unknown")).name for m in metadatas if m
        )
        pdf_dir = self.settings.pdf_dir
        on_disk = {p.name for p in pdf_dir.rglob("*") if p.suffix.lower() == ".pdf"} if pdf_dir.is_dir() else set()

        docs = [
            DocumentInfo(file=name, indexed_chunks=counts.get(name, 0), on_disk=name in on_disk)
            for name in sorted(set(counts) | on_disk)
        ]
        return DocumentsResponse(
            collection=self.settings.collection_name,
            total_chunks=sum(counts.values()),
            documents=docs,
        )

    def health(self) -> HealthResponse:
        with self._lock.read():
            total = self._count_chunks()
        return HealthResponse(
            llm_configured=self.llm_configured,
            groq_model=self.settings.groq_model,
            embedding_model=self.settings.embedding_model,
            collection=self.settings.collection_name,
            indexed_chunks=total,
        )
