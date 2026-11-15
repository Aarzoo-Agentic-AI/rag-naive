"""FastAPI application exposing the naive RAG pipeline over HTTP.

Run:
    python main.py serve                       # http://127.0.0.1:8000
    uvicorn rag_naive.api.app:app --reload     # equivalent, for development

Interactive docs: /docs (Swagger UI), /redoc. OpenAPI schema: /openapi.json
(Postman: Import -> Link -> http://127.0.0.1:8000/openapi.json).

Endpoints are plain ``def`` (not ``async def``) on purpose: embedding, Chroma
and the Groq client are blocking, so FastAPI runs them in its thread pool
instead of blocking the event loop.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from groq import APIError as GroqAPIError

from ..config import Settings
from .errors import ServiceError
from .schemas import (
    DocumentsResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrieveResponse,
    SourceChunk,
)
from .service import RAGService

logger = logging.getLogger(__name__)

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid input"},
    404: {"model": ErrorResponse, "description": "PDF file or folder not found"},
    502: {"model": ErrorResponse, "description": "Groq API error"},
    503: {"model": ErrorResponse, "description": "GROQ_API_KEY not configured"},
}


def create_app(settings: Settings | None = None, warm_up: bool = True) -> FastAPI:
    service = RAGService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up:
            logger.info("Loading embedding model and pipeline...")
            service.warm_up()
        yield

    app = FastAPI(
        title="Naive RAG API",
        version="1.0.0",
        description=(
            "Naive (standard) RAG over PDF files: query encoding -> ChromaDB top-k "
            "similarity retrieval -> Groq LLM generation. No filtering or re-ranking."
        ),
        lifespan=lifespan,
    )
    app.state.service = service

    # ------------------------------------------------------------------ #
    # Error mapping
    # ------------------------------------------------------------------ #
    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(GroqAPIError)
    async def _groq_error(_: Request, exc: GroqAPIError):
        logger.warning("Groq API error: %s", exc)
        return JSONResponse(status_code=502, content={"detail": f"Groq API error: {exc}"})

    def get_service(request: Request) -> RAGService:
        return request.app.state.service

    # ------------------------------------------------------------------ #
    # Meta
    # ------------------------------------------------------------------ #
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(svc: RAGService = Depends(get_service)):
        """Service status, configured models and number of indexed chunks."""
        return svc.health()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    @app.post(
        "/ingest", response_model=IngestResponse, tags=["ingestion"], responses=ERROR_RESPONSES
    )
    def ingest_from_folder(body: IngestRequest, svc: RAGService = Depends(get_service)):
        """Index PDFs already on the server, in PDF_DIR (or a file / sub-folder of it).

        Re-indexing is an upsert. Use `reset: true` after deleting or shortening PDFs
        so their stale chunks are removed.
        """
        return svc.ingest_path(body.path, body.reset)

    @app.post(
        "/ingest/upload",
        response_model=IngestResponse,
        tags=["ingestion"],
        responses=ERROR_RESPONSES,
    )
    def ingest_upload(
        files: list[UploadFile] = File(..., description="One or more PDF files"),
        reset: bool = Form(False, description="Clear the whole collection first"),
        svc: RAGService = Depends(get_service),
    ):
        """Upload PDFs (multipart/form-data, field name `files`), save them to PDF_DIR
        and index them.

        * All-or-nothing: if any file is invalid, none are saved.
        * A file with an existing name replaces the old one.
        * With `reset=true` the collection is cleared first, so **only the uploaded
          files** are indexed afterwards. Call `POST /ingest` to re-index the folder.
        """
        try:
            return svc.ingest_uploads([(f.filename, f.file) for f in files], reset)
        finally:
            for f in files:
                f.file.close()

    @app.get("/documents", response_model=DocumentsResponse, tags=["ingestion"])
    def list_documents(svc: RAGService = Depends(get_service)):
        """PDFs in the index and/or in PDF_DIR, with chunk counts per file."""
        return svc.documents()

    # ------------------------------------------------------------------ #
    # Retrieval + generation
    # ------------------------------------------------------------------ #
    @app.post(
        "/retrieve", response_model=RetrieveResponse, tags=["query"], responses=ERROR_RESPONSES
    )
    def retrieve(body: QueryRequest, svc: RAGService = Depends(get_service)):
        """Retrieval only (steps 1-2): the top-k chunks for a question, with no LLM call."""
        return RetrieveResponse(question=body.question, chunks=svc.retrieve(body.question, body.top_k))

    @app.post("/query", response_model=QueryResponse, tags=["query"], responses=ERROR_RESPONSES)
    def query(body: QueryRequest, svc: RAGService = Depends(get_service)):
        """Full RAG: retrieve the top-k chunks, then generate a cited answer with Groq."""
        result = svc.ask(body.question, body.top_k)
        return QueryResponse(
            question=body.question,
            answer=result.answer,
            sources=[SourceChunk.from_document(d) for d in result.sources],
        )

    @app.post(
        "/query/stream",
        tags=["query"],
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "Server-Sent Events: `sources`, then `token`..., then `done` "
                "(or `error`).",
            },
            **ERROR_RESPONSES,
        },
    )
    def query_stream(body: QueryRequest, svc: RAGService = Depends(get_service)):
        """Full RAG with the answer streamed token by token as Server-Sent Events.

        Events: `sources` (list of chunks), `token` (`{"text": ...}`) repeated,
        then `done`, or `error` (`{"detail": ...}`) if generation fails mid-stream.
        """
        svc.require_llm()  # fail with a proper status code before streaming starts
        return StreamingResponse(
            svc.stream_events(body.question, body.top_k),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
