"""Request / response models for the REST API (also drive the OpenAPI docs)."""

from __future__ import annotations

from pathlib import PureWindowsPath

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from ..retrieval import citation_label


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
class SourceChunk(BaseModel):
    """A retrieved chunk, as shown to API consumers."""

    label: str = Field(examples=["[Employee_Benefits_Guide_2027.pdf, p. 6]"])
    file: str = Field(examples=["Employee_Benefits_Guide_2027.pdf"])
    page: int | None = Field(None, description="1-based page number", examples=[6])
    content: str

    @classmethod
    def from_document(cls, doc: Document) -> SourceChunk:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        return cls(
            label=citation_label(doc),
            file=PureWindowsPath(source).name if source else "unknown",
            page=page + 1 if isinstance(page, int) else None,
            content=doc.page_content,
        )


# --------------------------------------------------------------------------- #
# Query / retrieve
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
        examples=["How many days of paid vacation do employees get?"],
    )
    top_k: int | None = Field(
        None, ge=1, le=50, description="Chunks to retrieve; defaults to TOP_K from .env"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


class RetrieveResponse(BaseModel):
    question: str
    chunks: list[SourceChunk]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    path: str | None = Field(
        None,
        description="File or sub-folder inside PDF_DIR to index. Omit to index all of PDF_DIR.",
        examples=[None, "HR_Policy_Handbook_2027.pdf"],
    )
    reset: bool = Field(
        False, description="Clear the whole collection before indexing (removes stale chunks)."
    )


class IngestResponse(BaseModel):
    files: list[str] = Field(description="PDF files that were indexed")
    chunks_indexed: int = Field(description="Chunks written by this request")
    total_chunks: int = Field(description="Chunks in the collection after indexing")
    collection: str
    reset: bool


# --------------------------------------------------------------------------- #
# Documents / health
# --------------------------------------------------------------------------- #
class DocumentInfo(BaseModel):
    file: str
    indexed_chunks: int
    on_disk: bool = Field(description="False if the PDF was deleted but its chunks remain")


class DocumentsResponse(BaseModel):
    collection: str
    total_chunks: int
    documents: list[DocumentInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_configured: bool
    groq_model: str
    embedding_model: str
    collection: str
    indexed_chunks: int


class ErrorResponse(BaseModel):
    detail: str
