"""Runtime configuration loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load <project root>/.env regardless of the current working directory.
# Real environment variables take precedence over values in the file.
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _project_path(value: str) -> Path:
    """Resolve relative paths against the project root, not the current directory,
    so the CLI finds the same PDFs / index no matter where it is launched from."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """All tunable parameters of the naive RAG pipeline."""

    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-120b"))
    groq_temperature: float = field(default_factory=lambda: float(_env("GROQ_TEMPERATURE", "0.0")))

    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    pdf_dir: Path = field(default_factory=lambda: _project_path(_env("PDF_DIR", "data/pdfs")))
    chroma_dir: Path = field(default_factory=lambda: _project_path(_env("CHROMA_DIR", "chroma_db")))
    collection_name: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "naive_rag"))

    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "200")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))

    # REST API (rag_naive.api)
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "50")))

    def require_groq_key(self) -> str:
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.groq_api_key
