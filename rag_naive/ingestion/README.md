# Ingestion

Offline indexing. Turns the PDF files into searchable chunks in ChromaDB.

```
PDF files ──► load_pdfs ──► split_documents ──► embed + store (ChromaDB)
            (1 doc / page)   (overlapping chunks)   (deterministic IDs)
```

Run it with `python main.py ingest` before you ask questions. Run it again whenever the PDFs change.

## Files

| File | Responsibility |
|---|---|
| [loader.py](loader.py) | Find PDFs and load each page as a LangChain `Document` |
| [splitter.py](splitter.py) | Split pages into overlapping, fixed-size chunks |
| [indexer.py](indexer.py) | Orchestrate load → split → embed → store; generate chunk IDs |

## Public API

```python
from rag_naive.ingestion import find_pdfs, load_pdfs, split_documents, chunk_id, ingest
```

| Function | Description |
|---|---|
| `find_pdfs(source)` | Returns `[source]` for a file, or every `*.pdf` under a directory: recursive, case-insensitive and sorted. Raises `FileNotFoundError` if the path doesn't exist or contains no PDFs. |
| `load_pdfs(source)` | One `Document` per page, with `source` (file path) and `page` (0-based) metadata. Skips blank pages; the remaining pages keep their original page numbers. |
| `split_documents(docs, chunk_size, chunk_overlap)` | `RecursiveCharacterTextSplitter` with `add_start_index=True`. Chunks never span two pages and inherit the page's metadata. |
| `chunk_id(chunk)` | SHA-256 of `source \| page \| start_index`. |
| `ingest(settings, source=None, embeddings=None, reset=False)` | Runs the full pipeline and returns the number of chunks stored. `source` overrides `settings.pdf_dir`. `reset=True` empties the collection first. |

## Chunk metadata

Every stored chunk carries the metadata the rest of the pipeline relies on:

| Key | Example | Used by |
|---|---|---|
| `source` | `C:\...\data\pdfs\HR_Policy_Handbook_2027.pdf` | Citation labels (file name only) |
| `page` | `3` (0-based; shown as `p. 4`) | Citation labels |
| `start_index` | `812` (character offset in the page) | `chunk_id` |

## Configuration

Read from `.env` through [`Settings`](../config.py):

| Variable | Default | Effect |
|---|---|---|
| `PDF_DIR` | `data/pdfs` | Default folder to index |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Characters shared between neighbouring chunks |
| `CHROMA_DIR` | `chroma_db` | Where the index is persisted |
| `CHROMA_COLLECTION` | `naive_rag` | Collection name (3–512 chars of `[a-zA-Z0-9._-]`) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model (Groq has no embeddings API) |

Relative paths resolve against the project root, not the current directory. If you change `EMBEDDING_MODEL` or the chunk settings, re-index with `--reset`.

## Re-ingesting behaviour

Chunk IDs depend only on a chunk's location (file, page, offset), not on its text. So `ingest` **upserts**: it updates chunks that already exist and adds new ones.

| Change to the PDFs | Plain `ingest` | `ingest --reset` |
|---|---|---|
| Run again, nothing changed | No duplicates | Rebuilt |
| Page text edited | Chunk updated in place | Rebuilt |
| New PDF added | Added | Rebuilt |
| PDF deleted, or a page becomes shorter | **Stale chunks remain** | Clean |

Plain `ingest` never deletes anything, so use `--reset` after removing or shortening documents.

## Limitations

- **No OCR.** Scanned or image-only PDFs produce blank pages, and those pages are skipped.
- **Text only.** Tables and layout are flattened to plain text by `pypdf`.

## Tests

[tests/ingestion/](../../tests/ingestion/) runs against real generated PDFs and a temporary ChromaDB, with fake embeddings:

```
.venv\Scripts\python -m pytest tests/ingestion
```
