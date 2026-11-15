# REST API

A FastAPI service that exposes the naive RAG pipeline over HTTP, so you can index PDFs and ask questions from Postman, curl, a web frontend or any other HTTP client.

```
Client ──HTTP──► FastAPI routes (app.py) ──► RAGService (service.py) ──► ingestion / retrieval / generation
```

## Running the server

From the project root, with the virtualenv set up and `.env` configured:

```
.venv\Scripts\python main.py serve                      # http://127.0.0.1:8000
.venv\Scripts\python main.py serve --port 9000
.venv\Scripts\python main.py serve --reload             # dev: auto-reload on code changes
```

Or run uvicorn directly: `uvicorn rag_naive.api.app:app --reload`.

The embedding model and pipeline load at startup, so the first request is fast. Wait for `Application startup complete.` in the log before sending requests.

| URL | What |
|---|---|
| `http://127.0.0.1:8000/docs` | Swagger UI: interactive docs where you can try every endpoint |
| `http://127.0.0.1:8000/redoc` | ReDoc reference docs |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI 3.1 schema |

## Using it from Postman

1. Start the server.
2. In Postman: **Import → Link →** `http://127.0.0.1:8000/openapi.json` → **Import**.
3. Postman creates a collection with every endpoint and its example request bodies.

> **curl on Windows:** the curl examples below use `cmd.exe` syntax (`^` continues a line). In Windows PowerShell 5.1, `curl` is an alias for `Invoke-WebRequest`, which takes different options, so type **`curl.exe`** instead and use a backtick (`` ` ``) to continue a line.

Typical first run:

1. `GET /health`. If `indexed_chunks` is `0`, nothing is indexed yet.
2. `POST /ingest` with `{"reset": true}`, which indexes everything in `PDF_DIR`.
3. `POST /query` with `{"question": "..."}`.

## Endpoints

| Method | Path | Purpose | Needs `GROQ_API_KEY` |
|---|---|---|---|
| GET | [`/health`](#get-health) | Status, models and number of indexed chunks | No |
| POST | [`/ingest`](#post-ingest) | Index PDFs already in `PDF_DIR` | No |
| POST | [`/ingest/upload`](#post-ingestupload) | Upload PDFs, save them to `PDF_DIR` and index them | No |
| GET | [`/documents`](#get-documents) | Files in the index and/or on disk, with chunk counts | No |
| POST | [`/retrieve`](#post-retrieve) | Retrieval only: top-k chunks, no LLM call | No |
| POST | [`/query`](#post-query) | Full RAG: retrieve, then generate a cited answer | **Yes** |
| POST | [`/query/stream`](#post-querystream) | Full RAG with the answer streamed as Server-Sent Events | **Yes** |

Without an API key, the server still starts and every endpoint except `/query` and `/query/stream` works. Those two return `503`.

---

### `GET /health`

```json
{
  "status": "ok",
  "llm_configured": true,
  "groq_model": "openai/gpt-oss-120b",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "collection": "naive_rag",
  "indexed_chunks": 64
}
```

---

### `POST /ingest`

Indexes PDFs that are **already on the server**, in `PDF_DIR` or a file or sub-folder inside it.

**Request** (`Content-Type: application/json`, all fields optional):

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | string | all of `PDF_DIR` | File or sub-folder, **relative to `PDF_DIR`** |
| `reset` | bool | `false` | Clear the whole collection first |

```json
{ "reset": true }
```
```json
{ "path": "HR_Policy_Handbook_2027.pdf" }
```

**Response:**

```json
{
  "files": [
    "Annual_Review_Merit_Bonus_Policy_2027.pdf",
    "Employee_Benefits_Guide_2027.pdf",
    "HR_Policy_Handbook_2027.pdf"
  ],
  "chunks_indexed": 64,
  "total_chunks": 64,
  "collection": "naive_rag",
  "reset": true
}
```

Re-indexing is an **upsert**: running it again doesn't create duplicates, and edited pages are updated. It never deletes anything, though. After you remove or shorten a PDF, send `"reset": true` so its leftover chunks are removed. `GET /documents` shows `on_disk: false` for such files.

---

### `POST /ingest/upload`

Uploads one or more PDFs, saves them to `PDF_DIR` and indexes **only those files**.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `files` | file (repeatable) | One or more `.pdf` files |
| `reset` | bool (`true`/`false`) | Optional. Clear the whole collection first |

In Postman: **Body → form-data**. Add the key `files`, switch its type from *Text* to **File**, and pick your PDFs. Add more `files` rows for more files.

```
curl -X POST http://127.0.0.1:8000/ingest/upload ^
  -F "files=@Travel_Policy.pdf" ^
  -F "files=@Expense_Policy.pdf"
```

The response has the same shape as `POST /ingest`.

**Rules:**

- **All-or-nothing.** Every file is checked before any is saved. If one is invalid, none are saved and the request returns `400`.
- **File checks:**
  - The name must end in `.pdf`.
  - The content must start with the PDF signature `%PDF-`.
  - Each file must be under `MAX_UPLOAD_MB` (default 50).
- **Names are sanitized.** Any directories in the uploaded file name are stripped, so `../../x.pdf` is saved as `x.pdf` inside `PDF_DIR`.
- **Same name replaces.** Uploading a file with an existing name replaces the old file, and its chunks are updated.
- **`reset=true` drops everything else from the index.** Only the uploaded files remain indexed. The other PDFs in `PDF_DIR` stay on disk, so call `POST /ingest` to index them again.

---

### `GET /documents`

```json
{
  "collection": "naive_rag",
  "total_chunks": 66,
  "documents": [
    { "file": "Employee_Benefits_Guide_2027.pdf", "indexed_chunks": 20, "on_disk": true },
    { "file": "Old_Policy.pdf",                   "indexed_chunks": 2,  "on_disk": false }
  ]
}
```

| `indexed_chunks` | `on_disk` | Meaning |
|---|---|---|
| `> 0` | `true` | Normal |
| `0` | `true` | File is in `PDF_DIR` but not indexed yet. Run `POST /ingest`. |
| `> 0` | `false` | File was deleted but its chunks remain. Run `POST /ingest` with `reset: true`. |

---

### `POST /retrieve`

Retrieval only (query encoding plus similarity search), with **no LLM call**. It's useful for checking what the model would see.

**Request** (same body as `/query`):

| Field | Type | Constraints | Description |
|---|---|---|---|
| `question` | string | 1–4000 characters | The question |
| `top_k` | int | 1–50, optional | Chunks to retrieve; defaults to `TOP_K` from `.env` |

```json
{ "question": "How many PTO days do employees get?", "top_k": 2 }
```

**Response:**

```json
{
  "question": "How many PTO days do employees get?",
  "chunks": [
    {
      "label": "[Employee_Benefits_Guide_2027.pdf, p. 6]",
      "file": "Employee_Benefits_Guide_2027.pdf",
      "page": 6,
      "content": "Paid time off: 0–2 years of service: 15 days per year ..."
    }
  ]
}
```

`page` is 1-based. `label` is the exact citation the LLM is told to use.

---

### `POST /query`

Full RAG: retrieves the top-k chunks, then generates an answer with Groq. The request body is the same as `/retrieve`.

```json
{ "question": "Who is eligible for the 401(k) match and what is the match percentage?" }
```

**Response:**

```json
{
  "question": "Who is eligible for the 401(k) match and what is the match percentage?",
  "answer": "All employees are eligible ... starting with their first paycheck. The match is 100 % of the first 4 % ... [Employee_Benefits_Guide_2027.pdf, p. 5].",
  "sources": [
    { "label": "[Employee_Benefits_Guide_2027.pdf, p. 5]", "file": "Employee_Benefits_Guide_2027.pdf", "page": 5, "content": "..." }
  ]
}
```

- **Sources:** `sources` lists **every retrieved chunk**, in ranking order, whether or not the answer cited it.
- **Fallback answer:** if the documents don't contain the answer, the model replies `I don't know based on the provided documents.`
- **Encoding:** answers can contain Unicode characters, such as narrow no-break spaces. The JSON is UTF-8.

---

### `POST /query/stream`

Same as `/query`, but the answer arrives token by token as **Server-Sent Events** (`Content-Type: text/event-stream`). The request body is the same.

Event sequence:

| Event | `data` | When |
|---|---|---|
| `sources` | Array of chunks (same shape as `/retrieve` chunks) | Once, before any tokens |
| `token` | `{"text": "..."}` | Repeated; join the `text` values to get the full answer |
| `done` | `{}` | Generation finished |
| `error` | `{"detail": "..."}` | Generation failed partway; replaces `done` |

Example stream:

```
event: sources
data: [{"label": "[Employee_Benefits_Guide_2027.pdf, p. 3]", "file": "Employee_Benefits_Guide_2027.pdf", "page": 3, "content": "..."}]

event: token
data: {"text": "The HDHP"}

event: token
data: {"text": " with HSA has a deductible of"}

event: done
data: {}
```

- **Postman:** shows each event as it arrives.
- **curl:** use `-N` to turn off buffering:
  ```
  curl -N -X POST http://127.0.0.1:8000/query/stream -H "Content-Type: application/json" -d "{\"question\": \"What is the HSA deductible?\"}"
  ```
- **Errors after streaming starts:** the HTTP status is already `200` by then, so a Groq error is reported as an `error` event. Errors found *before* streaming starts, such as a missing API key or invalid input, return a normal error status.

## Errors

Error bodies always have this shape: `{"detail": "..."}`.

| Status | Cause | Example `detail` |
|---|---|---|
| `400` | Invalid input | `'fake.pdf' is not a valid PDF file.`<br>`'big.pdf' exceeds the 50 MB upload limit.`<br>`'path' must be inside PDF_DIR (...)` |
| `404` | PDF file or folder not found | `PDF source not found: ...\data\pdfs\nope.pdf` |
| `422` | Request body fails validation (empty question, `top_k` out of range, wrong type) | List of field errors (FastAPI format) |
| `502` | Groq API error (bad or retired model, invalid key, rate limit) | `Groq API error: Error code: 404 - ... model_not_found` |
| `503` | `GROQ_API_KEY` not configured (`/query`, `/query/stream`) | `GROQ_API_KEY is not set. ...` |

For a `502` with `model_not_found`, set `GROQ_MODEL` in `.env` to a model your key can use (see the [generation README](../generation/README.md#configuration)), then restart the server.

## Configuration

The API uses the same `.env` as the command line (see [`.env.example`](../../.env.example)):

| Variable | Default | Used for |
|---|---|---|
| `GROQ_API_KEY` | (none) | `/query` and `/query/stream` |
| `GROQ_MODEL`, `GROQ_TEMPERATURE` | `openai/gpt-oss-120b`, `0.0` | Generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Query and document encoding |
| `PDF_DIR` | `data/pdfs` | Where `/ingest` reads from and `/ingest/upload` saves to |
| `CHROMA_DIR`, `CHROMA_COLLECTION` | `chroma_db`, `naive_rag` | Vector index |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `1000`, `200` | Chunking during ingestion |
| `TOP_K` | `4` | Default `top_k` for query endpoints |
| `MAX_UPLOAD_MB` | `50` | Per-file upload limit |

Relative paths resolve against the project root. The command line and the API share the same index, so PDFs indexed with `python main.py ingest` are immediately available to the API, and the reverse.

Settings are read at startup, so restart the server after you edit `.env`.

## Design notes

### Files

| File | Responsibility |
|---|---|
| [app.py](app.py) | FastAPI app factory (`create_app`), routes, mapping errors to HTTP status codes |
| [service.py](service.py) | `RAGService`: one shared pipeline, locking, upload checks, ingestion |
| [schemas.py](schemas.py) | Pydantic request/response models (also generate the OpenAPI docs) |
| [errors.py](errors.py) | Errors that don't depend on HTTP, each carrying its status code |

### Concurrency

- **Endpoints are plain `def`, not `async def`.** Embedding, Chroma and the Groq client all block, so FastAPI runs each request in its thread pool instead of stalling the event loop.
- **Reset needs a rebuild.** `ingest` with `reset` deletes and recreates the Chroma collection, and any store opened before that then fails with `Collection ... does not exist`. So every ingest drops the shared pipeline, which is rebuilt on the next query.
- **A reader/writer lock coordinates them.**
  - Queries share a read lock, so they run in parallel.
  - Ingestion takes the write lock, so it waits for running queries to finish and briefly blocks new ones.
  - A streaming response holds its read lock until the stream ends or the client disconnects.
- **Streaming doesn't share state between requests.** `/query/stream` does its own retrieval and generation instead of using `NaiveRAG.stream()`, because `NaiveRAG.last_sources` is shared and would mix up sources between concurrent users.
- **Empty tokens are dropped.** Reasoning models such as gpt-oss send many empty chunks, and the stream skips them.

### Testing without a server

`create_app(settings, warm_up=False)` builds an app with custom settings, for use with FastAPI's `TestClient`:

```python
from fastapi.testclient import TestClient
from rag_naive.api.app import create_app
from rag_naive.config import Settings

with TestClient(create_app(Settings(pdf_dir=..., chroma_dir=..., collection_name="test"))) as client:
    print(client.get("/health").json())
```

## Security

This API is built for **local / trusted use**. Before exposing it beyond your machine:

- **No authentication.** Anyone who can reach the port can upload files, rebuild the index and spend your Groq credits. The server listens on `127.0.0.1` (this machine only) by default. Don't pass `--host 0.0.0.0` unless you add an API key check (e.g. a FastAPI dependency that checks a header) or put it behind an authenticating reverse proxy.
- **Error messages include full server paths.** That's fine locally; trim them before public use.
- **No rate limiting.** Every `/query` call costs Groq tokens.
- **No CORS headers.** Postman and curl are unaffected. A browser frontend on another origin needs FastAPI's `CORSMiddleware`.
