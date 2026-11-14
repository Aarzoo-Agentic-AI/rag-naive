# Retrieval

Steps 1 and 2 of the naive RAG query path:

1. **Query encoding.** The question is embedded with the **same** model that embedded the chunks during ingestion.
2. **Similarity search.** ChromaDB returns the `top_k` nearest chunks by cosine distance.

```
question ──► embed_query ──► Chroma similarity search ──► top-k Documents ──► format_docs ──► context string
```

This is deliberately *naive* RAG: there's no query rewriting, metadata filtering, MMR, score threshold or re-ranking.

## Files

| File | Responsibility |
|---|---|
| [retriever.py](retriever.py) | Build the retriever; turn retrieved chunks into labelled prompt context |

## Public API

```python
from rag_naive.retrieval import build_retriever, citation_label, format_docs
```

| Function | Description |
|---|---|
| `build_retriever(vectorstore, top_k)` | `vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})`. Returns at most `top_k` documents, ranked from most to least similar. |
| `citation_label(doc)` | `[File.pdf, p. N]`: file name only (Windows or POSIX paths) and a 1-based page number. Leaves out the page if it's missing or not a whole number, and uses `unknown` if there's no source. |
| `format_docs(docs)` | Joins the chunks, each preceded by its label, with a blank line between chunks. Keeps retrieval order and passes the text through unchanged. |

## Context format

`format_docs` output is what the LLM sees as `{context}`:

```
[Employee_Benefits_Guide_2027.pdf, p. 6]
Paid time off: 0–2 years of service: 15 days per year ...

[HR_Policy_Handbook_2027.pdf, p. 4]
Employees accrue PTO each pay period ...
```

Passages are **not numbered** on purpose. When they were numbered (`[1]`, `[2]`), the model cited them as `【1, page 6】`, which means nothing to a reader. The label format is shared with the generation prompt and the CLI's `Sources:` list. A test checks that the label matches the example in [`SYSTEM_PROMPT`](../generation/prompts.py), so the two can't drift apart.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `TOP_K` | `4` | Number of chunks passed to the LLM |

The distance metric (cosine) and the embedding model are set where the vector store is created, in [vectorstore.py](../vectorstore.py).

## Known limitation

The whole question is encoded as **one vector**. A question that asks two unrelated things (for example "bonus rules *and* PTO rollover") tends to retrieve chunks for only the dominant topic, so the answer covers just half of it. You can work around this by:

- asking one thing per question
- raising `TOP_K`

Proper fixes are query decomposition, hybrid keyword + vector search or re-ranking. Those belong to more advanced RAG variants, not this one.

## Tests

[tests/retrieval/](../../tests/retrieval/) uses a temporary ChromaDB with keyword-based fake embeddings, so the expected ranking is predictable:

```
.venv\Scripts\python -m pytest tests/retrieval
```
