# Generation

Step 3 of the naive RAG query path. The retrieved context and the question go into a single prompt, and the Groq LLM's answer is returned directly. The answer isn't verified, filtered or re-ranked.

```
{question, context} ──► RAG_PROMPT ──► ChatGroq ──► StrOutputParser ──► normalize citations ──► answer
```

## Files

| File | Responsibility |
|---|---|
| [prompts.py](prompts.py) | System and user prompt template |
| [llm.py](llm.py) | Groq chat model factory and the generation chain |

## Public API

```python
from rag_naive.generation import RAG_PROMPT, SYSTEM_PROMPT, build_generation_chain, get_llm
```

| Name | Description |
|---|---|
| `SYSTEM_PROMPT` | Instructions to the LLM; takes `{context}` |
| `RAG_PROMPT` | `ChatPromptTemplate`: system message (instructions + context), then a user message (`{question}`) |
| `get_llm(settings)` | `ChatGroq` configured from settings. Raises `RuntimeError` if `GROQ_API_KEY` is missing. |
| `build_generation_chain(llm)` | LCEL chain. Input `{"question": str, "context": str}`, output answer `str`. Supports `invoke` and `stream`. |
| `normalize_citations(text)` | Converts `【` / `】` to `[` / `]` (in `llm.py`) |

`build_generation_chain` accepts any chat model, so tests and other providers can be dropped in without changing the chain.

## Prompt rules

The system prompt tells the model to:

1. **Use only the provided context.** If the answer isn't there, reply exactly: `I don't know based on the provided documents.`
2. **Be concise.**
3. **Cite every fact** by copying the passage label exactly, e.g. `[Employee_Benefits_Guide_2027.pdf, p. 6]`, with no numbering and no other citation style.

The context sits in the system message and the question in the user message. Curly braces in PDF text (e.g. `{salary}`) are inserted literally and aren't treated as template variables.

## Citation normalization

`openai/gpt-oss-120b` often wraps citations in `【…】` brackets, a habit from its training, even when told not to. The prompt alone doesn't reliably stop it, so the last step of the chain swaps those characters for `[` and `]`.

That step is a `RunnableGenerator`, not a `RunnableLambda`. A plain `RunnableLambda` would collect the whole streamed answer before passing it on, and `chat` would stop showing tokens as they arrive. A test checks that the output still arrives as separate tokens.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `GROQ_API_KEY` | (required) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Chat model |
| `GROQ_TEMPERATURE` | `0.0` | Sampling temperature (0 = most deterministic) |

Groq retires models from time to time. For example, `llama-3.3-70b-versatile` is no longer available. If you get `model_not_found`, list the models your key can use:

```
.venv\Scripts\python -c "from groq import Groq; from rag_naive import Settings; print([m.id for m in Groq(api_key=Settings().groq_api_key).models.list().data])"
```

## Tests

[tests/generation/](../../tests/generation/) uses LangChain fake chat models, so it makes no network calls and costs nothing:

```
.venv\Scripts\python -m pytest tests/generation
```

To check against the real Groq API, run the opt-in end-to-end tests: `pytest -m e2e`.
