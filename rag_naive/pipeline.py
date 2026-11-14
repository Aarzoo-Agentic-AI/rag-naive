"""Query-time orchestration: the classic three-step naive RAG.

1. Query encoding  - the question is embedded with the same model as the chunks.
2. Retrieval       - top-k nearest chunks are fetched from ChromaDB.   (retrieval/)
3. Generation      - chunks are stuffed into a prompt and sent to Groq. (generation/)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import VectorStore

from .config import Settings
from .generation import build_generation_chain, get_llm
from .retrieval import build_retriever, format_docs
from .vectorstore import get_vectorstore


@dataclass
class RAGResponse:
    answer: str
    sources: list[Document]


class NaiveRAG:
    """Retrieve-then-generate pipeline built with LangChain LCEL."""

    def __init__(
        self,
        settings: Settings | None = None,
        vectorstore: VectorStore | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.vectorstore = vectorstore or get_vectorstore(self.settings, embeddings=embeddings)
        self.llm = llm or get_llm(self.settings)

        self.retriever = build_retriever(self.vectorstore, self.settings.top_k)
        self.generation_chain = build_generation_chain(self.llm)
        self.chain: Runnable = self._build_chain()
        self.last_sources: list[Document] = []

    def _build_chain(self) -> Runnable:
        """{question} -> {question, docs} -> {question, docs, answer}"""
        return RunnablePassthrough.assign(
            docs=RunnableLambda(lambda x: x["question"]) | self.retriever
        ).assign(
            answer=(
                RunnableLambda(
                    lambda x: {"question": x["question"], "context": format_docs(x["docs"])}
                )
                | self.generation_chain
            )
        )

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def ask(self, question: str) -> RAGResponse:
        result = self.chain.invoke({"question": question})
        return RAGResponse(answer=result["answer"], sources=result["docs"])

    def stream(self, question: str) -> Iterator[str]:
        """Yield answer tokens; retrieved docs are available via ``self.last_sources``."""
        docs = self.retrieve(question)
        self.last_sources = docs
        yield from self.generation_chain.stream(
            {"question": question, "context": format_docs(docs)}
        )
