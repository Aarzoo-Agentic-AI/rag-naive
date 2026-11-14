"""Generation stage: Groq chat model and the prompt -> LLM -> text chain."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from collections.abc import Iterator

from langchain_core.runnables import Runnable, RunnableGenerator

from ..config import Settings
from .prompts import RAG_PROMPT


def get_llm(settings: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.require_groq_key(),
        model=settings.groq_model,
        temperature=settings.groq_temperature,
    )


# Some models (e.g. gpt-oss) habitually cite with CJK lenticular brackets 【…】.
# A 1:1 character mapping is safe to apply to each streamed token independently.
_CITATION_BRACKETS = str.maketrans({"【": "[", "】": "]"})


def normalize_citations(text: str) -> str:
    return text.translate(_CITATION_BRACKETS)


def _normalize_stream(chunks: Iterator[str]) -> Iterator[str]:
    for chunk in chunks:
        yield normalize_citations(chunk)


def build_generation_chain(llm: BaseChatModel) -> Runnable:
    """Input: ``{"question": str, "context": str}`` -> output: answer string.

    The final step is a generator so token streaming is preserved.
    """
    return RAG_PROMPT | llm | StrOutputParser() | RunnableGenerator(_normalize_stream)
