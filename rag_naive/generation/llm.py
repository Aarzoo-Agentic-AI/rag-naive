"""Generation stage: Groq chat model and the prompt -> LLM -> text chain."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from ..config import Settings
from .prompts import RAG_PROMPT


def get_llm(settings: Settings) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.require_groq_key(),
        model=settings.groq_model,
        temperature=settings.groq_temperature,
    )


def build_generation_chain(llm: BaseChatModel) -> Runnable:
    """Input: ``{"question": str, "context": str}`` -> output: answer string."""
    return RAG_PROMPT | llm | StrOutputParser()
