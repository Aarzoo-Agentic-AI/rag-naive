"""Generation stage: stuff retrieved context into a prompt and call the Groq LLM."""

from .llm import build_generation_chain, get_llm
from .prompts import RAG_PROMPT, SYSTEM_PROMPT

__all__ = ["build_generation_chain", "get_llm", "RAG_PROMPT", "SYSTEM_PROMPT"]
