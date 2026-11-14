"""Prompt templates for answer generation."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context "
    "extracted from PDF documents. If the answer is not contained in the context, say "
    "\"I don't know based on the provided documents.\" Be concise and cite sources as "
    "[source, page] when possible.\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)
