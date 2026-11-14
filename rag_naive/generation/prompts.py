"""Prompt templates for answer generation."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context "
    "extracted from PDF documents. If the answer is not contained in the context, say "
    "\"I don't know based on the provided documents.\" Be concise.\n\n"
    "Citations: each context passage starts with a label such as "
    "[Employee_Benefits_Guide_2027.pdf, p. 6]. After every fact you use, cite its passage "
    "by copying that label exactly, including the square brackets. Do not number "
    "passages, do not invent labels, and do not use any other citation style.\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)
