"""Tests for rag_naive.retrieval.retriever.

Similarity search runs against a real, temporary ChromaDB. A tiny keyword
"bag of words" embedding makes nearest-neighbour results predictable without
downloading a model.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.documents import Document

from rag_naive.generation.prompts import SYSTEM_PROMPT
from rag_naive.retrieval import build_retriever, citation_label, format_docs
from rag_naive.vectorstore import get_vectorstore

CHUNKS = [
    Document(page_content="PTO vacation days: 15 per year.",
             metadata={"source": "benefits.pdf", "page": 5}),
    Document(page_content="Bonus equals salary times target bonus percent.",
             metadata={"source": "bonus.pdf", "page": 4}),
    Document(page_content="401k match is 100% of the first 4%.",
             metadata={"source": "benefits.pdf", "page": 4}),
    Document(page_content="Insurance deductible is $1,700 for the HDHP.",
             metadata={"source": "benefits.pdf", "page": 2}),
]


@pytest.fixture
def embeddings(keyword_embeddings):
    return keyword_embeddings


@pytest.fixture
def store(make_settings, embeddings):
    vs = get_vectorstore(make_settings(), embeddings=embeddings)
    vs.add_documents(CHUNKS, ids=[f"id-{i}" for i in range(len(CHUNKS))])
    return vs


# --------------------------------------------------------------------------- #
# build_retriever
# --------------------------------------------------------------------------- #
def test_retriever_uses_plain_similarity_search_with_top_k(store):
    retriever = build_retriever(store, top_k=3)

    # Naive RAG: no MMR, no score threshold, no metadata filter.
    assert retriever.search_type == "similarity"
    assert retriever.search_kwargs == {"k": 3}


@pytest.mark.parametrize("k", [1, 2, 3])
def test_retriever_returns_exactly_top_k_documents(store, k):
    docs = build_retriever(store, top_k=k).invoke("pto vacation bonus 401k")

    assert len(docs) == k


def test_retriever_returns_all_documents_when_top_k_exceeds_store_size(store):
    docs = build_retriever(store, top_k=10).invoke("anything")

    assert len(docs) == len(CHUNKS)


@pytest.mark.parametrize(
    ("question", "expected_source", "expected_page"),
    [
        ("How many PTO vacation days do I get?", "benefits.pdf", 5),
        ("How is the bonus calculated from salary?", "bonus.pdf", 4),
        ("What is the 401k match?", "benefits.pdf", 4),
        ("What is the insurance deductible?", "benefits.pdf", 2),
    ],
)
def test_most_similar_chunk_is_ranked_first(store, question, expected_source, expected_page):
    top = build_retriever(store, top_k=2).invoke(question)[0]

    assert top.metadata["source"] == expected_source
    assert top.metadata["page"] == expected_page


def test_results_are_ordered_by_decreasing_similarity(store):
    # "bonus" x2 + "salary" -> the bonus chunk must beat everything else.
    docs = build_retriever(store, top_k=4).invoke("bonus bonus salary pto")

    assert docs[0].metadata["source"] == "bonus.pdf"
    assert "PTO" in docs[1].page_content


def test_query_is_encoded_with_the_store_embedding_model(store, embeddings):
    build_retriever(store, top_k=1).invoke("What is the 401k match?")

    assert embeddings.query_calls == ["What is the 401k match?"]


def test_retrieved_documents_keep_content_and_metadata(store):
    (doc,) = build_retriever(store, top_k=1).invoke("insurance deductible")

    assert doc.page_content == CHUNKS[3].page_content
    assert doc.metadata == CHUNKS[3].metadata


def test_retriever_on_empty_store_returns_no_documents(make_settings, embeddings):
    empty = get_vectorstore(make_settings(collection_name="empty"), embeddings=embeddings)

    assert build_retriever(empty, top_k=4).invoke("pto") == []


# --------------------------------------------------------------------------- #
# citation_label
# --------------------------------------------------------------------------- #
def _doc(**metadata) -> Document:
    return Document(page_content="text", metadata=metadata)


@pytest.mark.parametrize(
    ("source", "expected_name"),
    [
        (r"data\pdfs\HR_Policy_Handbook_2027.pdf", "HR_Policy_Handbook_2027.pdf"),
        (r"C:\docs\nested\Guide.pdf", "Guide.pdf"),
        ("data/pdfs/Guide.pdf", "Guide.pdf"),
        ("/home/user/Guide.pdf", "Guide.pdf"),
        ("Guide.pdf", "Guide.pdf"),
    ],
)
def test_citation_label_uses_file_name_only(source, expected_name):
    assert citation_label(_doc(source=source, page=0)) == f"[{expected_name}, p. 1]"


@pytest.mark.parametrize(("page", "shown"), [(0, 1), (3, 4), (41, 42)])
def test_citation_label_converts_zero_based_page_to_one_based(page, shown):
    assert citation_label(_doc(source="a.pdf", page=page)) == f"[a.pdf, p. {shown}]"


def test_citation_label_omits_page_when_missing():
    assert citation_label(_doc(source="a.pdf")) == "[a.pdf]"


def test_citation_label_omits_page_when_not_an_integer():
    assert citation_label(_doc(source="a.pdf", page="3")) == "[a.pdf]"


@pytest.mark.parametrize("source", [None, ""])
def test_citation_label_falls_back_to_unknown_source(source):
    assert citation_label(_doc(source=source, page=2)) == "[unknown, p. 3]"


def test_citation_label_without_any_metadata():
    assert citation_label(Document(page_content="text")) == "[unknown]"


def test_citation_label_matches_format_taught_in_system_prompt():
    # The prompt tells the LLM to copy labels like "[Employee_Benefits_Guide_2027.pdf, p. 6]".
    label = citation_label(_doc(source=r"data\pdfs\Employee_Benefits_Guide_2027.pdf", page=5))

    assert label in SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# format_docs
# --------------------------------------------------------------------------- #
def test_format_docs_with_no_documents_is_empty_string():
    assert format_docs([]) == ""


def test_format_docs_prefixes_chunk_with_its_label():
    doc = Document(page_content="PTO is 15 days.", metadata={"source": "g.pdf", "page": 5})

    assert format_docs([doc]) == "[g.pdf, p. 6]\nPTO is 15 days."


def test_format_docs_separates_chunks_with_blank_line_in_retrieval_order():
    docs = [
        Document(page_content="first", metadata={"source": "b.pdf", "page": 1}),
        Document(page_content="second", metadata={"source": "a.pdf", "page": 0}),
    ]

    assert format_docs(docs) == "[b.pdf, p. 2]\nfirst\n\n[a.pdf, p. 1]\nsecond"


def test_format_docs_does_not_number_passages():
    # Numbered passages ("[1]") led the LLM to cite "【1, page 6】" instead of labels.
    text = format_docs(CHUNKS)

    assert not re.search(r"^\[\d+\]", text, flags=re.MULTILINE)


def test_format_docs_keeps_chunk_text_verbatim():
    doc = Document(page_content="  Match (100%)\nof {salary}  ", metadata={"source": "x.pdf", "page": 0})

    assert format_docs([doc]).endswith("  Match (100%)\nof {salary}  ")
