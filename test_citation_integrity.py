import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from rag_graph import (
    validate_citations,
    extract_available_message_ids,
    extract_answer_citations,
    check_grounded,
    CitationValidationResult,
    GraphState,
)


def make_test_state(
    answer: str,
    documents=None,
    scope="specific",
    primary_intent="semantic_rag",
) -> GraphState:
    return {
        "question": "What time is the meeting?",
        "original_question": "What time is the meeting?",
        "documents": documents or [],
        "generation": answer,
        "rewrite_count": 0,
        "grounded": False,
        "semantic_grounded": False,
        "df": None,
        "scope": scope,
        "primary_intent": primary_intent,
        "secondary_intents": [],
        "query_spec": None,
        "citation_validation": None,
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "available_citation_ids": [],
    }


def make_mock_llm(response_text: str):
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=response_text)
    return mock


# =====================================================================
# 1. Valid citation to a retained message
# =====================================================================

def test_citation_case_1_valid_citation_retained_message():
    doc = Document(
        page_content="[ID 10] [2022-09-19 10:00 AM] Alice: The conference starts at 9am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The conference starts at 9am [ID 10]."
    val = validate_citations(answer, [doc])

    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.cited_ids == [10]
    assert val.valid_cited_ids == [10]
    assert val.invalid_cited_ids == []
    assert val.malformed_citations == []
    assert 10 in val.available_ids

    # Integration test with check_grounded
    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is True
    assert result["semantic_grounded"] is True
    assert result["valid_citation_ids"] == [10]
    assert result["invalid_citation_ids"] == []


# =====================================================================
# 2. Invented citation ID
# =====================================================================

def test_citation_case_2_invented_citation_id():
    doc = Document(
        page_content="[ID 10] Alice: The conference starts at 9am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The conference starts at 9am [ID 999]."
    val = validate_citations(answer, [doc])

    assert val.has_citations is True
    assert val.all_cited_valid is False
    assert val.cited_ids == [999]
    assert val.valid_cited_ids == []
    assert val.invalid_cited_ids == [999]

    # Integration: even if semantic grader returns 'yes', final grounded must be False
    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is False
    assert result["semantic_grounded"] is True
    assert result["invalid_citation_ids"] == [999]


# =====================================================================
# 3. Citation to a candidate removed before generation
# =====================================================================

def test_citation_case_3_citation_to_filtered_candidate():
    # Only ID 10 survived relevance grading and reached final context
    doc10 = Document(
        page_content="[ID 10] Alice: The conference starts at 9am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    # Answer cites ID 20, which was removed prior to generation
    answer = "The conference starts at 9am [ID 20]."
    val = validate_citations(answer, [doc10])

    assert val.all_cited_valid is False
    assert val.invalid_cited_ids == [20]
    assert 20 not in val.available_ids

    state = make_test_state(answer=answer, documents=[doc10])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is False
    assert result["invalid_citation_ids"] == [20]


# =====================================================================
# 4. Valid citation to a chronological neighbor present in final context
# =====================================================================

def test_citation_case_4_valid_chronological_neighbor():
    expanded_doc = Document(
        page_content=(
            "[RETRIEVED ANCHOR]\n"
            "[ID 10] [2022-09-19 10:00 AM] Alice: Hey team\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 9] [2022-09-19 09:59 AM] Charlie: Are we ready?\n"
            "[ID 10] [2022-09-19 10:00 AM] Alice: Hey team\n"
            "[ID 12] [2022-09-19 10:02 AM] Bob: Don't forget the presentation notes."
        ),
        metadata={
            "source": "hybrid_search",
            "message_id": 10,
            "context_expanded": True,
            "context_start_message_id": 9,
            "context_end_message_id": 12,
        },
    )
    answer = "Bob reminded everyone to bring the presentation notes [ID 12]."
    val = validate_citations(answer, [expanded_doc])

    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.cited_ids == [12]
    assert val.valid_cited_ids == [12]
    assert val.invalid_cited_ids == []
    # Both anchor and neighboring messages are available
    assert set(val.available_ids) == {9, 10, 12}

    state = make_test_state(answer=answer, documents=[expanded_doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is True
    assert 12 in result["available_citation_ids"]


# =====================================================================
# 5. Chunk ID incorrectly used as a message citation
# =====================================================================

def test_citation_case_5_chunk_id_incorrectly_used_as_citation():
    chunk_doc = Document(
        page_content=(
            "[message_id=1] [2022-09-19] Alice: Good morning\n"
            "[message_id=2] [2022-09-19] Bob: Hello everyone"
        ),
        metadata={
            "source": "conversation_chunk",
            "chunk_id": "chunk_abc123",
            "start_message_id": 1,
            "end_message_id": 2,
        },
    )

    # 5a: Citing non-numeric chunk string token
    answer_chunk_str = "The participants greeted each other [ID chunk_abc123]."
    val_a = validate_citations(answer_chunk_str, [chunk_doc])
    assert val_a.all_cited_valid is False
    assert "[ID chunk_abc123]" in val_a.malformed_citations

    # 5b: Citing raw chunk notation
    answer_chunk_raw = "The participants greeted each other [chunk_abc123]."
    val_b = validate_citations(answer_chunk_raw, [chunk_doc])
    assert val_b.all_cited_valid is False
    assert "[chunk_abc123]" in val_b.malformed_citations

    # 5c: Citing numeric chunk_id that is NOT an individual message ID in the chunk
    chunk_doc_numeric = Document(
        page_content="[message_id=1] Alice: Hi\n[message_id=2] Bob: Bye",
        metadata={
            "source": "conversation_chunk",
            "chunk_id": 777,
            "start_message_id": 1,
            "end_message_id": 2,
        },
    )
    # Available IDs should ONLY be message IDs {1, 2}, NOT chunk_id 777
    avail = extract_available_message_ids([chunk_doc_numeric])
    assert avail == {1, 2}
    assert 777 not in avail

    answer_numeric_chunk = "They said goodbye [ID 777]."
    val_c = validate_citations(answer_numeric_chunk, [chunk_doc_numeric])
    assert val_c.all_cited_valid is False
    assert val_c.invalid_cited_ids == [777]


# =====================================================================
# 6. Answer with no citations
# =====================================================================

def test_citation_case_6_answer_with_no_citations():
    doc = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The meeting is scheduled for 10am."

    # Specific query: missing_citations should be True
    val_specific = validate_citations(answer, [doc], scope="specific")
    assert val_specific.has_citations is False
    assert val_specific.cited_ids == []
    assert val_specific.missing_citations is True
    assert "No citations provided in answer for specific retrieval query." in val_specific.details

    # Broad query: whole-conversation statistics/profile does not require message citations
    val_broad = validate_citations(answer, [doc], scope="broad")
    assert val_broad.has_citations is False
    assert val_broad.missing_citations is False


# =====================================================================
# 7. Mixed valid and invalid citations
# =====================================================================

def test_citation_case_7_mixed_valid_and_invalid_citations():
    doc = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The meeting starts at 10am [ID 10] and lunch is served at noon [ID 50]."
    val = validate_citations(answer, [doc])

    assert val.has_citations is True
    assert val.all_cited_valid is False
    assert val.cited_ids == [10, 50]
    assert val.valid_cited_ids == [10]
    assert val.invalid_cited_ids == [50]

    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is False
    assert result["semantic_grounded"] is True


# =====================================================================
# 8. Valid ID attached to an unsupported claim (semantic check fails)
# =====================================================================

def test_citation_case_8_valid_id_with_unsupported_claim():
    doc1 = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    doc2 = Document(
        page_content="[ID 11] Bob: I had cereal for breakfast.",
        metadata={"source": "hybrid_search", "message_id": 11},
    )
    # The citation ID 11 exists mechanically, but Bob buying a yacht is completely unsupported
    answer = "The meeting is at 10am [ID 10], and Bob bought a yacht [ID 11]."
    val = validate_citations(answer, [doc1, doc2])

    # Mechanical validation passes!
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [10, 11]
    assert val.invalid_cited_ids == []

    # But semantic grounding check identifies the hallucinated claim and returns 'no'
    state = make_test_state(answer=answer, documents=[doc1, doc2])
    mock_llm = make_mock_llm("no")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["semantic_grounded"] is False
    # Final grounded must be False because semantic verification failed
    assert result["grounded"] is False


# =====================================================================
# 9. Missing metadata and malformed citation syntax
# =====================================================================

def test_citation_case_9_missing_metadata_and_malformed_syntax():
    # Documents with None and empty metadata
    doc_no_meta = Document(
        page_content="[ID 10] Alice: Meeting at 10",
        metadata={},
    )
    doc_none_meta = Document(
        page_content="[ID 11] Bob: Understood",
    )
    doc_none_meta.metadata = None

    avail = extract_available_message_ids([doc_no_meta, doc_none_meta])
    assert avail == {10, 11}

    # Answers with various malformed citation attempts
    answer_malformed = "Meeting is at 10 [ID ] or [ID abc] or [ID -5]."
    val = validate_citations(answer_malformed, [doc_no_meta, doc_none_meta])

    assert val.all_cited_valid is False
    assert len(val.malformed_citations) >= 2
    # Does not crash on empty / None answer
    val_empty = validate_citations("", [])
    assert val_empty.has_citations is False
    assert val_empty.all_cited_valid is False


# =====================================================================
# 10. Semantic grounding checker returns true but citation is invalid
# =====================================================================

def test_citation_case_10_semantic_yes_overridden_by_invalid_citation():
    doc = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    # The factual claim is true, but cited ID 999 is fabricated
    answer = "The meeting is at 10am [ID 999]."
    state = make_test_state(answer=answer, documents=[doc])

    # Semantic LLM evaluates prose claims and returns 'yes'
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)

    # Mechanical validation must override semantic 'yes' to False
    assert result["semantic_grounded"] is True
    assert result["grounded"] is False
    assert result["invalid_citation_ids"] == [999]
