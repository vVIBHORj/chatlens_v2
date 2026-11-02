"""
test_numeric_bracket_citations.py
==================================
Deterministic offline test suite for Task 63:
Numeric Bracket Citation Extraction and Validation.

Covers all 21 scenarios required by Task 63 specification:
  1. Existing [ID 123] citation.
  2. New [123] citation.
  3. Existing [ID 123, ID 124] citations.
  4. New [123, 124] citations.
  5. Mixed [ID 123, 124] citations.
  6. Mixed [123, ID 124] citations.
  7. Whitespace and case variations.
  8. Duplicate numeric citations.
  9. Multiple citation groups in one answer.
  10. Valid citation IDs matching retrieved evidence.
  11. Unknown numeric bracket values.
  12. Ordinary times such as [10:30].
  13. Ordinary years such as [2024].
  14. Malformed chunk and episode markers.
  15. Malformed message-ID markers.
  16. Mixed valid and malformed citations.
  17. Answers with no citations.
  18. Pure abstention with an ID-prefixed citation.
  19. Pure abstention with a numeric-only citation.
  20. Substantive answers with numeric citations must not be treated as abstentions.
  21. Existing citation-integrity behavior remains unchanged.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from rag_graph import (
    extract_answer_citations,
    validate_citations,
    is_pure_abstention,
    strip_abstention_citations,
    generate,
    check_grounded,
    GraphState,
)


def make_mock_doc(msg_id: int, text: str = "Test message content") -> Document:
    return Document(
        page_content=f"[ID {msg_id}] [2026-03-31 14:30] Alice: {text}",
        metadata={"source": "hybrid_search", "message_id": msg_id},
    )


def make_mock_state(answer: str, documents=None, scope="specific") -> GraphState:
    return {
        "question": "What happened during the release?",
        "original_question": "What happened during the release?",
        "documents": documents or [],
        "generation": answer,
        "rewrite_count": 0,
        "grounded": False,
        "semantic_grounded": False,
        "df": None,
        "scope": scope,
        "primary_intent": "semantic_rag",
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
# Scenario 1: Existing [ID 123] citation
# =====================================================================
def test_01_existing_id_123_citation():
    ans = "The release build finished cleanly [ID 123]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [123]
    assert malformed == []

    doc = make_mock_doc(123)
    val = validate_citations(ans, [doc])
    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [123]
    assert val.invalid_cited_ids == []
    assert val.malformed_citations == []


# =====================================================================
# Scenario 2: New [123] citation
# =====================================================================
def test_02_new_123_numeric_citation():
    ans = "The release build finished cleanly [123]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [123]
    assert malformed == []

    doc = make_mock_doc(123)
    val = validate_citations(ans, [doc])
    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [123]
    assert val.invalid_cited_ids == []
    assert val.malformed_citations == []


# =====================================================================
# Scenario 3: Existing [ID 123, ID 124] citations
# =====================================================================
def test_03_existing_id_123_id_124_citations():
    ans = "Both nodes were updated [ID 123, ID 124]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [123, 124]
    assert malformed == []

    docs = [make_mock_doc(123), make_mock_doc(124)]
    val = validate_citations(ans, docs)
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [123, 124]
    assert val.invalid_cited_ids == []


# =====================================================================
# Scenario 4: New [123, 124] citations
# =====================================================================
def test_04_new_123_124_numeric_citations():
    ans = "Both nodes were updated [123, 124]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [123, 124]
    assert malformed == []

    docs = [make_mock_doc(123), make_mock_doc(124)]
    val = validate_citations(ans, docs)
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [123, 124]
    assert val.invalid_cited_ids == []


# =====================================================================
# Scenario 5: Mixed [ID 123, 124] citations
# =====================================================================
def test_05_mixed_id_123_124_citations():
    ans = "Both nodes were updated [ID 123, 124]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [123, 124]
    assert malformed == []

    docs = [make_mock_doc(123), make_mock_doc(124)]
    val = validate_citations(ans, docs)
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [123, 124]
    assert val.invalid_cited_ids == []


# =====================================================================
# Scenario 6: Mixed [123, ID 124] citations
# =====================================================================
def test_06_mixed_123_id_124_citations():
    ans = "Both nodes were updated [123, ID 124]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [123, 124]
    assert malformed == []

    docs = [make_mock_doc(123), make_mock_doc(124)]
    val = validate_citations(ans, docs)
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [123, 124]
    assert val.invalid_cited_ids == []


# =====================================================================
# Scenario 7: Whitespace and case variations
# =====================================================================
def test_07_whitespace_and_case_variations():
    variations = [
        ("Deploy clean [ 123 ].", [123]),
        ("Deploy clean [ID  123].", [123]),
        ("Deploy clean [id 123].", [123]),
        ("Deploy clean [Id 123].", [123]),
        ("Deploy clean [  123  ,  124  ].", [123, 124]),
        ("Deploy clean [id 123,ID 124].", [123, 124]),
        ("Deploy clean [123,124].", [123, 124]),
    ]
    for text, expected in variations:
        c, m = extract_answer_citations(text)
        assert c == expected, f"Failed for {text}"
        assert m == [], f"Failed malformed for {text}"


# =====================================================================
# Scenario 8: Duplicate numeric citations
# =====================================================================
def test_08_duplicate_numeric_citations():
    ans = "Alice confirmed [123] and Bob re-confirmed [123] in [ID 123]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [123]
    assert malformed == []

    ans_multi = "Nodes updated [123, 123, 124, 124]."
    cited_multi, _ = extract_answer_citations(ans_multi)
    assert cited_multi == [123, 124]


# =====================================================================
# Scenario 9: Multiple citation groups in one answer
# =====================================================================
def test_09_multiple_citation_groups_in_one_answer():
    ans = "Alice deployed the service [101]. Bob verified telemetry [102], and Charlie signed off [ID 103]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == [101, 102, 103]
    assert malformed == []

    docs = [make_mock_doc(101), make_mock_doc(102), make_mock_doc(103)]
    val = validate_citations(ans, docs)
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [101, 102, 103]
    assert val.invalid_cited_ids == []


# =====================================================================
# Scenario 10: Valid citation IDs matching retrieved evidence
# =====================================================================
def test_10_valid_citation_ids_matching_retrieved_evidence():
    docs = [make_mock_doc(50), make_mock_doc(51), make_mock_doc(52)]
    ans = "The failover completed at 14:35 [51] after primary node shutdown [50]."
    val = validate_citations(ans, docs)
    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [51, 50]
    assert val.invalid_cited_ids == []
    assert val.missing_citations is False


# =====================================================================
# Scenario 11: Unknown numeric bracket values
# =====================================================================
def test_11_unknown_numeric_bracket_values():
    docs = [make_mock_doc(101)]

    # Unknown numeric brackets without ID prefix are preserved as ordinary text
    ans_unknown = "The gateway returned status [502] and incident code [9999]."
    val_unknown = validate_citations(ans_unknown, docs)
    assert val_unknown.has_citations is False
    assert val_unknown.valid_cited_ids == []
    assert val_unknown.invalid_cited_ids == []
    assert val_unknown.cited_ids == []
    assert val_unknown.missing_citations is True

    # Unknown numeric brackets accompanying valid citations do not invalidate the answer
    ans_mixed_text = "Alice confirmed the build succeeded [101] after handling HTTP error [500]."
    val_mixed = validate_citations(ans_mixed_text, docs)
    assert val_mixed.has_citations is True
    assert val_mixed.all_cited_valid is True
    assert val_mixed.valid_cited_ids == [101]
    assert val_mixed.invalid_cited_ids == []


# =====================================================================
# Scenario 12: Ordinary times such as [10:30]
# =====================================================================
def test_12_ordinary_times_in_brackets():
    ans = "The meeting took place between [10:30] and [11:45 AM]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == []
    assert malformed == []

    docs = [make_mock_doc(101)]
    val = validate_citations(ans, docs)
    assert val.has_citations is False
    assert val.malformed_citations == []


# =====================================================================
# Scenario 13: Ordinary years such as [2024]
# =====================================================================
def test_13_ordinary_years_in_brackets():
    docs = [make_mock_doc(101)]
    ans = "The system migration originally started in [2024] and finalized in [2026]."
    val = validate_citations(ans, docs)
    assert val.has_citations is False
    assert val.valid_cited_ids == []
    assert val.invalid_cited_ids == []
    assert val.malformed_citations == []


# =====================================================================
# Scenario 14: Malformed chunk and episode markers
# =====================================================================
def test_14_malformed_chunk_and_episode_markers():
    ans = "The data was found in [chunk_123] and [episode_45]."
    cited, malformed = extract_answer_citations(ans)
    assert cited == []
    assert malformed == ["[chunk_123]", "[episode_45]"]

    docs = [make_mock_doc(123)]
    val = validate_citations(ans, docs)
    assert val.has_citations is True
    assert val.all_cited_valid is False
    assert val.malformed_citations == ["[chunk_123]", "[episode_45]"]


# =====================================================================
# Scenario 15: Malformed message-ID markers
# =====================================================================
def test_15_malformed_message_id_markers():
    ans = "Reference markers [message_id=123] and [message_id: 45] were used."
    cited, malformed = extract_answer_citations(ans)
    assert cited == []
    assert malformed == ["[message_id=123]", "[message_id: 45]"]

    docs = [make_mock_doc(123)]
    val = validate_citations(ans, docs)
    assert val.all_cited_valid is False
    assert val.malformed_citations == ["[message_id=123]", "[message_id: 45]"]


# =====================================================================
# Scenario 16: Mixed valid and malformed citations
# =====================================================================
def test_16_mixed_valid_and_malformed_citations():
    ans = "Alice confirmed the release [101] with context in [chunk_88]."
    docs = [make_mock_doc(101)]
    val = validate_citations(ans, docs)
    assert val.has_citations is True
    assert val.all_cited_valid is False
    assert val.valid_cited_ids == [101]
    assert val.malformed_citations == ["[chunk_88]"]


# =====================================================================
# Scenario 17: Answers with no citations
# =====================================================================
def test_17_answers_with_no_citations():
    ans = "The team discussed the upcoming quarterly goals without reaching a conclusion."
    cited, malformed = extract_answer_citations(ans)
    assert cited == []
    assert malformed == []

    docs = [make_mock_doc(101)]
    val = validate_citations(ans, docs)
    assert val.has_citations is False
    assert val.missing_citations is True
    assert val.all_cited_valid is False


# =====================================================================
# Scenario 18: Pure abstention with an ID-prefixed citation
# =====================================================================
def test_18_pure_abstention_with_id_prefixed_citation():
    raw_ans = "The staging API credentials were not found in the provided conversation. [ID 101]"
    assert is_pure_abstention(raw_ans) is True

    doc = make_mock_doc(101)
    state = make_mock_state(answer="", documents=[doc])
    mock_llm = make_mock_llm(raw_ans)

    gen_state = generate(state, generation_llm=mock_llm)
    assert gen_state["raw_generation"] == raw_ans
    assert gen_state["abstention_citations_stripped"] is True
    assert gen_state["generation"] == "The staging API credentials were not found in the provided conversation."
    assert "[ID 101]" not in gen_state["generation"]

    val = validate_citations(gen_state["generation"], [doc])
    assert val.has_citations is False
    assert "Valid abstention" in val.details


# =====================================================================
# Scenario 19: Pure abstention with a numeric-only citation
# =====================================================================
def test_19_pure_abstention_with_numeric_only_citation():
    raw_ans = "The staging API credentials were not discussed in the provided conversation. [101]"
    doc = make_mock_doc(101)
    assert is_pure_abstention(raw_ans, available_ids={101}) is True

    state = make_mock_state(answer="", documents=[doc])
    mock_llm = make_mock_llm(raw_ans)

    gen_state = generate(state, generation_llm=mock_llm)
    assert gen_state["raw_generation"] == raw_ans
    assert gen_state["abstention_citations_stripped"] is True
    assert gen_state["generation"] == "The staging API credentials were not discussed in the provided conversation."
    assert "[101]" not in gen_state["generation"]

    val = validate_citations(gen_state["generation"], [doc])
    assert val.has_citations is False
    assert "Valid abstention" in val.details


# =====================================================================
# Scenario 20: Substantive answers with numeric citations must not be treated as abstentions
# =====================================================================
def test_20_substantive_answers_with_numeric_citations_not_abstentions():
    substantive_ans = "Alice deployed the v2.4.0 release build cleanly [101]."
    doc = make_mock_doc(101)

    assert is_pure_abstention(substantive_ans, available_ids={101}) is False

    state = make_mock_state(answer="", documents=[doc])
    mock_llm = make_mock_llm(substantive_ans)

    gen_state = generate(state, generation_llm=mock_llm)
    assert gen_state["abstention_citations_stripped"] is False
    assert gen_state["generation"] == substantive_ans
    assert "[101]" in gen_state["generation"]

    val = validate_citations(gen_state["generation"], [doc])
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [101]


# =====================================================================
# Scenario 21: Existing citation-integrity behavior remains unchanged
# =====================================================================
def test_21_existing_citation_integrity_behavior_remains_unchanged():
    doc = make_mock_doc(10)

    # Valid [ID 10] passes
    val_valid = validate_citations("The meeting starts at 9am [ID 10].", [doc])
    assert val_valid.all_cited_valid is True
    assert val_valid.valid_cited_ids == [10]
    assert val_valid.invalid_cited_ids == []

    # Invented [ID 999] rejected with invalid_cited_ids=[999]
    val_invented = validate_citations("The meeting starts at 9am [ID 999].", [doc])
    assert val_invented.has_citations is True
    assert val_invented.all_cited_valid is False
    assert val_invented.valid_cited_ids == []
    assert val_invented.invalid_cited_ids == [999]

    # Groundedness check overrides semantic yes to False when invalid citations are present
    state = make_mock_state(answer="The meeting starts at 9am [ID 999].", documents=[doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is False
    assert result["semantic_grounded"] is True
    assert result["invalid_citation_ids"] == [999]

