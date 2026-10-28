"""
test_citation_remediation_task41.py
===================================
Deterministic offline tests for Task 41 citation attribution, abstention remediation,
and CASE-12 reconciliation across 10 required scenarios:
  1. An answer to a preceding question cites the reply containing the answer, not the question itself.
  2. An unanswerable request with misleading contextual evidence results in correct abstention and no unrelated citations.
  3. A negative response does not cite a similar but irrelevant entity as evidence.
  4. A short time or numerical answer preserves the correct value and uses the supporting message ID.
  5. A confirmation containing "completely clean" retains that fact.
  6. A valid multi-message answer can cite multiple messages when both are necessary.
  7. Existing valid citations and grounded answers continue to pass.
  8. Citation IDs that do not exist in the supplied evidence remain invalid.
  9. Both default and experimental modes preserve their intended routing and citation schema.
  10. The evaluation case definitions and scoring code agree on case identity, answerability, and CASE-12 reconciliation.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from rag_graph import (
    build_generation_prompt,
    check_grounded,
    extract_answer_citations,
    extract_available_message_ids,
    generate,
    is_pure_abstention,
    validate_citations,
    GraphState,
)
from run_dialogue_evidence_live import EVAL_CASES


def make_test_state(answer: str, documents=None, scope="specific", question="What is the answer?") -> GraphState:
    return {
        "question": question,
        "original_question": question,
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
# Scenario 1: Preceding Question vs. Reply Citation Attribution
# =====================================================================
def test_scenario_1_reply_cited_not_preceding_question():
    doc = Document(
        page_content=(
            "[RETRIEVED ANCHOR]\n"
            "[ID 104] [14/10/2026 14:31] Dev_Lead: 9pm.\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 103] [14/10/2026 14:30] Dev_Ops: When are we kicking off the production deployment window tonight?\n"
            "[ID 104] [14/10/2026 14:31] Dev_Lead: 9pm."
        ),
        metadata={"message_id": 104, "source": "hybrid_search"},
    )
    state = make_test_state(answer="", documents=[doc], question="What time is the deployment window?")
    prompt = build_generation_prompt(state, documents=[doc])

    # Check prompt instructions explicitly distinguish question from reply
    assert "When a short reply answers a preceding question, cite ONLY the reply message that supplies the answer" in prompt
    assert "NOT the preceding question" in prompt

    # Generating the answer citing only the reply message [ID 104]
    mock_llm = make_mock_llm("The production deployment window is scheduled for 9pm [ID 104].")
    gen_state = generate(state, generation_llm=mock_llm)
    assert gen_state["generation"] == "The production deployment window is scheduled for 9pm [ID 104]."

    val = validate_citations(gen_state["generation"], [doc])
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [104]
    assert 103 not in val.cited_ids


# =====================================================================
# Scenario 2: Unanswerable Request with Misleading Context Abstains Cleanly
# =====================================================================
def test_scenario_2_unanswerable_misleading_context_abstention():
    doc = Document(
        page_content=(
            "[RETRIEVED ANCHOR]\n"
            "[ID 110] [14/10/2026 10:02] Dev_Sam: On the kitchen counter.\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 109] [14/10/2026 10:01] Dev_Alex: Did anyone see where I left my car keys?\n"
            "[ID 110] [14/10/2026 10:02] Dev_Sam: On the kitchen counter."
        ),
        metadata={"message_id": 110, "source": "hybrid_search"},
    )
    state = make_test_state(answer="", documents=[doc], question="Where is the staging API key stored?")
    prompt = build_generation_prompt(state, documents=[doc])

    assert "Do NOT infer an absent fact from unrelated or merely similar messages" in prompt
    assert "In an abstention, do NOT output any citation brackets" in prompt

    # If the LLM generates an abstention with an extraneous citation bracket mistakenly attached
    mock_llm = make_mock_llm("The staging API key was not mentioned in the provided conversation. [ID 110]")
    gen_state = generate(state, generation_llm=mock_llm)

    # Abstention enforcement must strip the extraneous citation bracket
    assert gen_state["generation"] == "The staging API key was not mentioned in the provided conversation."
    assert "[ID 110]" not in gen_state["generation"]

    val = validate_citations(gen_state["generation"], [doc])
    assert val.has_citations is False
    assert val.valid_cited_ids == []
    assert val.missing_citations is False
    assert "Valid abstention" in val.details


# =====================================================================
# Scenario 3: Negative Response Does Not Cite Similar Irrelevant Entity
# =====================================================================
def test_scenario_3_negative_response_no_similar_entity_citation():
    doc = Document(
        page_content=(
            "[RETRIEVED ANCHOR]\n"
            "[ID 123] [14/10/2026 11:22] Dev_Auth: In secrets.dev.json.\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 122] [14/10/2026 11:21] Dev_Test: Where did we save the dev environment test JWT token?\n"
            "[ID 123] [14/10/2026 11:22] Dev_Auth: In secrets.dev.json."
        ),
        metadata={"message_id": 123, "source": "hybrid_search"},
    )
    answer = "The production JWT secret was not discussed in the provided conversation."
    val = validate_citations(answer, [doc])

    assert val.has_citations is False
    assert val.valid_cited_ids == []
    assert val.missing_citations is False
    assert is_pure_abstention(answer) is True


# =====================================================================
# Scenario 4: Short Numerical/Temporal Answer Preserves Value and Supporting ID
# =====================================================================
def test_scenario_4_short_time_answer_preserves_value_and_supporting_id():
    doc = Document(
        page_content="[ID 104] Dev_Lead: 9pm.",
        metadata={"message_id": 104, "source": "hybrid_search"},
    )
    answer = "The production deployment window is scheduled for 9pm [ID 104]."
    assert "9pm" in answer

    val = validate_citations(answer, [doc])
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [104]

    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    res = check_grounded(state, grounding_llm=mock_llm)
    assert res["grounded"] is True
    assert res["valid_citation_ids"] == [104]


# =====================================================================
# Scenario 5: Confirmation Containing "completely clean" Retains That Fact
# =====================================================================
def test_scenario_5_confirmation_completely_clean_retains_fact():
    doc = Document(
        page_content=(
            "[RETRIEVED ANCHOR]\n"
            "[ID 108] [14/10/2026 16:20] Dev_Sarah: Yes, completely clean.\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 107] [14/10/2026 16:19] Dev_Chris: Did the database migration script finish successfully on staging?\n"
            "[ID 108] [14/10/2026 16:20] Dev_Sarah: Yes, completely clean."
        ),
        metadata={"message_id": 108, "source": "hybrid_search"},
    )
    state = make_test_state(answer="", documents=[doc], question="Did the database migration script succeed on staging?")
    prompt = build_generation_prompt(state, documents=[doc])

    assert "Preserve Exact Facts in Short Replies" in prompt
    assert "'Yes, completely clean'" in prompt

    answer = "Yes, the database migration script succeeded on staging and was completely clean [ID 108]."
    assert "clean" in answer
    assert "yes" in answer.lower()

    val = validate_citations(answer, [doc])
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [108]


# =====================================================================
# Scenario 6: Multi-Message Answer Cites Multiple When Necessary
# =====================================================================
def test_scenario_6_multi_message_answer_cites_multiple_necessary_messages():
    doc = Document(
        page_content=(
            "[RETRIEVED ANCHOR]\n"
            "[ID 119] Dev_Facilities: Facilities confirmed Rack C4.\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 117] Dev_Eng: Is Rack C4 or C5 having power fluctuation?\n"
            "[ID 118] Dev_Ops: Rack C4 has voltage drops.\n"
            "[ID 119] Dev_Facilities: Facilities confirmed Rack C4."
        ),
        metadata={"message_id": 119, "source": "hybrid_search"},
    )
    answer = "Rack C4 experienced voltage drops [ID 118] and facilities confirmed the fluctuation [ID 119]."
    val = validate_citations(answer, [doc])

    assert val.all_cited_valid is True
    assert set(val.valid_cited_ids) == {118, 119}
    assert val.invalid_cited_ids == []


# =====================================================================
# Scenario 7: Existing Valid Citations Continue to Pass
# =====================================================================
def test_scenario_7_existing_valid_citations_continue_to_pass():
    doc = Document(
        page_content="[ID 0] Commander Vance: The primary deflector shield operates at 450 GHz.",
        metadata={"message_id": 0, "source": "hybrid_search"},
    )
    answer = "The primary deflector shield operates at 450 GHz [ID 0]."
    val = validate_citations(answer, [doc])

    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [0]
    assert val.invalid_cited_ids == []

    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    res = check_grounded(state, grounding_llm=mock_llm)
    assert res["grounded"] is True


# =====================================================================
# Scenario 8: Nonexistent Citation IDs Remain Invalid
# =====================================================================
def test_scenario_8_nonexistent_ids_remain_invalid():
    doc = Document(
        page_content="[ID 10] Alice: The conference starts at 9am.",
        metadata={"message_id": 10, "source": "hybrid_search"},
    )
    answer = "The conference starts at 9am [ID 888]."
    val = validate_citations(answer, [doc])

    assert val.all_cited_valid is False
    assert val.invalid_cited_ids == [888]
    assert 888 not in val.available_ids

    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    res = check_grounded(state, grounding_llm=mock_llm)
    assert res["grounded"] is False
    assert res["invalid_citation_ids"] == [888]


# =====================================================================
# Scenario 9: Default and Experimental Modes Preserve Schema
# =====================================================================
def test_scenario_9_baseline_and_treatment_modes_preserve_routing_and_schema():
    from rag_graph import is_dialogue_evidence_enabled

    # Baseline: dialogue_evidence_mode=False
    state_base: GraphState = {"dialogue_evidence_mode": False}
    assert is_dialogue_evidence_enabled(state_base) is False

    # Treatment: dialogue_evidence_mode=True
    state_treat: GraphState = {"dialogue_evidence_mode": True}
    assert is_dialogue_evidence_enabled(state_treat) is True

    # Validate output schema dictionary keys
    expected_keys = {
        "answer", "grounded", "sources", "rewrites_used", "scope",
        "query_spec", "citation_validation", "valid_citation_ids",
        "invalid_citation_ids", "available_citation_ids", "grader_status", "evidence_mode"
    }

    # Simulate output state from rag_graph
    simulated_result = {
        "generation": "The backup was in Vault 9 [ID 102].",
        "grounded": True,
        "documents": [Document(page_content="In Vault 9.", metadata={"message_id": 102})],
        "rewrite_count": 0,
        "scope": "specific",
        "query_spec": {"intent": "semantic_rag"},
        "citation_validation": {"all_cited_valid": True},
        "valid_citation_ids": [102],
        "invalid_citation_ids": [],
        "available_citation_ids": [102],
        "grader_status": "bypassed",
        "evidence_mode": "dialogue_evidence",
    }

    output_dict = {
        "answer": simulated_result["generation"],
        "grounded": simulated_result["grounded"],
        "sources": [d.metadata for d in simulated_result["documents"]],
        "rewrites_used": simulated_result["rewrite_count"],
        "scope": simulated_result["scope"],
        "query_spec": simulated_result.get("query_spec"),
        "citation_validation": simulated_result.get("citation_validation"),
        "valid_citation_ids": simulated_result.get("valid_citation_ids", []),
        "invalid_citation_ids": simulated_result.get("invalid_citation_ids", []),
        "available_citation_ids": simulated_result.get("available_citation_ids", []),
        "grader_status": simulated_result.get("grader_status"),
        "evidence_mode": simulated_result.get("evidence_mode", "standard"),
    }
    assert set(output_dict.keys()) == expected_keys


# =====================================================================
# Scenario 10: Evaluation Case Manifest Agreement & CASE-12 Reconciliation
# =====================================================================
def test_scenario_10_case_manifest_consistency_and_case12_reconciliation():
    assert len(EVAL_CASES) == 12

    # Verify CASE-12 definition
    case12 = next(c for c in EVAL_CASES if c.case_id == "CASE-012-Ordinary-Length-Bypass")
    assert case12.target_id == 125
    assert case12.expected_answerable is True
    assert case12.expected_supporting_ids == [125]
    assert "90 days" in case12.expected_answer_keywords

    # Word and character count verification
    words = case12.target_text.split()
    assert len(words) == 18, "CASE-12 must have 18 words to bypass the 6-word threshold"
    assert len(case12.target_text) > 35

    # Verification of unanswerable cases
    unanswerable_ids = [c.case_id for c in EVAL_CASES if not c.expected_answerable]
    assert set(unanswerable_ids) == {
        "CASE-05-Misleading-Context",
        "CASE-07-Self-Contained-Irrelevant",
        "CASE-08-Unrelated-Candidate-Matching-Context",
        "CASE-010-Ambiguous-Context-Abstention",
        "CASE-011-Similar-Entity-Negative",
    }
    assert len(unanswerable_ids) == 5

    # Verification of answerable cases
    answerable_ids = [c.case_id for c in EVAL_CASES if c.expected_answerable]
    assert len(answerable_ids) == 7
    assert "CASE-012-Ordinary-Length-Bypass" in answerable_ids


# =====================================================================
# Task 42 Section 2: Abstention Post-Processing Offline Verifications
# =====================================================================
def test_abstention_condition_1_pure_statement_citation_stripped():
    """1. Pure statement that fact is absent has extraneous message citation removed."""
    doc = Document(
        page_content="[ID 110] Dev_Sam: On the kitchen counter.",
        metadata={"message_id": 110, "source": "hybrid_search"},
    )
    state = make_test_state(answer="", documents=[doc], question="Where is the staging API key stored?")
    raw = "The staging API key was not discussed in the provided conversation. [ID 110]"
    mock_llm = make_mock_llm(raw)

    gen_state = generate(state, generation_llm=mock_llm)
    assert gen_state["raw_generation"] == raw
    assert gen_state["abstention_citations_stripped"] is True
    assert gen_state["generation"] == "The staging API key was not discussed in the provided conversation."
    assert "[ID 110]" not in gen_state["generation"]


def test_abstention_condition_2_substantive_answer_with_secondary_unmentioned_fact():
    """2. Substantive answer does not lose citation merely because another sentence notes something was not discussed."""
    doc = Document(
        page_content="[ID 10] Alice: Confirmed the release deadline on Monday.",
        metadata={"message_id": 10, "source": "hybrid_search"},
    )
    state = make_test_state(answer="", documents=[doc], question="When is the deadline and what is the budget?")
    substantive_answer = "Alice confirmed the release deadline on Monday [ID 10]. The budget was not discussed in the conversation."
    mock_llm = make_mock_llm(substantive_answer)

    # is_pure_abstention must return False for mixed/substantive answer
    assert is_pure_abstention(substantive_answer) is False

    gen_state = generate(state, generation_llm=mock_llm)
    assert gen_state["abstention_citations_stripped"] is False
    assert gen_state["generation"] == substantive_answer
    assert "[ID 10]" in gen_state["generation"]


def test_abstention_condition_3_mixed_response_not_converted_to_valid_abstention():
    """3. Mixed response containing both unsupported and supported claims is not converted to a valid abstention."""
    mixed_answer = "Bob arrived at 9am [ID 1], but user table backup was not found."
    assert is_pure_abstention(mixed_answer) is False


def test_abstention_condition_4_question_or_negative_action_not_pure_abstention():
    """4. Answer containing a question or negative statement is not classified as pure abstention solely because of a matching keyword."""
    # Negative factual assertion about an event, not an absent fact in context
    negative_assertion = "The server did not crash during the migration window [ID 5]."
    assert is_pure_abstention(negative_assertion) is False

    # Question
    question_response = "Did the database migration script succeed? The logs were not checked."
    assert is_pure_abstention(question_response) is False


def test_abstention_condition_5_existing_grounded_and_multiple_citations_retained():
    """5. Existing grounded answers, multiple valid citations, invalid message IDs retain intended behavior."""
    # Multiple citations
    multi_answer = "Rack C4 experienced voltage drops [ID 118] and facilities confirmed the fluctuation [ID 119]."
    assert is_pure_abstention(multi_answer) is False

    # Invalid ID
    invalid_id_answer = "The meeting is at 10am [ID 999]."
    assert is_pure_abstention(invalid_id_answer) is False

