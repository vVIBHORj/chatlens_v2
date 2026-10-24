"""
test_controlled_grading_comparison.py
======================================
Controlled offline comparison between context-aware relevance grading
and standalone grading for short conversational replies in ChatLens-RAG.

CRITICAL EVALUATION DISCLAIMER:
--------------------------------
This test module provides a DETERMINISTIC PLUMBING AND REGRESSION EVALUATION
of the pipeline logic, prompt assembly, and document state preservation.
It uses deterministic mock rule functions to verify that:
  1. The pipeline correctly formats preceding context when available.
  2. The pipeline routes mock decisions into candidate retention or exclusion.
  3. Preceding context is strictly grading-only and never mutates document content,
     metadata, or citation identifiers.
  4. Non-short messages, chunks, and missing database states safely bypass or fall back.

This module does NOT constitute evidence or empirical measurement of real-world
LLM (e.g. Ollama/Qwen) semantic accuracy, precision, or recall. Real model performance
must be evaluated using live benchmark runs when authorized.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_graph import (
    grade_documents,
    grade_candidate_batch,
    GraphState,
)


def make_test_state(
    documents=None,
    question="Where are the backup cryogenic coolant canisters stored?",
    original_question=None,
    scope="specific",
    primary_intent="semantic_rag",
    database_path="/mock/test_database.db",
) -> GraphState:
    orig = original_question if original_question is not None else question
    return {
        "question": question,
        "original_question": orig,
        "documents": documents or [],
        "generation": "",
        "rewrite_count": 0,
        "grounded": False,
        "df": None,
        "scope": scope,
        "primary_intent": primary_intent,
        "secondary_intents": [],
        "query_spec": None,
        "database_path": database_path,
    }


class ConfigurableGraderMockLLM:
    """
    Mock LLM that captures prompts and evaluates decisions using a deterministic rule function.
    """
    def __init__(self, rule_fn=None, fixed_json_response=None):
        self.recorded_prompts = []
        self.rule_fn = rule_fn
        self.fixed_json_response = fixed_json_response

    def invoke(self, prompt):
        prompt_str = str(prompt)
        self.recorded_prompts.append(prompt_str)

        if self.fixed_json_response is not None:
            return AIMessage(content=self.fixed_json_response)

        if self.rule_fn is not None:
            return AIMessage(content=self.rule_fn(prompt_str))

        # Default fallback: uncertain
        return AIMessage(content=json.dumps([
            {"candidate": 1, "decision": "uncertain", "reason": "Default mock fallback"}
        ]))


# =====================================================================
# Scenario A: Relevant Short Reply (Context-Aware vs. Standalone)
# =====================================================================

def test_scenario_a_relevant_short_reply_context_aware_vs_standalone_comparison():
    """
    Scenario A:
    Compare context-aware grading with standalone grading for a short location reply.
    - Query: "Where are the backup cryogenic coolant canisters stored?"
    - Candidate: "Sub-level 4." (Short reply, Message ID 4)
    - Context: Preceding message ID 3 asks about cryogenic coolant canisters.

    Rule:
    - If prompt includes preceding context establishing the connection to canisters -> "relevant".
    - If candidate evaluated alone without context -> "irrelevant" (isolated fragment has no subject).

    Verification:
    - Pipeline in context-aware mode retains candidate (avoids false negative).
    - Pipeline in standalone mode rejects candidate.
    """
    query = "Where are the backup cryogenic coolant canisters stored?"
    candidate_doc = Document(
        page_content="[ID 4] [15/10/2026 10:46 AM] Specialist Lyra: Sub-level 4.",
        metadata={
            "source": "hybrid_search",
            "message_id": 4,
            "timestamp": "15/10/2026 10:46 AM",
            "sender": "Specialist Lyra",
            "hybrid_score": 0.88,
        },
    )

    mock_db_context = [
        {
            "id": 3,
            "timestamp": "2026-10-15T10:45:00",
            "sender": "Tech Kael",
            "message": "Where are the backup cryogenic coolant canisters stored?",
        },
        {
            "id": 4,
            "timestamp": "2026-10-15T10:46:00",
            "sender": "Specialist Lyra",
            "message": "Sub-level 4.",
        },
    ]

    def deterministic_short_reply_rule(prompt_str: str) -> str:
        # Check if preceding context connects candidate to the query topic
        if "Preceding Conversational Context:" in prompt_str and "cryogenic coolant canisters" in prompt_str:
            return json.dumps([{
                "candidate": 1,
                "decision": "relevant",
                "reason": "Preceding context connects location reply to the requested coolant canisters.",
            }])
        else:
            return json.dumps([{
                "candidate": 1,
                "decision": "irrelevant",
                "reason": "Standalone short fragment lacks topic context.",
            }])

    # 1. Context-Aware Run:
    mock_llm_context = ConfigurableGraderMockLLM(rule_fn=deterministic_short_reply_rule)
    state_context = make_test_state(
        documents=[Document(page_content=candidate_doc.page_content, metadata=dict(candidate_doc.metadata))],
        question=query,
    )
    with patch("rag_graph.get_chronological_context", return_value=mock_db_context) as mock_lookup:
        result_context = grade_documents(state_context, grader_llm=mock_llm_context)

    mock_lookup.assert_called_once()
    assert len(result_context["documents"]) == 1
    retained_doc = result_context["documents"][0]
    assert retained_doc.metadata["message_id"] == 4
    assert retained_doc.metadata["relevance_decision"] == "relevant"

    # Confirm prompt structure in context-aware run
    prompt_context = mock_llm_context.recorded_prompts[0]
    assert "Preceding Conversational Context:" in prompt_context
    assert "[ID 3] [15/10/2026 10:45 AM] Tech Kael: Where are the backup cryogenic coolant canisters stored?" in prompt_context
    assert "Candidate Message to Evaluate:" in prompt_context
    assert "[ID 4] [15/10/2026 10:46 AM] Specialist Lyra: Sub-level 4." in prompt_context

    # 2. Standalone Baseline Run (no preceding context available):
    mock_llm_standalone = ConfigurableGraderMockLLM(rule_fn=deterministic_short_reply_rule)
    state_standalone = make_test_state(
        documents=[Document(page_content=candidate_doc.page_content, metadata=dict(candidate_doc.metadata))],
        question=query,
    )
    with patch("rag_graph.get_chronological_context", return_value=[]):
        result_standalone = grade_documents(state_standalone, grader_llm=mock_llm_standalone)

    assert len(result_standalone["documents"]) == 0

    prompt_standalone = mock_llm_standalone.recorded_prompts[0]
    assert "Preceding Conversational Context:" not in prompt_standalone
    assert "[Candidate 1] (Single Message ID: 4)" in prompt_standalone


# =====================================================================
# Scenario B: Misleading Preceding Context
# =====================================================================

def test_scenario_b_misleading_preceding_context_separate_representation_and_rejection():
    """
    Scenario B:
    Query concerns a financial audit, but preceding message concerns lunch orders.
    - Query: "When is the annual financial audit scheduled?"
    - Candidate: "At noon." (Message ID 13)
    - Context: Preceding message ID 12 asks "Are we getting lunch delivered today?"

    Verification:
    - Grader prompt accurately isolates context from candidate.
    - Deterministic rule recognizes reply relates to lunch, not the audit.
    - Candidate is safely rejected as irrelevant (false positive prevented).
    """
    query = "When is the annual financial audit scheduled?"
    candidate_doc = Document(
        page_content="[ID 13] [15/10/2026 12:00 PM] Charlie: At noon.",
        metadata={
            "source": "hybrid_search",
            "message_id": 13,
            "timestamp": "15/10/2026 12:00 PM",
            "sender": "Charlie",
        },
    )

    mock_db_context = [
        {
            "id": 12,
            "timestamp": "2026-10-15T11:58:00",
            "sender": "Bob",
            "message": "Are we getting lunch delivered today?",
        },
        {
            "id": 13,
            "timestamp": "2026-10-15T12:00:00",
            "sender": "Charlie",
            "message": "At noon.",
        },
    ]

    def deterministic_misleading_context_rule(prompt_str: str) -> str:
        # Context shows the reply answers lunch delivery, not financial audit
        if "lunch" in prompt_str.lower() and "audit" in prompt_str.lower():
            return json.dumps([{
                "candidate": 1,
                "decision": "irrelevant",
                "reason": "Preceding context reveals time refers to lunch delivery, not the financial audit.",
            }])
        return json.dumps([{"candidate": 1, "decision": "uncertain", "reason": "Ambiguous"}])

    mock_llm = ConfigurableGraderMockLLM(rule_fn=deterministic_misleading_context_rule)
    state = make_test_state(documents=[candidate_doc], question=query)

    with patch("rag_graph.get_chronological_context", return_value=mock_db_context):
        result_state = grade_documents(state, grader_llm=mock_llm)

    # 1. Assert separate representation in prompt
    assert len(mock_llm.recorded_prompts) == 1
    prompt = mock_llm.recorded_prompts[0]
    assert "Preceding Conversational Context:" in prompt
    assert "[ID 12] [15/10/2026 11:58 AM] Bob: Are we getting lunch delivered today?" in prompt
    assert "Candidate Message to Evaluate:" in prompt
    assert "[ID 13] [15/10/2026 12:00 PM] Charlie: At noon." in prompt

    # 2. Assert candidate was rejected as irrelevant
    assert len(result_state["documents"]) == 0


# =====================================================================
# Scenario C: Candidate and Citation Integrity
# =====================================================================

def test_scenario_c_candidate_and_citation_integrity_preserved():
    """
    Scenario C:
    Verify that grading does not mutate original candidate content or metadata,
    and that preceding context IDs are NOT injected into the document or its citations.
    """
    original_content = "[ID 4] [15/10/2026 10:46 AM] Specialist Lyra: Sub-level 4."
    doc = Document(
        page_content=original_content,
        metadata={
            "source": "hybrid_search",
            "message_id": 4,
            "timestamp": "15/10/2026 10:46 AM",
            "sender": "Specialist Lyra",
            "hybrid_score": 0.94,
            "evidence_score": 0.81,
        },
    )
    state = make_test_state(documents=[doc])

    mock_context = [
        {"id": 2, "timestamp": "2026-10-15T10:44:00", "sender": "Alice", "message": "Starting inspection."},
        {"id": 3, "timestamp": "2026-10-15T10:45:00", "sender": "Tech Kael", "message": "Where are the canisters?"},
        {"id": 4, "timestamp": "2026-10-15T10:46:00", "sender": "Specialist Lyra", "message": "Sub-level 4."},
    ]

    mock_llm = ConfigurableGraderMockLLM(fixed_json_response=json.dumps([{
        "candidate": 1,
        "decision": "relevant",
        "reason": "Direct location answer.",
    }]))

    with patch("rag_graph.get_chronological_context", return_value=mock_context):
        result_state = grade_documents(state, grader_llm=mock_llm)

    assert len(result_state["documents"]) == 1
    retained_doc = result_state["documents"][0]

    # Verify content was not modified
    assert retained_doc.page_content == original_content
    # Verify metadata fields are intact
    assert retained_doc.metadata["message_id"] == 4
    assert retained_doc.metadata["hybrid_score"] == 0.94
    assert retained_doc.metadata["evidence_score"] == 0.81
    assert retained_doc.metadata["relevance_decision"] == "relevant"

    # Context IDs (2, 3) must NOT be injected into candidate metadata or content
    assert "[ID 2]" not in retained_doc.page_content
    assert "[ID 3]" not in retained_doc.page_content
    assert retained_doc.metadata.get("context_start_message_id") is None
    assert retained_doc.metadata.get("expanded_start_message_id") is None
    assert "context_messages_before" not in retained_doc.metadata


# =====================================================================
# Scenario D: Existing Behavior Remains Unchanged
# =====================================================================

def test_scenario_d_ordinary_length_messages_bypass_context_lookup():
    """
    Scenario D1:
    Ordinary-length messages (> 6 words) bypass context lookup and have no context block.
    """
    doc = Document(
        page_content="[ID 10] [15/10/2026 11:00 AM] Commander Vance: The orbital relay vessel embarks as soon as dusk falls across the platform.",
        metadata={
            "source": "hybrid_search",
            "message_id": 10,
            "timestamp": "15/10/2026 11:00 AM",
            "sender": "Commander Vance",
        },
    )
    state = make_test_state(documents=[doc])

    mock_llm = ConfigurableGraderMockLLM(fixed_json_response=json.dumps([{
        "candidate": 1,
        "decision": "relevant",
        "reason": "Contains departure schedule.",
    }]))

    with patch("rag_graph.get_chronological_context") as mock_lookup:
        result_state = grade_documents(state, grader_llm=mock_llm)

    # get_chronological_context must NOT be called
    mock_lookup.assert_not_called()
    assert len(result_state["documents"]) == 1
    prompt = mock_llm.recorded_prompts[0]
    assert "Preceding Conversational Context:" not in prompt


def test_scenario_d_conversation_chunks_bypass_context_lookup():
    """
    Scenario D2:
    Conversation chunks bypass context lookup and preserve their transcript chunk headers.
    """
    chunk_content = (
        "[ID 1] Alice: Meeting begins now.\n"
        "[ID 2] Bob: Acknowledged, joining room."
    )
    chunk_doc = Document(
        page_content=chunk_content,
        metadata={
            "source": "conversation_chunk",
            "chunk_id": 55,
            "start_message_id": 1,
            "end_message_id": 2,
        },
    )
    state = make_test_state(documents=[chunk_doc])

    mock_llm = ConfigurableGraderMockLLM(fixed_json_response=json.dumps([{
        "candidate": 1,
        "decision": "relevant",
        "reason": "Meeting start dialogue.",
    }]))

    with patch("rag_graph.get_chronological_context") as mock_lookup:
        result_state = grade_documents(state, grader_llm=mock_llm)

    mock_lookup.assert_not_called()
    assert len(result_state["documents"]) == 1
    prompt = mock_llm.recorded_prompts[0]
    assert "Preceding Conversational Context:" not in prompt
    assert "(Conversation Transcript Chunk ID: 55)" in prompt


def test_scenario_d_missing_context_or_lookup_error_falls_back_safely():
    """
    Scenario D3:
    When context lookup raises an exception or database path is missing,
    the pipeline safely falls back to evaluating candidate content alone without crashing.
    """
    doc = Document(
        page_content="[ID 4] [15/10/2026 10:46 AM] Specialist Lyra: Sub-level 4.",
        metadata={"source": "hybrid_search", "message_id": 4},
    )
    # Database path is None
    state_no_db = make_test_state(documents=[doc], database_path=None)

    mock_llm = ConfigurableGraderMockLLM(fixed_json_response=json.dumps([{
        "candidate": 1,
        "decision": "uncertain",
        "reason": "Fallback evaluation.",
    }]))

    with patch("rag_graph.get_chronological_context", side_effect=RuntimeError("SQLite disk read error")):
        result_state = grade_documents(state_no_db, grader_llm=mock_llm)

    # Uncertainty decision retains the document
    assert len(result_state["documents"]) == 1
    prompt = mock_llm.recorded_prompts[0]
    assert "Preceding Conversational Context:" not in prompt
    assert "[Candidate 1] (Single Message ID: 4)" in prompt
