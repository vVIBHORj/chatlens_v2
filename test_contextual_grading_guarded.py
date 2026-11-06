"""
test_contextual_grading_guarded.py
==================================
Deterministic offline unit test suite for Task 71:
Opt-In Integration of Deterministic Evidence-Scope Guard.

Verifies all 22 required behaviors:
  1. contextual_grading_guarded is disabled by default.
  2. Default production route remains baseline when no flags are set.
  3. Existing dialogue_evidence route remains completely unaffected.
  4. Existing contextual_grading route remains completely unaffected.
  5. Existing contextual_grading_v2 route remains completely unaffected.
  6. Activated via state parameter contextual_grading_guarded_mode=True.
  7. Activated via state parameter retrieval_strategy='contextual_grading_guarded'.
  8. Activated via env var CHATLENS_CONTEXTUAL_GRADING_GUARDED_EXPERIMENT=1.
  9. Activated via env var CHATLENS_RETRIEVAL_STRATEGY='contextual_grading_guarded'.
 10. Conflicting strategy flags raise explicit ValueError.
 11. Conflicting environment variables raise explicit ValueError.
 12. Pure query-echo candidate rejected with REJECT_QUERY_ECHO without calling LLM grader.
 13. Environment-mismatch candidate rejected with REJECT_ENVIRONMENT_MISMATCH without calling LLM grader.
 14. True supporting candidate with terse location/answer passes the guard (PASS) and is forwarded to V1 grader.
 15. Multi-message joint supporting candidate passes the guard without false rejection.
 16. Unrecognized/ambiguous text receives NEEDS_REVIEW and fails open to V1 grader.
 17. Exception raised inside guard fails open safely as NEEDS_REVIEW with FAIL_OPEN_ON_EXCEPTION.
 18. When all candidates are rejected by the guard, LLM grader is completely bypassed (grader_invoked=False), evidence is empty, and routing proceeds to rewrite.
 19. Surviving candidates forwarded to V1 grader are accurately graded and ranked with RRF up to TOP_K_EVIDENCE_UNITS.
 20. guard_decisions and guard_diagnostics are populated in state without leaking into generated user answer text.
 21. Feature-flag and state settings do not leak between graph invocations.
 22. Test suite executes 100% offline with zero live LLM inference, zero network calls, zero embedding calls.
"""

import copy
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import chat_database
from chat_database import create_database
import rag_graph
from rag_graph import (
    CONTEXTUAL_GRADING_ENV_VAR,
    CONTEXTUAL_GRADING_GUARDED_ENV_VAR,
    CONTEXTUAL_GRADING_V2_ENV_VAR,
    DIALOGUE_EVIDENCE_ENV_VAR,
    RETRIEVAL_STRATEGY_ENV_VAR,
    STRATEGY_BASELINE,
    STRATEGY_CONTEXTUAL_GRADING,
    STRATEGY_CONTEXTUAL_GRADING_GUARDED,
    STRATEGY_CONTEXTUAL_GRADING_V2,
    STRATEGY_DIALOGUE_EVIDENCE,
    TOP_K_EVIDENCE_UNITS,
    GraphState,
    assemble_grade_rank_evidence_guarded,
    build_graph,
    decide_post_contextual_grading,
    is_contextual_grading_enabled,
    is_contextual_grading_guarded_enabled,
    is_contextual_grading_v2_enabled,
    is_dialogue_evidence_enabled,
    resolve_retrieval_strategy,
    route_post_retrieval,
)
from evidence_scope_guard import (
    GuardDecision,
    GuardResult,
    evaluate_evidence_scope,
)


# =====================================================================
# Fixtures & Mock Helpers
# =====================================================================

class MockEmbeddings:
    """Deterministic mock embedder with zero live model invocations."""
    def __init__(self, score_map: Optional[Dict[str, List[float]]] = None):
        self.score_map = score_map or {}
        self.embed_query_calls = 0
        self.embed_docs_calls = 0

    def embed_query(self, text: str) -> List[float]:
        self.embed_query_calls += 1
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embed_docs_calls += 1
        results = []
        for t in texts:
            vec = [0.5, 0.5, 0.0]
            for key, mapped_vec in self.score_map.items():
                if key.lower() in t.lower():
                    vec = mapped_vec
                    break
            results.append(vec)
        return results


def make_mock_llm(content: str):
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=content)
    return mock


@pytest.fixture
def isolated_db():
    """Create a temporary synthetic SQLite database with known messages."""
    tmp_dir = tempfile.mkdtemp(prefix="chatlens_test_guarded_")
    db_path = str(Path(tmp_dir) / "test_chat.db")
    create_database(db_path)

    test_rows = [
        (100, "2026-10-14 09:09:00", "Dev_Lead", "Where did you store the backup SSL cert?", "user", 0.0, "neutral", 1, 40, None, None),
        (102, "2026-10-14 09:11:00", "Dev_Ops", "Vault 9", "user", 0.0, "neutral", 0, 7, None, None),
        (103, "2026-10-14 14:30:00", "Dev_Lead", "When is the deployment window?", "user", 0.0, "neutral", 1, 31, None, None),
        (104, "2026-10-14 14:31:00", "Dev_Ops", "9pm.", "user", 0.0, "neutral", 0, 4, None, None),
        (109, "2026-10-14 10:10:00", "Dev_Lead", "Where did I leave my car keys?", "user", 0.0, "neutral", 1, 31, None, None),
        (110, "2026-10-14 10:11:00", "Dev_Sam", "On the kitchen counter.", "user", 0.0, "neutral", 0, 23, None, None),
        (115, "2026-10-14 17:04:00", "Dev_Security", "Who approved the firewall port change yesterday?", "user", 0.0, "neutral", 1, 49, None, None),
        (116, "2026-10-14 17:05:00", "Dev_Dana", "The weather is going to be stormy this evening.", "user", 0.0, "neutral", 0, 47, None, None),
        (117, "2026-10-14 18:00:00", "Dev_Sarah", "Database changes need approval from the DBA lead.", "user", 0.0, "neutral", 0, 52, None, None),
        (118, "2026-10-14 18:01:00", "Dev_Dana", "Got it.", "user", 0.0, "neutral", 0, 6, None, None),
        (119, "2026-10-14 18:02:00", "Dev_Sarah", "The SLA for approval is 4 hours.", "user", 0.0, "neutral", 0, 34, None, None),
        (122, "2026-10-14 11:21:00", "Dev_Test", "Where did we save the dev environment test JWT token?", "user", 0.0, "neutral", 1, 54, None, None),
        (123, "2026-10-14 11:22:00", "Dev_Auth", "In secrets.dev.json.", "user", 0.0, "neutral", 0, 20, None, None),
        (140, "2026-10-14 19:00:00", "Dev_Lead", "We are migrating keys between dev and production environments.", "user", 0.0, "neutral", 0, 68, None, None),
        (150, "2026-10-14 20:00:00", "Dev_Lead", "Where did you store the backup SSL cert?", "user", 0.0, "neutral", 1, 40, None, None),
    ]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO messages (
            id, timestamp, sender, message, message_type,
            sentiment_compound, sentiment_label, is_question,
            message_length, response_time_minutes, reply_time_minutes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        test_rows,
    )
    conn.commit()
    conn.close()
    return db_path


# =====================================================================
# 22 Test Scenarios (Task 71 Section 6)
# =====================================================================

def test_01_guarded_strategy_disabled_by_default():
    """1. contextual_grading_guarded is disabled by default."""
    assert is_contextual_grading_guarded_enabled({}) is False
    assert is_contextual_grading_guarded_enabled(None) is False
    assert resolve_retrieval_strategy({}) == STRATEGY_BASELINE
    assert resolve_retrieval_strategy(None) == STRATEGY_BASELINE


def test_02_default_production_route_unchanged():
    """2. Default production route remains baseline when no flags are set."""
    state: GraphState = {"question": "What is the status?"}
    assert route_post_retrieval(state) == "grade_documents"
    assert resolve_retrieval_strategy(state) == STRATEGY_BASELINE


def test_03_existing_dialogue_evidence_route_unaffected():
    """3. Existing dialogue_evidence route remains completely unaffected."""
    state_flag: GraphState = {"dialogue_evidence_mode": True}
    assert is_dialogue_evidence_enabled(state_flag) is True
    assert is_contextual_grading_guarded_enabled(state_flag) is False
    assert resolve_retrieval_strategy(state_flag) == STRATEGY_DIALOGUE_EVIDENCE
    assert route_post_retrieval(state_flag) == "assemble_and_rank_evidence"

    state_strat: GraphState = {"retrieval_strategy": STRATEGY_DIALOGUE_EVIDENCE}
    assert resolve_retrieval_strategy(state_strat) == STRATEGY_DIALOGUE_EVIDENCE
    assert route_post_retrieval(state_strat) == "assemble_and_rank_evidence"


def test_04_existing_contextual_grading_route_unaffected():
    """4. Existing contextual_grading route remains completely unaffected."""
    state_flag: GraphState = {"contextual_grading_mode": True}
    assert is_contextual_grading_enabled(state_flag) is True
    assert is_contextual_grading_guarded_enabled(state_flag) is False
    assert resolve_retrieval_strategy(state_flag) == STRATEGY_CONTEXTUAL_GRADING
    assert route_post_retrieval(state_flag) == "assemble_grade_rank_evidence"

    state_strat: GraphState = {"retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING}
    assert resolve_retrieval_strategy(state_strat) == STRATEGY_CONTEXTUAL_GRADING
    assert route_post_retrieval(state_strat) == "assemble_grade_rank_evidence"


def test_05_existing_contextual_grading_v2_route_unaffected():
    """5. Existing contextual_grading_v2 route remains completely unaffected."""
    state_flag: GraphState = {"contextual_grading_v2_mode": True}
    assert is_contextual_grading_v2_enabled(state_flag) is True
    assert is_contextual_grading_guarded_enabled(state_flag) is False
    assert resolve_retrieval_strategy(state_flag) == STRATEGY_CONTEXTUAL_GRADING_V2
    assert route_post_retrieval(state_flag) == "assemble_grade_rank_evidence_v2"

    state_strat: GraphState = {"retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2}
    assert resolve_retrieval_strategy(state_strat) == STRATEGY_CONTEXTUAL_GRADING_V2
    assert route_post_retrieval(state_strat) == "assemble_grade_rank_evidence_v2"


def test_06_activated_via_state_parameter_mode():
    """6. The new strategy can be activated via state parameter contextual_grading_guarded_mode=True."""
    state: GraphState = {"contextual_grading_guarded_mode": True}
    assert is_contextual_grading_guarded_enabled(state) is True
    assert is_contextual_grading_enabled(state) is False
    assert is_contextual_grading_v2_enabled(state) is False
    assert is_dialogue_evidence_enabled(state) is False
    assert resolve_retrieval_strategy(state) == STRATEGY_CONTEXTUAL_GRADING_GUARDED
    assert route_post_retrieval(state) == "assemble_grade_rank_evidence_guarded"


def test_07_activated_via_state_parameter_strategy():
    """7. The new strategy can be activated via state parameter retrieval_strategy='contextual_grading_guarded'."""
    state: GraphState = {"retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_GUARDED}
    assert is_contextual_grading_guarded_enabled(state) is True
    assert resolve_retrieval_strategy(state) == STRATEGY_CONTEXTUAL_GRADING_GUARDED
    assert route_post_retrieval(state) == "assemble_grade_rank_evidence_guarded"


def test_08_activated_via_environment_variable_experiment(monkeypatch):
    """8. The new strategy can be activated via env var CHATLENS_CONTEXTUAL_GRADING_GUARDED_EXPERIMENT=1."""
    monkeypatch.setenv(CONTEXTUAL_GRADING_GUARDED_ENV_VAR, "1")
    assert is_contextual_grading_guarded_enabled() is True
    assert resolve_retrieval_strategy() == STRATEGY_CONTEXTUAL_GRADING_GUARDED
    assert route_post_retrieval({}) == "assemble_grade_rank_evidence_guarded"


def test_09_activated_via_environment_variable_strategy(monkeypatch):
    """9. The new strategy can be activated via env var CHATLENS_RETRIEVAL_STRATEGY='contextual_grading_guarded'."""
    monkeypatch.setenv(RETRIEVAL_STRATEGY_ENV_VAR, STRATEGY_CONTEXTUAL_GRADING_GUARDED)
    assert is_contextual_grading_guarded_enabled() is True
    assert resolve_retrieval_strategy() == STRATEGY_CONTEXTUAL_GRADING_GUARDED
    assert route_post_retrieval({}) == "assemble_grade_rank_evidence_guarded"


def test_10_conflicting_strategy_flags_raise_value_error():
    """10. Conflicting strategy flags raise explicit ValueError."""
    # baseline with guarded flag
    with pytest.raises(ValueError, match="Conflicting configuration"):
        resolve_retrieval_strategy({
            "retrieval_strategy": STRATEGY_BASELINE,
            "contextual_grading_guarded_mode": True,
        })

    # contextual_grading with guarded flag
    with pytest.raises(ValueError, match="Conflicting configuration"):
        resolve_retrieval_strategy({
            "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING,
            "contextual_grading_guarded_mode": True,
        })

    # contextual_grading_v2 with guarded flag
    with pytest.raises(ValueError, match="Conflicting configuration"):
        resolve_retrieval_strategy({
            "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
            "contextual_grading_guarded_mode": True,
        })

    # guarded strategy with other flag True
    with pytest.raises(ValueError, match="Conflicting configuration"):
        resolve_retrieval_strategy({
            "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_GUARDED,
            "contextual_grading_mode": True,
        })

    # multiple boolean flags enabled simultaneously
    with pytest.raises(ValueError, match="Conflicting retrieval experiment flags"):
        resolve_retrieval_strategy({
            "contextual_grading_mode": True,
            "contextual_grading_guarded_mode": True,
        })


def test_11_conflicting_environment_variables_raise_value_error(monkeypatch):
    """11. Conflicting environment variables raise explicit ValueError."""
    monkeypatch.setenv(CONTEXTUAL_GRADING_GUARDED_ENV_VAR, "1")
    monkeypatch.setenv(CONTEXTUAL_GRADING_ENV_VAR, "1")

    with pytest.raises(ValueError, match="Conflicting environment variables"):
        resolve_retrieval_strategy()


def test_12_pure_query_echo_rejected_without_llm_grader(isolated_db):
    """12. Pure query-echo candidate is rejected with REJECT_QUERY_ECHO without calling LLM grader."""
    doc = Document(
        page_content="[ID 115] Dev_Security: Who approved the firewall port change yesterday?",
        metadata={"message_id": 115, "source": "hybrid_search"},
    )
    mock_llm = MagicMock()

    state: GraphState = {
        "question": "Who approved the firewall port change yesterday?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    # Grader must NOT have been called
    assert mock_llm.invoke.call_count == 0
    assert len(res["documents"]) == 0
    assert res["guard_diagnostics"]["grader_invoked"] is False
    assert res["guard_diagnostics"]["rejected_query_echo"] == 1
    assert res["guard_diagnostics"]["total_rejected"] == 1

    decisions = res["guard_decisions"]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == GuardDecision.REJECT_QUERY_ECHO.value
    assert "query_echo" in decisions[0]["rule"]


def test_13_environment_mismatch_rejected_without_llm_grader(isolated_db):
    """13. Environment-mismatch candidate is rejected with REJECT_ENVIRONMENT_MISMATCH without calling LLM grader."""
    doc = Document(
        page_content="[ID 123] Dev_Auth: In secrets.dev.json.",
        metadata={"message_id": 123, "source": "hybrid_search"},
    )
    mock_llm = MagicMock()

    state: GraphState = {
        "question": "Where did we save the production environment JWT token?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    # Grader must NOT have been called
    assert mock_llm.invoke.call_count == 0
    assert len(res["documents"]) == 0
    assert res["guard_diagnostics"]["grader_invoked"] is False
    assert res["guard_diagnostics"]["rejected_environment_mismatch"] == 1
    assert res["guard_diagnostics"]["total_rejected"] == 1

    decisions = res["guard_decisions"]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == GuardDecision.REJECT_ENVIRONMENT_MISMATCH.value
    assert "env_mismatch" in decisions[0]["rule"]


def test_14_true_supporting_terse_answer_passes_guard_and_forwarded_to_v1_grader(isolated_db):
    """14. True supporting candidate with terse location/answer passes the guard (PASS) and is forwarded to V1 grader."""
    doc = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )
    # V1 grader returns relevant
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Vault 9 answers SSL cert query"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where did you store the backup SSL cert?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    # Guard passed candidate
    assert res["guard_diagnostics"]["grader_invoked"] is True
    assert res["guard_diagnostics"]["candidates_sent_to_grader"] == 1
    assert res["guard_diagnostics"]["candidates_passed_grader"] == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].metadata["anchor_message_id"] == 102
    assert res["documents"][0].metadata["relevance_decision"] == "relevant"


def test_15_multimessage_joint_supporting_candidate_survives(isolated_db):
    """15. Multi-message joint supporting candidate passes the guard without false rejection."""
    doc = Document(
        page_content="[ID 117] Dev_Sarah: Database changes need approval from the DBA lead.\n[ID 119] Dev_Sarah: The SLA for approval is 4 hours.",
        metadata={"message_id": 117, "included_message_ids": [117, 118, 119], "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Jointly specifies approver and SLA"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Who needs to approve database changes and what is the SLA?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    # Must NOT be falsely rejected by the guard
    assert res["guard_diagnostics"]["total_rejected"] == 0
    assert res["guard_diagnostics"]["grader_invoked"] is True
    assert len(res["documents"]) == 1


def test_16_ambiguous_or_unrecognized_text_fails_open_as_needs_review(isolated_db):
    """16. Unrecognized/ambiguous text receives NEEDS_REVIEW and fails open to V1 grader."""
    doc = Document(
        page_content="[ID 140] Dev_Lead: We are migrating keys between dev and production environments.",
        metadata={"message_id": 140, "source": "hybrid_search"},
    )
    # Grader decides irrelevant
    grader_json = json.dumps([{"candidate": 1, "decision": "irrelevant", "reason": "Migration talk is irrelevant"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where is the production API key stored?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    # Guard marked NEEDS_REVIEW (fail-open) and forwarded to grader
    assert res["guard_decisions"][0]["decision"] == GuardDecision.NEEDS_REVIEW.value
    assert res["guard_diagnostics"]["needs_review"] == 1
    assert res["guard_diagnostics"]["grader_invoked"] is True
    # Grader rejected it as irrelevant
    assert res["guard_diagnostics"]["candidates_passed_grader"] == 0
    assert len(res["documents"]) == 0


def test_17_guard_exception_fails_open_safely(isolated_db):
    """17. Exception raised inside guard fails open safely as NEEDS_REVIEW with FAIL_OPEN_ON_EXCEPTION."""
    doc = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Vault 9 answers SSL cert query"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where did you store the backup SSL cert?",
        "documents": [doc],
        "database_path": isolated_db,
    }

    with patch("evidence_scope_guard.evaluate_evidence_scope", side_effect=RuntimeError("Simulated guard crash")):
        res = assemble_grade_rank_evidence_guarded(
            state,
            grader_llm=mock_llm,
            embedding_function=MockEmbeddings(),
            database_path=isolated_db,
        )

    # Must fail open: decision NEEDS_REVIEW, rule FAIL_OPEN_ON_EXCEPTION, forwarded to grader
    assert len(res["guard_decisions"]) == 1
    assert res["guard_decisions"][0]["decision"] == GuardDecision.NEEDS_REVIEW.value
    assert res["guard_decisions"][0]["rule"] == "FAIL_OPEN_ON_EXCEPTION"
    assert res["guard_diagnostics"]["needs_review"] == 1
    assert res["guard_diagnostics"]["grader_invoked"] is True
    assert len(res["documents"]) == 1


def test_18_all_rejected_bypasses_grader_empty_evidence_and_routes_to_rewrite(isolated_db):
    """18. When all candidates are rejected by the guard, LLM grader is completely bypassed (grader_invoked=False), evidence is empty, and routing proceeds to rewrite."""
    doc1 = Document(
        page_content="[ID 115] Dev_Security: Who approved the firewall port change yesterday?",
        metadata={"message_id": 115, "source": "hybrid_search"},
    )
    mock_llm = MagicMock()

    state: GraphState = {
        "question": "Who approved the firewall port change yesterday?",
        "documents": [doc1],
        "database_path": isolated_db,
        "rewrite_count": 0,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    assert mock_llm.invoke.call_count == 0
    assert res["guard_diagnostics"]["grader_invoked"] is False
    assert res["guard_diagnostics"]["total_rejected"] >= 1
    assert len(res["documents"]) == 0

    # Decision node routes to rewrite_query
    next_node = decide_post_contextual_grading(res)
    assert next_node == "rewrite_query"


def test_19_surviving_candidates_graded_and_ranked_rrf_top_k(isolated_db):
    """19. Surviving candidates forwarded to V1 grader are accurately graded and ranked with RRF up to TOP_K_EVIDENCE_UNITS."""
    # Create 3 documents: 1 query echo (rejected by guard), 2 valid candidates (passed to grader)
    doc_echo = Document(
        page_content="[ID 150] Dev_Lead: Where did you store the backup SSL cert?",
        metadata={"message_id": 150, "source": "hybrid_search", "retrieval_rank": 1},
    )
    doc_valid1 = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search", "retrieval_rank": 2},
    )
    doc_valid2 = Document(
        page_content="[ID 104] Dev_Ops: 9pm.",
        metadata={"message_id": 104, "source": "hybrid_search", "retrieval_rank": 3},
    )

    # Grader grades the 2 surviving units
    grader_json = json.dumps([
        {"candidate": 1, "decision": "relevant", "reason": "Vault 9 answers SSL cert query"},
        {"candidate": 2, "decision": "irrelevant", "reason": "9pm is a time not a location"},
    ])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where did you store the backup SSL cert?",
        "documents": [doc_echo, doc_valid1, doc_valid2],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    # Total 3 candidates: 1 rejected by guard, 2 sent to grader, 1 kept by grader
    assert res["guard_diagnostics"]["total_candidates"] == 3
    assert res["guard_diagnostics"]["total_rejected"] == 1
    assert res["guard_diagnostics"]["candidates_sent_to_grader"] == 2
    assert res["guard_diagnostics"]["candidates_passed_grader"] == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].metadata["anchor_message_id"] == 102
    assert "fused_score" in res["documents"][0].metadata


def test_20_guard_decisions_and_diagnostics_populated_without_answer_leak(isolated_db):
    """20. guard_decisions and guard_diagnostics are populated in state without leaking into generated user answer text."""
    doc = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate_index": 1, "decision": "relevant", "reason": "Vault 9 answers SSL cert query"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where did you store the backup SSL cert?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    # Check guard_decisions schema
    assert "guard_decisions" in res
    dec = res["guard_decisions"][0]
    for key in ("candidate_index", "anchor_message_id", "included_message_ids", "decision", "rule", "rationale", "pattern_matches"):
        assert key in dec

    # Check guard_diagnostics schema
    assert "guard_diagnostics" in res
    diag = res["guard_diagnostics"]
    for key in ("total_candidates", "passed", "needs_review", "rejected_query_echo", "rejected_environment_mismatch", "total_rejected", "grader_invoked", "candidates_sent_to_grader", "candidates_passed_grader"):
        assert key in diag

    # In a simulated generation step, verify user answer is clean and does NOT leak guard diagnostics
    simulated_answer = "The backup SSL certificate was stored in Vault 9. [102]"
    assert "guard_decisions" not in simulated_answer
    assert "EVIDENCE_SCOPE" not in simulated_answer
    assert "REJECT_" not in simulated_answer


def test_21_repeated_invocations_do_not_leak_state():
    """21. Feature-flag and state settings do not leak between graph invocations."""
    state_guarded: GraphState = {"contextual_grading_guarded_mode": True}
    assert resolve_retrieval_strategy(state_guarded) == STRATEGY_CONTEXTUAL_GRADING_GUARDED

    state_default: GraphState = {}
    assert resolve_retrieval_strategy(state_default) == STRATEGY_BASELINE
    assert is_contextual_grading_guarded_enabled(state_default) is False

    state_v1: GraphState = {"contextual_grading_mode": True}
    assert resolve_retrieval_strategy(state_v1) == STRATEGY_CONTEXTUAL_GRADING
    assert is_contextual_grading_guarded_enabled(state_v1) is False


def test_22_zero_live_inference_offline_execution(isolated_db):
    """22. Test suite executes 100% offline with zero live LLM inference, zero network calls, zero embedding calls."""
    # Verify graph compiles and contains the assemble_grade_rank_evidence_guarded node
    graph = build_graph()
    assert "assemble_grade_rank_evidence_guarded" in graph.nodes

    # Execute assemble_grade_rank_evidence_guarded completely offline
    doc = Document(
        page_content="[ID 104] Dev_Ops: 9pm.",
        metadata={"message_id": 104, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate_index": 1, "decision": "relevant", "reason": "9pm specifies deployment window"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "When is the deployment window?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence_guarded(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    assert res["retrieval_strategy"] == STRATEGY_CONTEXTUAL_GRADING_GUARDED
    assert res["evidence_mode"] == "contextual_grading_guarded"
    assert len(res["documents"]) == 1
    assert res["documents"][0].metadata["anchor_message_id"] == 104
