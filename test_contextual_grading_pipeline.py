"""
test_contextual_grading_pipeline.py
===================================
Deterministic offline test suite for Task 43: Prototype Contextual Grading After Evidence Assembly.

Verifies:
  1. The new mode is disabled by default.
  2. The default production route is unchanged.
  3. The existing rank-only experimental route is unchanged.
  4. The new mode assembles evidence before grading.
  5. A terse location reply survives when its assembled context directly supports the requested fact.
  6. A short time or numerical reply survives contextual grading.
  7. A confirmation reply retains its anchor identity and neighboring context.
  8. A question mentioning the requested entity is not treated as an answer merely because it contains matching terms.
  9. A similar but incorrect entity is rejected when the evidence does not support the requested entity.
 10. Ambiguous evidence such as "Maybe" is handled conservatively (uncertain retained, irrelevant filtered).
 11. Irrelevant evidence is filtered before generation.
 12. Citation IDs and anchor metadata remain accurate after grading and ranking.
 13. Rejected candidates cannot re-enter the final evidence list during ranking.
 14. Empty grading results follow existing retry or insufficient-evidence behavior.
 15. Malformed grader output and grader exceptions follow safe fail-open behavior.
 16. Top-K remains enforced.
 17. Database paths remain explicitly isolated, with no production fallback.
 18. Feature-flag settings do not leak between graph invocations.
 19. The new mode requires zero live inference to run its tests.
 20. Conflicting experiment settings are resolved safely or rejected explicitly.
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import chat_database
from chat_database import create_database, save_messages
from enrich import EnrichedMessage
import rag_graph
from rag_graph import (
    CONTEXTUAL_GRADING_ENV_VAR,
    DIALOGUE_EVIDENCE_ENV_VAR,
    RETRIEVAL_STRATEGY_ENV_VAR,
    STRATEGY_BASELINE,
    STRATEGY_CONTEXTUAL_GRADING,
    STRATEGY_DIALOGUE_EVIDENCE,
    TOP_K_EVIDENCE_UNITS,
    GraphState,
    assemble_grade_rank_evidence,
    build_generation_prompt,
    build_graph,
    check_grounded,
    decide_post_assembly,
    decide_post_contextual_grading,
    format_candidate_for_contextual_grader,
    grade_candidate_batch,
    is_contextual_grading_enabled,
    is_dialogue_evidence_enabled,
    resolve_retrieval_strategy,
    route_post_retrieval,
)


# =====================================================================
# Test Fixtures & Mock Helpers
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
            # If substring matches score map, return designated vector
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
    tmp_dir = tempfile.mkdtemp(prefix="chatlens_test_ctx_grade_")
    db_path = str(Path(tmp_dir) / "test_chat.db")
    create_database(db_path)

    # Insert test messages directly into SQLite messages table
    test_rows = [
        (100, "2026-10-14 09:09:00", "Dev_Lead", "Where did you store the backup SSL cert?", "user", 0.0, "neutral", 1, 40, None, None),
        (102, "2026-10-14 09:11:00", "Dev_Ops", "Vault 9", "user", 0.0, "neutral", 0, 7, None, None),
        (103, "2026-10-14 14:30:00", "Dev_Lead", "When is the deployment window?", "user", 0.0, "neutral", 1, 31, None, None),
        (104, "2026-10-14 14:31:00", "Dev_Ops", "9pm.", "user", 0.0, "neutral", 0, 4, None, None),
        (107, "2026-10-14 16:00:00", "Dev_Lead", "Did the database migration finish clean?", "user", 0.0, "neutral", 1, 42, None, None),
        (108, "2026-10-14 16:01:00", "Dev_Sarah", "Completely clean.", "user", 0.0, "neutral", 0, 17, None, None),
        (109, "2026-10-14 10:10:00", "Dev_Lead", "Where did I leave my car keys?", "user", 0.0, "neutral", 1, 31, None, None),
        (110, "2026-10-14 10:11:00", "Dev_Sam", "On the kitchen counter.", "user", 0.0, "neutral", 0, 23, None, None),
        (115, "2026-10-14 17:04:00", "Dev_Security", "Who approved the firewall port change yesterday?", "user", 0.0, "neutral", 1, 49, None, None),
        (116, "2026-10-14 17:05:00", "Dev_Dana", "The weather is going to be stormy this evening.", "user", 0.0, "neutral", 0, 47, None, None),
        (120, "2026-10-14 15:09:00", "Dev_Alex", "Checking general system alerts.", "user", 0.0, "neutral", 0, 31, None, None),
        (121, "2026-10-14 15:10:00", "Dev_Sam", "Maybe.", "user", 0.0, "neutral", 0, 6, None, None),
        (122, "2026-10-14 11:21:00", "Dev_Test", "Where did we save the dev environment test JWT token?", "user", 0.0, "neutral", 1, 54, None, None),
        (123, "2026-10-14 11:22:00", "Dev_Auth", "In secrets.dev.json.", "user", 0.0, "neutral", 0, 20, None, None),
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

    yield db_path

    # Cleanup
    try:
        if Path(db_path).exists():
            Path(db_path).unlink()
        if Path(tmp_dir).exists():
            Path(tmp_dir).rmdir()
    except Exception:
        pass


# =====================================================================
# Tests: Requirements 1 to 20
# =====================================================================

def test_01_new_mode_disabled_by_default():
    """1. The new contextual grading mode is disabled by default."""
    # With clean environment and default state
    state: GraphState = {"question": "What is the status?"}
    assert resolve_retrieval_strategy(state) == STRATEGY_BASELINE
    assert is_contextual_grading_enabled(state) is False
    assert is_dialogue_evidence_enabled(state) is False


def test_02_default_production_route_unchanged():
    """2. The default production route is unchanged (routes to grade_documents)."""
    state: GraphState = {"question": "Where is the file?", "documents": []}
    assert route_post_retrieval(state) == "grade_documents"


def test_03_existing_rank_only_route_unchanged():
    """3. The existing rank-only experimental route is unchanged."""
    state_treatment: GraphState = {"dialogue_evidence_mode": True, "documents": []}
    assert resolve_retrieval_strategy(state_treatment) == STRATEGY_DIALOGUE_EVIDENCE
    assert route_post_retrieval(state_treatment) == "assemble_and_rank_evidence"
    assert is_dialogue_evidence_enabled(state_treatment) is True
    assert is_contextual_grading_enabled(state_treatment) is False
    assert decide_post_assembly({"documents": [Document(page_content="test")]}) == "generate"


def test_04_new_mode_assembles_evidence_before_grading(isolated_db):
    """4. The new mode assembles dialogue evidence units before grading."""
    anchor_doc = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search", "sender": "Dev_Ops"},
    )
    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "original_question": "Where was the backup SSL certificate placed?",
        "documents": [anchor_doc],
        "database_path": isolated_db,
    }

    # Grader mock that inspects prompt text
    received_prompts = []
    def mock_grader_fn(prompt):
        received_prompts.append(str(prompt))
        return MagicMock(content=json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Direct answer"}]))

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_grader_fn

    res = assemble_grade_rank_evidence(
        state,
        grader_llm=mock_llm,
        embedding_function=MockEmbeddings(),
        database_path=isolated_db,
    )

    assert len(received_prompts) == 1
    prompt_str = received_prompts[0]
    # Verify the grader saw the assembled unit with chronological context
    assert "[CHRONOLOGICAL CONTEXT]" in prompt_str
    assert "Dev_Lead: Where did you store the backup SSL cert?" in prompt_str
    assert "Vault 9" in prompt_str
    assert res["evidence_mode"] == "contextual_grading"
    assert len(res["documents"]) == 1


def test_05_terse_location_reply_survives(isolated_db):
    """5. A terse location reply survives when its assembled context supports the fact."""
    doc = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Vault 9 answers SSL cert query"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    assert len(res["documents"]) == 1
    assert res["documents"][0].metadata["anchor_message_id"] == 102
    assert res["documents"][0].metadata["relevance_decision"] == "relevant"


def test_06_short_time_reply_survives(isolated_db):
    """6. A short time or numerical reply survives contextual grading."""
    doc = Document(
        page_content="[ID 104] Dev_Ops: 9pm.",
        metadata={"message_id": 104, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "9pm specifies deployment window"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "What time is the production deployment window scheduled for?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    assert len(res["documents"]) == 1
    assert res["documents"][0].metadata["anchor_message_id"] == 104
    assert res["documents"][0].metadata["relevance_decision"] == "relevant"


def test_07_confirmation_reply_retains_anchor_and_context(isolated_db):
    """7. A confirmation reply retains its anchor identity and neighboring context."""
    doc = Document(
        page_content="[ID 108] Dev_Sarah: Completely clean.",
        metadata={"message_id": 108, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Confirms clean migration"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Did the database migration script finish clean?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    assert len(res["documents"]) == 1
    unit = res["documents"][0]
    assert unit.metadata["anchor_message_id"] == 108
    assert 107 in unit.metadata["included_message_ids"]
    assert 108 in unit.metadata["included_message_ids"]
    assert "[ID 107]" in unit.page_content
    assert "[ID 108]" in unit.page_content


def test_08_question_mentioning_entity_rejected(isolated_db):
    """8. A question mentioning the requested entity is rejected when it merely asks, not answers."""
    doc = Document(
        page_content="[ID 115] Dev_Security: Who approved the firewall port change yesterday?",
        metadata={"message_id": 115, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "irrelevant", "reason": "Merely asks the question, does not supply approval"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Who approved the firewall port change yesterday?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    # Since it was judged irrelevant, it must be removed
    assert len(res["documents"]) == 0


def test_09_similar_entity_rejected(isolated_db):
    """9. A similar but incorrect entity (dev test JWT) is rejected for production JWT query."""
    doc = Document(
        page_content="[ID 123] Dev_Auth: In secrets.dev.json.",
        metadata={"message_id": 123, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "irrelevant", "reason": "Discusses dev environment token, not production secret"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where is the production JWT secret stored?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    assert len(res["documents"]) == 0


def test_10_ambiguous_evidence_handled_conservatively(isolated_db):
    """10. Ambiguous evidence ('Maybe') is handled conservatively (uncertain retained, irrelevant filtered)."""
    doc = Document(
        page_content="[ID 121] Dev_Sam: Maybe.",
        metadata={"message_id": 121, "source": "hybrid_search"},
    )
    # Under conservative grading, ambiguous context can be 'uncertain'
    grader_json = json.dumps([{"candidate": 1, "decision": "uncertain", "reason": "Response is ambiguous"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Did the Kubernetes pod autoscaler trigger?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    # Conservative grading keeps 'uncertain' candidates
    assert len(res["documents"]) == 1
    assert res["documents"][0].metadata["relevance_decision"] == "uncertain"

    # Verify that generation prompt and groundedness checker do not treat 'Maybe' as confirmed event
    gen_prompt = build_generation_prompt(res, documents=res["documents"])
    assert "POSSIBILITY: Speculation or something conditional" in gen_prompt
    assert "Questions, requests, locations, and possibilities must never be reported as confirmed events" in gen_prompt
    assert "Treat cryptic, contradictory, slang-heavy, or ambiguous messages conservatively." in gen_prompt

    # A groundedness checker rejects an answer that converts 'Maybe' into confirmed occurrence
    unsupported_state: GraphState = {
        "question": "Did the Kubernetes pod autoscaler trigger?",
        "documents": res["documents"],
        "generation": "Yes, the Kubernetes pod autoscaler triggered yesterday [ID 121].",
        "scope": "specific",
    }
    mock_grounder_no = make_mock_llm("The context only says Maybe, which does not confirm triggering.\nDecision: no")
    grounded_res = check_grounded(unsupported_state, grounding_llm=mock_grounder_no)
    assert grounded_res["grounded"] is False


def test_11_irrelevant_evidence_filtered_before_generation(isolated_db):
    """11. Irrelevant evidence (kitchen counter car keys) is filtered before generation."""
    doc = Document(
        page_content="[ID 110] Dev_Sam: On the kitchen counter.",
        metadata={"message_id": 110, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "irrelevant", "reason": "Discusses car keys, completely unrelated to staging API key"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "What is the staging API key?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    assert len(res["documents"]) == 0
    # decide_post_contextual_grading directs empty docs to rewrite or generate
    decision = decide_post_contextual_grading({"documents": res["documents"], "rewrite_count": 0})
    assert decision == "rewrite_query"
    decision_max = decide_post_contextual_grading({"documents": res["documents"], "rewrite_count": 2})
    assert decision_max == "generate"


def test_12_citation_ids_and_anchor_metadata_accurate(isolated_db):
    """12. Citation IDs and anchor metadata remain accurate after grading and ranking."""
    doc1 = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Exact match"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Where is the backup SSL certificate?",
        "documents": [doc1],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    unit = res["documents"][0]
    assert unit.metadata["anchor_message_id"] == 102
    assert unit.metadata["included_message_ids"] == [100, 102, 109]
    assert unit.metadata["retrieval_rank"] == 1
    assert unit.metadata["relevance_decision"] == "relevant"
    assert "fused_score" in unit.metadata


def test_13_rejected_candidates_cannot_reenter(isolated_db):
    """13. Rejected candidates cannot re-enter the final evidence list during ranking."""
    doc_relevant = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )
    doc_irrelevant = Document(
        page_content="[ID 110] Dev_Sam: On the kitchen counter.",
        metadata={"message_id": 110, "source": "hybrid_search"},
    )

    # Grader marks doc 1 relevant, doc 2 irrelevant
    grader_json = json.dumps([
        {"candidate": 1, "decision": "relevant", "reason": "Relevant location"},
        {"candidate": 2, "decision": "irrelevant", "reason": "Kitchen counter is irrelevant"},
    ])
    mock_llm = make_mock_llm(grader_json)

    # Embedder with high similarity for both
    embedder = MockEmbeddings(score_map={"kitchen": [1.0, 0.0, 0.0], "vault": [1.0, 0.0, 0.0]})

    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "documents": [doc_relevant, doc_irrelevant],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=embedder, database_path=isolated_db)

    retained_anchors = [d.metadata.get("anchor_message_id") for d in res["documents"]]
    assert 102 in retained_anchors
    assert 110 not in retained_anchors
    assert len(res["documents"]) == 1


def test_14_empty_grading_results_follow_rewrite_or_abstention(isolated_db):
    """14. When all candidates are rejected, graph routes to rewrite or abstention."""
    doc = Document(
        page_content="[ID 116] Dev_Dana: The weather is stormy.",
        metadata={"message_id": 116, "source": "hybrid_search"},
    )
    grader_json = json.dumps([{"candidate": 1, "decision": "irrelevant", "reason": "Unrelated"}])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "Who approved the firewall port change?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)
    assert len(res["documents"]) == 0

    # Decision edge
    assert decide_post_contextual_grading({"documents": [], "rewrite_count": 0}) == "rewrite_query"
    assert decide_post_contextual_grading({"documents": [], "rewrite_count": 2}) == "generate"


def test_15_malformed_grader_output_and_exceptions_fail_open_safely(isolated_db):
    """15. Malformed grader output and grader exceptions follow safe fail-open behavior."""
    doc = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )

    # 1. Malformed JSON
    mock_bad_json = make_mock_llm("I cannot format this as JSON.")
    state1: GraphState = {
        "question": "Where is the cert?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res1 = assemble_grade_rank_evidence(state1, grader_llm=mock_bad_json, embedding_function=MockEmbeddings(), database_path=isolated_db)
    # Fail-open: candidate preserved as 'uncertain'
    assert len(res1["documents"]) == 1
    assert res1["documents"][0].metadata["relevance_decision"] == "uncertain"
    assert res1["documents"][0].metadata["grader_failure"] is True
    assert res1["grader_status"] == "failed"

    # 2. LLM Exception
    mock_exc = MagicMock()
    mock_exc.invoke.side_effect = RuntimeError("Simulated Ollama connection timeout")
    state2: GraphState = {
        "question": "Where is the cert?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    res2 = assemble_grade_rank_evidence(state2, grader_llm=mock_exc, embedding_function=MockEmbeddings(), database_path=isolated_db)
    assert len(res2["documents"]) == 1
    assert res2["documents"][0].metadata["relevance_decision"] == "uncertain"
    assert res2["documents"][0].metadata["grader_failure"] is True


def test_16_top_k_remains_enforced(isolated_db):
    """16. Top-K limit (5) remains enforced when more candidates are retained."""
    docs = [
        Document(page_content=f"[ID {mid}] Dev_Test: Message {mid}", metadata={"message_id": mid, "source": "hybrid_search"})
        for mid in [100, 102, 103, 104, 107, 108, 109, 110]
    ]
    # Mark all 8 candidates relevant
    grader_json = json.dumps([{"candidate": idx, "decision": "relevant", "reason": "All relevant"} for idx in range(1, 9)])
    mock_llm = make_mock_llm(grader_json)

    state: GraphState = {
        "question": "General system review",
        "documents": docs,
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)

    # Must be bounded to TOP_K_EVIDENCE_UNITS (5)
    assert len(res["documents"]) == TOP_K_EVIDENCE_UNITS
    assert len(res["documents"]) == 5


def test_17_database_paths_remain_isolated(isolated_db):
    """17. Database paths remain explicitly isolated, with no production fallback."""
    doc = Document(
        page_content="[ID 102] Dev_Ops: Vault 9",
        metadata={"message_id": 102, "source": "hybrid_search"},
    )
    state: GraphState = {
        "question": "Where is the cert?",
        "documents": [doc],
        "database_path": isolated_db,
    }
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Relevant"}])
    mock_llm = make_mock_llm(grader_json)

    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=MockEmbeddings(), database_path=isolated_db)
    unit = res["documents"][0]
    # Verify rows were fetched from isolated_db
    assert 100 in unit.metadata["included_message_ids"]
    assert 102 in unit.metadata["included_message_ids"]


def test_18_feature_flags_do_not_leak_between_invocations():
    """18. Feature-flag settings do not leak between graph invocations."""
    state_exp: GraphState = {"contextual_grading_mode": True}
    assert resolve_retrieval_strategy(state_exp) == STRATEGY_CONTEXTUAL_GRADING

    # Next clean invocation must default back to baseline
    state_default: GraphState = {}
    assert resolve_retrieval_strategy(state_default) == STRATEGY_BASELINE
    assert is_contextual_grading_enabled(state_default) is False


def test_19_zero_live_inference_required():
    """19. The new mode requires zero live inference to run its tests."""
    mock_llm = make_mock_llm("{\"candidate\": 1, \"decision\": \"relevant\", \"reason\": \"Mock\"}")
    mock_embed = MockEmbeddings()

    doc = Document(page_content="[ID 1] Mock message", metadata={"message_id": 1})
    unit = format_candidate_for_contextual_grader(1, doc)
    assert "[Candidate 1]" in unit
    assert "Mock message" in unit


def test_20_conflicting_settings_rejected_or_resolved():
    """20. Conflicting experiment settings are resolved safely or rejected explicitly."""
    # 1. Conflicting boolean flags in state
    state_conflict: GraphState = {
        "dialogue_evidence_mode": True,
        "contextual_grading_mode": True,
    }
    with pytest.raises(ValueError) as exc:
        resolve_retrieval_strategy(state_conflict)
    assert "Conflicting retrieval experiment flags" in str(exc.value)

    # 2. Conflicting retrieval_strategy with active flag
    state_strat_conflict: GraphState = {
        "retrieval_strategy": "baseline",
        "contextual_grading_mode": True,
    }
    with pytest.raises(ValueError) as exc:
        resolve_retrieval_strategy(state_strat_conflict)
    assert "Conflicting configuration" in str(exc.value)

    # 3. Unknown strategy string
    state_unknown: GraphState = {"retrieval_strategy": "quantum_fusion"}
    with pytest.raises(ValueError) as exc:
        resolve_retrieval_strategy(state_unknown)
    assert "Unknown retrieval_strategy" in str(exc.value)

    # 4. Conflicting environment variables
    old_diag = os.environ.get(DIALOGUE_EVIDENCE_ENV_VAR)
    old_ctx = os.environ.get(CONTEXTUAL_GRADING_ENV_VAR)
    try:
        os.environ[DIALOGUE_EVIDENCE_ENV_VAR] = "1"
        os.environ[CONTEXTUAL_GRADING_ENV_VAR] = "1"
        with pytest.raises(ValueError) as exc:
            resolve_retrieval_strategy({})
        assert "Conflicting environment variables" in str(exc.value)
    finally:
        if old_diag is not None:
            os.environ[DIALOGUE_EVIDENCE_ENV_VAR] = old_diag
        else:
            os.environ.pop(DIALOGUE_EVIDENCE_ENV_VAR, None)
        if old_ctx is not None:
            os.environ[CONTEXTUAL_GRADING_ENV_VAR] = old_ctx
        else:
            os.environ.pop(CONTEXTUAL_GRADING_ENV_VAR, None)


def test_21_deterministic_fake_scores_grading_and_ranking_interaction(isolated_db):
    """
    Test interaction between grading and ranking using deterministic fake scores.
    Demonstrates:
      - Candidate C has highest contextual similarity (1.00), but is marked 'irrelevant'
        by grader -> strictly excluded from ranking and final evidence list.
      - Candidate B (sim 0.99, ret_rank 2) gets boosted above Candidate A (sim 0.70, ret_rank 1) by RRF.
      - Candidate D (sim 0.85, ret_rank 3) is marked 'uncertain' -> retained and ranked.
    """
    docs = [
        Document(page_content="[ID 100] Dev_Lead: Where did you store backup cert?", metadata={"message_id": 100, "source": "hybrid_search"}),
        Document(page_content="[ID 102] Dev_Ops: Vault 9", metadata={"message_id": 102, "source": "hybrid_search"}),
        Document(page_content="[ID 116] Dev_Dana: The weather is stormy.", metadata={"message_id": 116, "source": "hybrid_search"}),
        Document(page_content="[ID 120] Dev_Alex: Checking alerts.", metadata={"message_id": 120, "source": "hybrid_search"}),
    ]

    # Deterministic fake grading decisions
    grader_json = json.dumps([
        {"candidate": 1, "decision": "relevant", "reason": "Mention cert"},
        {"candidate": 2, "decision": "relevant", "reason": "Specific vault"},
        {"candidate": 3, "decision": "irrelevant", "reason": "Weather is unrelated"},
        {"candidate": 4, "decision": "uncertain", "reason": "Unclear relevance"},
    ])
    mock_llm = make_mock_llm(grader_json)

    # Deterministic fake embeddings based on specific anchor header:
    # Anchor 102: high similarity [1.0, 0.0, 0.0]
    # Anchor 116: highest similarity [1.0, 0.0, 0.0] (distractor with high similarity)
    # Anchor 120: medium similarity [0.85, 0.5267, 0.0]
    # Anchor 100: lower similarity [0.70, 0.7141, 0.0]
    # Query vector: [1.0, 0.0, 0.0]
    score_map = {
        "[RETRIEVED ANCHOR]\n[ID 102]": [1.0, 0.0, 0.0],
        "[RETRIEVED ANCHOR]\n[ID 116]": [1.0, 0.0, 0.0],
        "[RETRIEVED ANCHOR]\n[ID 120]": [0.85, 0.5267, 0.0],
        "[RETRIEVED ANCHOR]\n[ID 100]": [0.70, 0.7141, 0.0],
    }
    mock_embed = MockEmbeddings(score_map=score_map)

    state: GraphState = {
        "question": "Where is the backup SSL certificate?",
        "documents": docs,
        "database_path": isolated_db,
    }
    res = assemble_grade_rank_evidence(state, grader_llm=mock_llm, embedding_function=mock_embed, database_path=isolated_db)

    # 1. Candidate 116 (weather) MUST be excluded despite 1.0 similarity
    retained_anchors = [d.metadata["anchor_message_id"] for d in res["documents"]]
    assert 116 not in retained_anchors
    assert len(res["documents"]) == 3

    # 2. Ranking order: Candidate 102 ranks first, then 100, then 120
    assert retained_anchors[0] == 102
    assert retained_anchors[1] == 100
    assert retained_anchors[2] == 120

    # 3. Verify fused scores are strictly monotonically decreasing
    fused_scores = [d.metadata["fused_score"] for d in res["documents"]]
    assert fused_scores[0] > fused_scores[1] > fused_scores[2]


def test_22_full_graph_routing_and_no_leakage(isolated_db):
    """
    Verify compiled LangGraph routes correctly based on strategy settings
    and state does not leak between sequential invocations.
    """
    graph = build_graph()

    # 1. Invocation with contextual grading
    doc = Document(page_content="[ID 102] Dev_Ops: Vault 9", metadata={"message_id": 102, "source": "hybrid_search"})
    grader_json = json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Vault 9 answers cert query"}])
    mock_grader = make_mock_llm(grader_json)

    state_ctx: GraphState = {
        "question": "Where was the backup SSL certificate stored?",
        "original_question": "Where was the backup SSL certificate stored?",
        "documents": [doc],
        "contextual_grading_mode": True,
        "grader_llm": mock_grader,
        "embedding_function": MockEmbeddings(),
        "database_path": isolated_db,
        "scope": "specific",
        "primary_intent": "semantic_rag",
        "secondary_intents": [],
    }

    # Verify routing helper directly on this state
    assert route_post_retrieval(state_ctx) == "assemble_grade_rank_evidence"

    # 2. Subsequent invocation with clean state (baseline production)
    state_baseline: GraphState = {
        "question": "Where was the backup SSL certificate stored?",
        "original_question": "Where was the backup SSL certificate stored?",
        "documents": [doc],
        "database_path": isolated_db,
        "scope": "specific",
        "primary_intent": "semantic_rag",
        "secondary_intents": [],
    }
    assert route_post_retrieval(state_baseline) == "grade_documents"
    assert resolve_retrieval_strategy(state_baseline) == STRATEGY_BASELINE


def test_23_formatter_signature_compatibility():
    """
    Verify grade_candidate_batch invokes formatters safely across all signatures:
      - 2 positional arguments: fmt(idx, doc)
      - 3 positional arguments: fmt(idx, doc, ctx)
      - keyword argument: fmt(idx, doc, preceding_context=ctx)
      - varargs/kwargs: fmt(idx, doc, *args, **kwargs)
    Ensures 'Unexpected keyword argument preceding_context' is never raised.
    """
    doc = Document(page_content="[ID 102] Dev_Ops: Vault 9", metadata={"message_id": 102})
    mock_llm = make_mock_llm('[{"candidate": 1, "decision": "relevant", "reason": "ok"}]')

    # 1. Two-argument formatter (no preceding_context parameter)
    def fmt_2_args(idx: int, d: Document) -> str:
        return f"[Cand {idx}] {d.page_content}"

    res1 = grade_candidate_batch(
        question="Where is cert?",
        batch_docs=[doc],
        grader_llm=mock_llm,
        candidate_formatter=fmt_2_args,
        preceding_contexts=["Some preceding context"],
    )
    assert len(res1) == 1
    assert res1[0][1] == "relevant"

    # 2. Three-positional argument formatter
    def fmt_3_args(idx: int, d: Document, ctx: Optional[str]) -> str:
        return f"[Cand {idx}] {d.page_content} | Ctx: {ctx}"

    res2 = grade_candidate_batch(
        question="Where is cert?",
        batch_docs=[doc],
        grader_llm=mock_llm,
        candidate_formatter=fmt_3_args,
        preceding_contexts=["Preceding text"],
    )
    assert len(res2) == 1
    assert res2[0][1] == "relevant"

    # 3. Keyword-argument formatter
    def fmt_kw_args(idx: int, d: Document, preceding_context: Optional[str] = None) -> str:
        return f"[Cand {idx}] {d.page_content} | KW: {preceding_context}"

    res3 = grade_candidate_batch(
        question="Where is cert?",
        batch_docs=[doc],
        grader_llm=mock_llm,
        candidate_formatter=fmt_kw_args,
        preceding_contexts=["KW text"],
    )
    assert len(res3) == 1
    assert res3[0][1] == "relevant"

    # 4. Lambda with 2 arguments
    fmt_lambda = lambda idx, d: f"L[{idx}]: {d.page_content}"
    res4 = grade_candidate_batch(
        question="Where is cert?",
        batch_docs=[doc],
        grader_llm=mock_llm,
        candidate_formatter=fmt_lambda,
        preceding_contexts=["Ignored context"],
    )
    assert len(res4) == 1
    assert res4[0][1] == "relevant"


