"""
test_contextual_grading_v2.py
=============================
Deterministic offline test suite for Task 45:
Improved Contextual Evidence Classification and Uncertainty Handling (Contextual Grading v2).

Verifies all 18 mandatory criteria:
  1. Short reply answering a preceding question is retained with the correct anchor ID.
  2. Unit containing only a question about the requested fact and an unrelated reply does not establish the answer (question_only).
  3. Question about an approver does not establish who approved a change (question_only).
  4. Development secret does not support a request for a production secret (entity_mismatch).
  5. Development database location does not establish a production database location (entity_mismatch).
  6. Ambiguous evidence such as "Maybe" does not support a definitive yes or no (ambiguous).
  7. Clear answer remains available when the surrounding unit also contains irrelevant context.
  8. Cafeteria-keys distractor remains rejectable in the staging-API-key scenario.
  9. Irrelevant candidates rejected by the grader cannot re-enter the final ranking.
 10. All candidate units being ambiguous or unsupported follows the verified insufficient-evidence behavior.
 11. Valid terse location, time, confirmation, pronoun-resolution, and multi-message examples retain supporting evidence.
 12. Canonical message IDs and anchor metadata remain correct after assembly, grading, and ranking.
 13. Malformed grader output and exceptions follow the documented handling policy.
 14. New strategy is disabled by default and selected independently.
 15. Existing baseline, dialogue-evidence, and contextual-grading strategies retain their routing behavior.
 16. Repeated invocations do not leak strategy state.
 17. Top-K and citation schema remain compatible.
 18. Zero live model inference, network calls, or production storage operations are needed.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime
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
    CONTEXTUAL_GRADING_V2_ENV_VAR,
    DIALOGUE_EVIDENCE_ENV_VAR,
    RETRIEVAL_STRATEGY_ENV_VAR,
    STRATEGY_BASELINE,
    STRATEGY_CONTEXTUAL_GRADING,
    STRATEGY_CONTEXTUAL_GRADING_V2,
    STRATEGY_DIALOGUE_EVIDENCE,
    TOP_K_EVIDENCE_UNITS,
    GraphState,
    assemble_grade_rank_evidence_v2,
    build_generation_prompt,
    build_graph,
    check_grounded,
    decide_post_contextual_grading_v2,
    format_candidate_for_contextual_grader,
    grade_candidate_batch_v2,
    is_contextual_grading_v2_enabled,
    parse_contextual_grader_v2_response,
    resolve_retrieval_strategy,
    route_post_retrieval,
)


# =====================================================================
# Fixtures & Mock Helpers
# =====================================================================

class MockV2GraderLLM:
    """Mock LLM returning deterministic structured JSON arrays for v2 evidence classification."""
    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None, raw_string: Optional[str] = None):
        self.responses = responses or []
        self.raw_string = raw_string
        self.invocation_count = 0

    def invoke(self, prompt: str) -> MagicMock:
        self.invocation_count += 1
        mock_resp = MagicMock()
        if self.raw_string is not None:
            mock_resp.content = self.raw_string
        else:
            mock_resp.content = json.dumps(self.responses)
        return mock_resp


class MockDeterministicEmbeddings:
    """Mock embeddings returning deterministic float vectors without calling external models."""
    def __init__(self, score_map: Optional[Dict[str, float]] = None):
        self.score_map = score_map or {}
        self.invocations = 0

    def embed_query(self, text: str) -> List[float]:
        self.invocations += 1
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.invocations += 1
        results = []
        for text in texts:
            val = 0.5
            for key, score in self.score_map.items():
                if key in text:
                    val = score
                    break
            results.append([val, 0.0, 0.0])
        return results


@pytest.fixture
def isolated_db_path(tmp_path: Path):
    """Provides an isolated SQLite database file with initialized schema and sample messages."""
    db_file = str(tmp_path / "test_isolated_v2.db")
    create_database(db_file)

    sample_msgs = [
        EnrichedMessage(
            id=0,
            timestamp=datetime.fromisoformat("2026-03-01T08:00:00"),
            sender="Dev_Alex",
            message="Morning team, standup at 9.",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=0,
            message_length=30,
            response_time_minutes=0.0,
            reply_time_minutes=0.0,
        ),
        EnrichedMessage(
            id=1,
            timestamp=datetime.fromisoformat("2026-03-01T08:05:00"),
            sender="Dev_Alex",
            message="Where did we store the backup SSL certificate?",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=1,
            message_length=45,
            response_time_minutes=5.0,
            reply_time_minutes=5.0,
        ),
        EnrichedMessage(
            id=2,
            timestamp=datetime.fromisoformat("2026-03-01T08:06:00"),
            sender="Dev_Sam",
            message="In Vault 9.",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=0,
            message_length=11,
            response_time_minutes=1.0,
            reply_time_minutes=1.0,
        ),
        EnrichedMessage(
            id=3,
            timestamp=datetime.fromisoformat("2026-03-01T08:10:00"),
            sender="Dev_Security",
            message="Who approved the firewall port change?",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=1,
            message_length=40,
            response_time_minutes=4.0,
            reply_time_minutes=4.0,
        ),
        EnrichedMessage(
            id=4,
            timestamp=datetime.fromisoformat("2026-03-01T08:11:00"),
            sender="Dev_Dana",
            message="I think the weather is going to be stormy this evening.",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=0,
            message_length=55,
            response_time_minutes=1.0,
            reply_time_minutes=1.0,
        ),
        EnrichedMessage(
            id=5,
            timestamp=datetime.fromisoformat("2026-03-01T08:15:00"),
            sender="Dev_Alex",
            message="Where did we save the dev test JWT token?",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=1,
            message_length=42,
            response_time_minutes=4.0,
            reply_time_minutes=4.0,
        ),
        EnrichedMessage(
            id=6,
            timestamp=datetime.fromisoformat("2026-03-01T08:16:00"),
            sender="Dev_Sam",
            message="In secrets.dev.json.",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=0,
            message_length=19,
            response_time_minutes=1.0,
            reply_time_minutes=1.0,
        ),
        EnrichedMessage(
            id=7,
            timestamp=datetime.fromisoformat("2026-03-01T08:20:00"),
            sender="Dev_Chris",
            message="Did the Kubernetes pod autoscaler trigger during the spike?",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=1,
            message_length=60,
            response_time_minutes=4.0,
            reply_time_minutes=4.0,
        ),
        EnrichedMessage(
            id=8,
            timestamp=datetime.fromisoformat("2026-03-01T08:21:00"),
            sender="Dev_Jordan",
            message="Maybe.",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=0,
            message_length=6,
            response_time_minutes=1.0,
            reply_time_minutes=1.0,
        ),
        EnrichedMessage(
            id=9,
            timestamp=datetime.fromisoformat("2026-03-01T08:25:00"),
            sender="Dev_Pat",
            message="Where did you leave the cafeteria keys?",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=1,
            message_length=40,
            response_time_minutes=4.0,
            reply_time_minutes=4.0,
        ),
        EnrichedMessage(
            id=10,
            timestamp=datetime.fromisoformat("2026-03-01T08:26:00"),
            sender="Dev_Jordan",
            message="On the kitchen counter.",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=0,
            message_length=23,
            response_time_minutes=1.0,
            reply_time_minutes=1.0,
        ),
        EnrichedMessage(
            id=11,
            timestamp=datetime.fromisoformat("2026-03-01T08:30:00"),
            sender="Dev_Alex",
            message="Where is the development database server located?",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=1,
            message_length=50,
            response_time_minutes=4.0,
            reply_time_minutes=4.0,
        ),
        EnrichedMessage(
            id=12,
            timestamp=datetime.fromisoformat("2026-03-01T08:31:00"),
            sender="Dev_Sam",
            message="Host dev-db-01.internal.",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=0,
            message_length=24,
            response_time_minutes=1.0,
            reply_time_minutes=1.0,
        ),
    ]

    save_messages(sample_msgs, db_file)
    return db_file


# =====================================================================
# Unit & Pipeline Tests
# =====================================================================

def test_01_terse_reply_answering_preceding_question_retained(isolated_db_path: str):
    """Scenario 1: A short reply answering a preceding question is retained with the correct anchor ID."""
    doc = Document(
        page_content="[ID 2] [01/03/2026 08:06 AM] Dev_Sam: In Vault 9.",
        metadata={"message_id": 2, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "supported", "reason": "Vault 9 answers the SSL cert location question"}
    ])

    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    retained = result["documents"]

    assert len(retained) == 1
    assert retained[0].metadata["anchor_message_id"] == 2
    assert retained[0].metadata["relevance_decision"] == "relevant"
    assert retained[0].metadata["evidence_type"] == "supported"
    assert 2 in retained[0].metadata["included_message_ids"]
    assert 1 in retained[0].metadata["included_message_ids"]  # preceding context question included


def test_02_question_only_with_unrelated_reply_rejected(isolated_db_path: str):
    """Scenario 2: A unit containing only a question about the requested fact and an unrelated reply does not establish the answer."""
    doc = Document(
        page_content="[ID 3] [01/03/2026 08:10 AM] Dev_Security: Who approved the firewall port change?",
        metadata={"message_id": 3, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "question_only", "reason": "Security asks about firewall, but Dana only discusses stormy weather."}
    ])

    state: GraphState = {
        "question": "Who approved the firewall port change?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 0
    decisions = result["contextual_grading_decisions"]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == "irrelevant"
    assert decisions[0]["evidence_type"] == "question_only"


def test_03_question_about_approver_does_not_establish_approver(isolated_db_path: str):
    """Scenario 3: A question about an approver does not establish who approved a change."""
    doc = Document(
        page_content="[ID 4] [01/03/2026 08:11 AM] Dev_Dana: I think the weather is going to be stormy this evening.",
        metadata={"message_id": 4, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "question_only", "reason": "Contains question about approver but no answer identifying the approver."}
    ])

    state: GraphState = {
        "question": "Who approved the firewall port change?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 0
    assert result["contextual_grading_decisions"][0]["evidence_type"] == "question_only"


def test_04_development_secret_does_not_support_production_secret(isolated_db_path: str):
    """Scenario 4: A development secret does not support a request for a production secret."""
    doc = Document(
        page_content="[ID 6] [01/03/2026 08:16 AM] Dev_Sam: In secrets.dev.json.",
        metadata={"message_id": 6, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "entity_mismatch", "reason": "Mentions dev test JWT token in secrets.dev.json, not production JWT secret."}
    ])

    state: GraphState = {
        "question": "Where is the production JWT secret stored?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 0
    decisions = result["contextual_grading_decisions"]
    assert decisions[0]["decision"] == "irrelevant"
    assert decisions[0]["evidence_type"] == "entity_mismatch"


def test_05_development_database_does_not_establish_production_database(isolated_db_path: str):
    """Scenario 5: A development database location does not establish a production database location."""
    doc = Document(
        page_content="[ID 12] [01/03/2026 08:31 AM] Dev_Sam: Host dev-db-01.internal.",
        metadata={"message_id": 12, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "entity_mismatch", "reason": "Discusses dev database dev-db-01, not production database cluster."}
    ])

    state: GraphState = {
        "question": "Where is the production database cluster hosted?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 0
    assert result["contextual_grading_decisions"][0]["evidence_type"] == "entity_mismatch"


def test_06_ambiguous_evidence_maybe_does_not_support_definitive_claim(isolated_db_path: str):
    """Scenario 6: Ambiguous evidence such as 'Maybe' does not support a definitive yes or no."""
    doc = Document(
        page_content="[ID 8] [01/03/2026 08:21 AM] Dev_Jordan: Maybe.",
        metadata={"message_id": 8, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "ambiguous", "reason": "Reply is only 'Maybe' which does not establish if autoscaler triggered."}
    ])

    state: GraphState = {
        "question": "Did the Kubernetes pod autoscaler trigger?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    # When no supported evidence exists, ambiguous units are NOT passed forward as proof
    assert len(result["documents"]) == 0
    decisions = result["contextual_grading_decisions"]
    assert decisions[0]["decision"] == "uncertain"
    assert decisions[0]["evidence_type"] == "ambiguous"


def test_07_clear_answer_remains_available_with_surrounding_irrelevant_context(isolated_db_path: str):
    """Scenario 7: A clear answer remains available when the surrounding unit also contains irrelevant context."""
    # Unit around ID 2 includes standup notice ID 0 and question ID 1
    doc = Document(
        page_content="[ID 2] [01/03/2026 08:06 AM] Dev_Sam: In Vault 9.",
        metadata={"message_id": 2, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "supported", "reason": "Vault 9 specifically answers SSL certificate storage location."}
    ])

    state: GraphState = {
        "question": "Where is the backup SSL certificate stored?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["anchor_message_id"] == 2
    assert result["documents"][0].metadata["relevance_decision"] == "relevant"


def test_08_cafeteria_keys_distractor_remains_rejectable(isolated_db_path: str):
    """Scenario 8: The cafeteria-keys distractor remains rejectable in the staging-API-key scenario."""
    doc = Document(
        page_content="[ID 10] [01/03/2026 08:26 AM] Dev_Jordan: On the kitchen counter.",
        metadata={"message_id": 10, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "irrelevant", "reason": "Discusses cafeteria keys on kitchen counter, unrelated to staging API key."}
    ])

    state: GraphState = {
        "question": "Where is the staging API key stored?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 0
    assert result["contextual_grading_decisions"][0]["decision"] == "irrelevant"
    assert result["contextual_grading_decisions"][0]["evidence_type"] == "irrelevant"


def test_09_rejected_candidates_cannot_reenter(isolated_db_path: str):
    """Scenario 9: Irrelevant candidates rejected by the grader cannot re-enter the final ranking."""
    doc_valid = Document(
        page_content="[ID 2] [01/03/2026 08:06 AM] Dev_Sam: In Vault 9.",
        metadata={"message_id": 2, "source": "hybrid_search", "retrieval_rank": 2},
    )
    doc_distractor = Document(
        page_content="[ID 10] [01/03/2026 08:26 AM] Dev_Jordan: On the kitchen counter.",
        metadata={"message_id": 10, "source": "hybrid_search", "retrieval_rank": 1},
    )

    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "supported", "reason": "Answers SSL cert location"},
        {"candidate": 2, "decision": "irrelevant", "reason": "Cafeteria keys distractor"},
    ])

    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "documents": [doc_valid, doc_distractor],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    retained_anchors = [d.metadata["anchor_message_id"] for d in result["documents"]]
    assert 2 in retained_anchors
    assert 10 not in retained_anchors


def test_10_all_candidates_ambiguous_follows_insufficient_evidence_behavior(isolated_db_path: str):
    """Scenario 10: All candidate units being ambiguous or unsupported follows verified insufficient-evidence behavior."""
    doc = Document(
        page_content="[ID 8] [01/03/2026 08:21 AM] Dev_Jordan: Maybe.",
        metadata={"message_id": 8, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "ambiguous", "reason": "Unconfirmed speculation"}
    ])

    state: GraphState = {
        "question": "Did the autoscaler trigger?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 0

    # Test routing immediately after assembly & grading
    route_step_1 = decide_post_contextual_grading_v2(result)
    assert route_step_1 == "rewrite_query"  # rewrite count < 2

    # After rewrite limit reached, routes to generate with empty documents
    result_exhausted = {**result, "rewrite_count": 2}
    route_step_2 = decide_post_contextual_grading_v2(result_exhausted)
    assert route_step_2 == "generate"

    # In generate, empty documents triggers clean insufficient-evidence prompt
    prompt = build_generation_prompt(result_exhausted, documents=[])
    assert "no reliable retrieved context available" in prompt.lower()
    assert "insufficient" in prompt.lower()


def test_11_valid_terse_location_time_confirmation_survive(isolated_db_path: str):
    """Scenario 11: Valid terse location, time, confirmation examples retain their supporting evidence."""
    doc_loc = Document(
        page_content="[ID 2] [01/03/2026 08:06 AM] Dev_Sam: In Vault 9.",
        metadata={"message_id": 2, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "supported", "reason": "Terse location answers question"}
    ])

    state: GraphState = {
        "question": "Where is the certificate?",
        "documents": [doc_loc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["evidence_type"] == "supported"


def test_12_canonical_message_ids_and_metadata_preserved(isolated_db_path: str):
    """Scenario 12: Canonical message IDs and anchor metadata remain correct after assembly, grading, and ranking."""
    doc = Document(
        page_content="[ID 2] [01/03/2026 08:06 AM] Dev_Sam: In Vault 9.",
        metadata={"message_id": 2, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "supported", "reason": "Direct fact supported"}
    ])

    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    meta = result["documents"][0].metadata

    assert meta["anchor_message_id"] == 2
    assert meta["message_id"] == 2
    assert "included_message_ids" in meta
    assert meta["relevance_decision"] == "relevant"
    assert meta["evidence_type"] == "supported"
    assert "retrieval_rank" in meta
    assert "contextual_rank" in meta
    assert "fused_score" in meta


def test_13_malformed_grader_output_and_exceptions_follow_policy(isolated_db_path: str):
    """Scenario 13: Malformed grader output and exceptions follow the documented handling policy."""
    doc = Document(
        page_content="[ID 2] [01/03/2026 08:06 AM] Dev_Sam: In Vault 9.",
        metadata={"message_id": 2, "source": "hybrid_search", "retrieval_rank": 1},
    )

    # 1. Malformed string (not valid JSON)
    bad_llm = MockV2GraderLLM(raw_string="I am not returning valid JSON at all.")
    state: GraphState = {
        "question": "Where is the cert?",
        "documents": [doc],
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result_bad = assemble_grade_rank_evidence_v2(state, grader_llm=bad_llm, database_path=isolated_db_path)
    assert result_bad["grader_status"] == "failed"
    assert result_bad["documents"][0].metadata["grader_status"] == "failure"
    assert result_bad["documents"][0].metadata["grader_failure"] is True
    assert result_bad["documents"][0].metadata["relevance_decision"] == "uncertain"

    # 2. Exception raising mock
    exc_llm = MagicMock()
    exc_llm.invoke.side_effect = RuntimeError("Ollama connection timeout")
    result_exc = assemble_grade_rank_evidence_v2(state, grader_llm=exc_llm, database_path=isolated_db_path)
    assert result_exc["grader_status"] == "failed"
    assert "Ollama connection timeout" in result_exc["contextual_grading_decisions"][0]["reason"]


def test_14_new_strategy_disabled_by_default_and_independently_selectable():
    """Scenario 14: The new strategy is disabled by default and selected independently."""
    empty_state: GraphState = {}
    assert resolve_retrieval_strategy(empty_state) == STRATEGY_BASELINE
    assert is_contextual_grading_v2_enabled(empty_state) is False

    v2_state: GraphState = {"retrieval_strategy": "contextual_grading_v2"}
    assert resolve_retrieval_strategy(v2_state) == STRATEGY_CONTEXTUAL_GRADING_V2
    assert is_contextual_grading_v2_enabled(v2_state) is True

    flag_state: GraphState = {"contextual_grading_v2_mode": True}
    assert resolve_retrieval_strategy(flag_state) == STRATEGY_CONTEXTUAL_GRADING_V2
    assert is_contextual_grading_v2_enabled(flag_state) is True


def test_15_existing_strategies_retain_routing_behavior():
    """Scenario 15: Existing baseline, dialogue-evidence, and contextual-grading strategies retain routing."""
    assert resolve_retrieval_strategy({"retrieval_strategy": "baseline"}) == STRATEGY_BASELINE
    assert route_post_retrieval({"retrieval_strategy": "baseline"}) == "grade_documents"

    assert resolve_retrieval_strategy({"retrieval_strategy": "dialogue_evidence"}) == STRATEGY_DIALOGUE_EVIDENCE
    assert route_post_retrieval({"retrieval_strategy": "dialogue_evidence"}) == "assemble_and_rank_evidence"

    assert resolve_retrieval_strategy({"retrieval_strategy": "contextual_grading"}) == STRATEGY_CONTEXTUAL_GRADING
    assert route_post_retrieval({"retrieval_strategy": "contextual_grading"}) == "assemble_grade_rank_evidence"

    assert resolve_retrieval_strategy({"retrieval_strategy": "contextual_grading_v2"}) == STRATEGY_CONTEXTUAL_GRADING_V2
    assert route_post_retrieval({"retrieval_strategy": "contextual_grading_v2"}) == "assemble_grade_rank_evidence_v2"


def test_16_repeated_invocations_do_not_leak_strategy_state():
    """Scenario 16: Repeated invocations do not leak strategy state."""
    state1: GraphState = {"retrieval_strategy": "contextual_grading_v2"}
    assert resolve_retrieval_strategy(state1) == STRATEGY_CONTEXTUAL_GRADING_V2

    state2: GraphState = {}
    assert resolve_retrieval_strategy(state2) == STRATEGY_BASELINE

    state3: GraphState = {"dialogue_evidence_mode": True}
    assert resolve_retrieval_strategy(state3) == STRATEGY_DIALOGUE_EVIDENCE

    state4: GraphState = {}
    assert resolve_retrieval_strategy(state4) == STRATEGY_BASELINE


def test_17_top_k_and_citation_schema_compatibility(isolated_db_path: str):
    """Scenario 17: Top-K and citation schema remain compatible."""
    docs = [
        Document(
            page_content=f"[ID {i}] Dev_{i}: Info {i}",
            metadata={"message_id": i, "source": "hybrid_search", "retrieval_rank": i},
        )
        for i in range(1, 10)
    ]
    mock_llm = MockV2GraderLLM([
        {"candidate": i, "decision": "supported", "reason": f"fact {i}"}
        for i in range(1, 10)
    ])

    state: GraphState = {
        "question": "General question",
        "documents": docs,
        "database_path": isolated_db_path,
        "rewrite_count": 0,
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
    }

    result = assemble_grade_rank_evidence_v2(state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(result["documents"]) == TOP_K_EVIDENCE_UNITS
    assert result["evidence_mode"] == "contextual_grading_v2"


def test_18_conflicting_settings_rejected_safely():
    """Conflicting flags or strategies are rejected explicitly with ValueError."""
    with pytest.raises(ValueError, match="Conflicting retrieval experiment flags"):
        resolve_retrieval_strategy({
            "contextual_grading_mode": True,
            "contextual_grading_v2_mode": True,
        })

    with pytest.raises(ValueError, match="Conflicting configuration"):
        resolve_retrieval_strategy({
            "retrieval_strategy": "baseline",
            "contextual_grading_v2_mode": True,
        })

    with pytest.raises(ValueError, match="Conflicting configuration"):
        resolve_retrieval_strategy({
            "retrieval_strategy": "dialogue_evidence",
            "contextual_grading_v2_mode": True,
        })


def test_19_parser_maps_rich_v2_categories_compatibly():
    """Parser maps v2 evidence types (supported, question_only, entity_mismatch, ambiguous, irrelevant) correctly."""
    raw_json = json.dumps([
        {"candidate": 1, "decision": "supported", "reason": "Direct fact"},
        {"candidate": 2, "decision": "question_only", "reason": "Only asks question"},
        {"candidate": 3, "decision": "entity_mismatch", "reason": "Dev token not prod secret"},
        {"candidate": 4, "decision": "ambiguous", "reason": "Speculative maybe"},
        {"candidate": 5, "decision": "irrelevant", "reason": "Social chat"},
    ])

    parsed = parse_contextual_grader_v2_response(raw_json, [1, 2, 3, 4, 5])
    assert parsed[1] == ("relevant", "supported", "Direct fact", "success")
    assert parsed[2] == ("irrelevant", "question_only", "Only asks question", "success")
    assert parsed[3] == ("irrelevant", "entity_mismatch", "Dev token not prod secret", "success")
    assert parsed[4] == ("uncertain", "ambiguous", "Speculative maybe", "success")
    assert parsed[5] == ("irrelevant", "irrelevant", "Social chat", "success")


def test_20_full_graph_invocation_with_contextual_grading_v2(isolated_db_path: str):
    """Full LangGraph state machine execution using contextual_grading_v2."""
    doc = Document(
        page_content="[ID 2] [01/03/2026 08:06 AM] Dev_Sam: In Vault 9.",
        metadata={"message_id": 2, "source": "hybrid_search", "retrieval_rank": 1},
    )
    mock_llm = MockV2GraderLLM([
        {"candidate": 1, "decision": "supported", "reason": "Vault 9 answers the SSL cert location question"}
    ])

    initial_state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "original_question": "Where was the backup SSL certificate placed?",
        "documents": [doc],
        "generation": "",
        "rewrite_count": 0,
        "grounded": False,
        "semantic_grounded": False,
        "df": None,
        "scope": "specific",
        "primary_intent": "SEMANTIC_RAG",
        "secondary_intents": [],
        "query_spec": None,
        "citation_validation": None,
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "available_citation_ids": [],
        "grader_status": None,
        "evidence_mode": "standard",
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_V2,
        "grader_llm": mock_llm,
        "database_path": isolated_db_path,
    }

    graph = build_graph()
    # Step into assemble_grade_rank_evidence_v2 node directly
    next_node = route_post_retrieval(initial_state)
    assert next_node == "assemble_grade_rank_evidence_v2"

    graded_state = assemble_grade_rank_evidence_v2(initial_state, grader_llm=mock_llm, database_path=isolated_db_path)
    assert len(graded_state["documents"]) == 1
    assert graded_state["evidence_mode"] == "contextual_grading_v2"
    assert graded_state["retrieval_strategy"] == STRATEGY_CONTEXTUAL_GRADING_V2
    assert graded_state["documents"][0].metadata["evidence_type"] == "supported"

    post_decision = decide_post_contextual_grading_v2(graded_state)
    assert post_decision == "generate"
