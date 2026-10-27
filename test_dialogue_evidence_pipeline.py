"""
test_dialogue_evidence_pipeline.py
==================================
Deterministic offline test suite for the opt-in dialogue-evidence retrieval pipeline
introduced in Task 37.

Verifies:
  1. The feature flag is disabled by default.
  2. The default production graph preserves its existing route and grading behavior.
  3. Enabling the flag invokes assembly before ranking and generation.
  4. The new route does not call grade_documents as a hard relevance gate.
  5. A terse anchor such as 'In Vault 9.' retains its anchor ID and surrounding turns through the new path.
  6. A short time reply and a confirmation reply survive evidence selection when included in the retrieved candidates.
  7. A contextual similarity signal can change evidence ordering deterministically without losing the original retrieval metadata.
  8. RRF ranking and tie-breaking are deterministic.
  9. Duplicate copies of the same anchor are handled correctly.
  10. Distinct anchors with overlapping dialogue windows remain distinct.
  11. A pre-existing conversation chunk is not expanded twice.
  12. Missing context or malformed metadata follows the Task 36 safe-fallback policy.
  13. Embedding/ranking failure falls back safely and does not silently discard all candidates.
  14. An empty result preserves the existing retry or insufficient-evidence path.
  15. Citation IDs and anchor identities remain available through generation and verification.
  16. A nearby but unrelated message cannot silently overwrite the anchor's identity.
  17. The experimental top-K limit is respected.
  18. No model inference is needed to run the tests.

SAFETY: Strictly offline; zero Ollama, network, or production database access.
"""

import copy
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import rag_graph
from rag_graph import (
    DIALOGUE_EVIDENCE_ENV_VAR,
    TOP_K_EVIDENCE_UNITS,
    RRF_K_CONTEXTUAL,
    WEIGHT_RETRIEVAL,
    WEIGHT_CONTEXTUAL,
    GraphState,
    assemble_and_rank_evidence,
    build_graph,
    compute_cosine_similarity,
    decide_post_assembly,
    extract_available_message_ids,
    is_dialogue_evidence_enabled,
    rank_dialogue_evidence_units,
    route_post_retrieval,
)


# ---------------------------------------------------------------------
# Isolated SQLite Test Fixture
# ---------------------------------------------------------------------

@pytest.fixture
def isolated_pipeline_db(tmp_path: Path) -> str:
    """
    Creates an isolated temporary SQLite database populated with a synthetic
    chat archive containing terse replies, questions, and distractor messages:
      ID 10: [09:08] Dev_Security: Checking port access.
      ID 11: [09:10] Dev_Alex: Where did we store the backup SSL certificate?
      ID 12: [09:11] Dev_Sam: In Vault 9.
      ID 13: [09:12] Dev_Alex: Got it, deploying certs now.
      ID 14: [09:14] Dev_Ops: When is the production deployment window scheduled?
      ID 15: [09:15] Dev_Lead: 9pm.
      ID 16: [09:18] Dev_Dev: Did the database migration script succeed on staging?
      ID 17: [09:19] Dev_Sarah: Yes, completely clean.
      ID 18: [09:25] Dev_Carlos: Cafeteria lunch menu today is vegetable lasagna.
      ID 19: [09:30] Dev_Dana: Anyone ordering pizza today?
    """
    db_path = str(tmp_path / "pipeline_test_chat.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            sender TEXT,
            message TEXT,
            message_type TEXT,
            sentiment_compound REAL,
            sentiment_label TEXT,
            is_question INTEGER,
            message_length INTEGER,
            response_time_minutes REAL,
            reply_time_minutes REAL
        )
        """
    )
    rows = [
        (10, "2026-10-14T09:08:00", "Dev_Security", "Checking port access.", "text", 0.0, "neutral", 0, 21, None, None),
        (11, "2026-10-14T09:10:00", "Dev_Alex", "Where did we store the backup SSL certificate?", "text", 0.0, "neutral", 1, 46, None, None),
        (12, "2026-10-14T09:11:00", "Dev_Sam", "In Vault 9.", "text", 0.0, "neutral", 0, 11, None, None),
        (13, "2026-10-14T09:12:00", "Dev_Alex", "Got it, deploying certs now.", "text", 0.0, "neutral", 0, 28, None, None),
        (14, "2026-10-14T09:14:00", "Dev_Ops", "When is the production deployment window scheduled?", "text", 0.0, "neutral", 1, 51, None, None),
        (15, "2026-10-14T09:15:00", "Dev_Lead", "9pm.", "text", 0.0, "neutral", 0, 4, None, None),
        (16, "2026-10-14T09:18:00", "Dev_Dev", "Did the database migration script succeed on staging?", "text", 0.0, "neutral", 1, 53, None, None),
        (17, "2026-10-14T09:19:00", "Dev_Sarah", "Yes, completely clean.", "text", 0.0, "neutral", 0, 21, None, None),
        (18, "2026-10-14T09:25:00", "Dev_Carlos", "Cafeteria lunch menu today is vegetable lasagna.", "text", 0.0, "neutral", 0, 48, None, None),
        (19, "2026-10-14T09:30:00", "Dev_Dana", "Anyone ordering pizza today?", "text", 0.0, "neutral", 1, 28, None, None),
    ]
    cursor.executemany(
        """
        INSERT INTO messages (
            id, timestamp, sender, message, message_type,
            sentiment_compound, sentiment_label, is_question,
            message_length, response_time_minutes, reply_time_minutes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------
# Deterministic Mock Embeddings Class
# ---------------------------------------------------------------------

class DeterministicMockEmbeddings:
    """Mock embeddings providing deterministic vector responses without external model calls."""
    def __init__(self, mapping: Optional[Dict[str, List[float]]] = None):
        self.mapping = mapping or {}
        self.query_invocations = 0
        self.doc_invocations = 0

    def embed_query(self, text: str) -> List[float]:
        self.query_invocations += 1
        for key, vec in self.mapping.items():
            if key.lower() in text.lower():
                return vec
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.doc_invocations += len(texts)
        res = []
        for t in texts:
            matched = False
            for key, vec in self.mapping.items():
                if key.lower() in t.lower():
                    res.append(vec)
                    matched = True
                    break
            if not matched:
                res.append([0.0, 1.0, 0.0])
        return res


# ---------------------------------------------------------------------
# Test 1 & 2: Feature flag disabled by default & production path preserved
# ---------------------------------------------------------------------

def test_feature_flag_disabled_by_default(monkeypatch):
    """The dialogue evidence feature flag must be disabled by default."""
    monkeypatch.delenv(DIALOGUE_EVIDENCE_ENV_VAR, raising=False)
    assert is_dialogue_evidence_enabled() is False

    state: GraphState = {"question": "test"}
    assert is_dialogue_evidence_enabled(state) is False

    # Route after retrieve must route to grade_documents in default mode
    assert route_post_retrieval(state) == "grade_documents"


def test_default_production_graph_preserves_routing(monkeypatch):
    """When the flag is disabled, LangGraph routing must execute the production sequence."""
    monkeypatch.delenv(DIALOGUE_EVIDENCE_ENV_VAR, raising=False)
    state: GraphState = {
        "question": "Where is the key?",
        "documents": [Document(page_content="[ID 1] key")],
        "rewrite_count": 0,
    }
    # Verifies route_post_retrieval returns grade_documents
    next_node = route_post_retrieval(state)
    assert next_node == "grade_documents"


# ---------------------------------------------------------------------
# Test 3 & 4: Enabling flag invokes assembly & ranking and bypasses grade_documents
# ---------------------------------------------------------------------

def test_enabling_flag_invokes_assembly_and_bypasses_grader(monkeypatch):
    """When enabled via env var or state, route_post_retrieval routes to assemble_and_rank_evidence."""
    monkeypatch.setenv(DIALOGUE_EVIDENCE_ENV_VAR, "1")
    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "documents": [Document(page_content="[ID 12] In Vault 9.")],
        "rewrite_count": 0,
    }
    next_node = route_post_retrieval(state)
    assert next_node == "assemble_and_rank_evidence"

    # State-level override
    state_override: GraphState = {
        "question": "test",
        "dialogue_evidence_mode": True,
    }
    monkeypatch.setenv(DIALOGUE_EVIDENCE_ENV_VAR, "0")
    assert route_post_retrieval(state_override) == "assemble_and_rank_evidence"

    # Post assembly routing: if documents exist, routes directly to generate
    post_assembly_state: GraphState = {
        "documents": [Document(page_content="assembled evidence")],
        "rewrite_count": 0,
    }
    assert decide_post_assembly(post_assembly_state) == "generate"


# ---------------------------------------------------------------------
# Test 5: Terse anchor 'In Vault 9.' retains turns and anchor ID
# ---------------------------------------------------------------------

def test_terse_anchor_retains_turns_and_anchor_id(isolated_pipeline_db: str):
    """Anchor 12 ('In Vault 9.') must retain its anchor identity and acquire turns [10, 11, 12, 13]."""
    doc12 = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12, "hybrid_score": 0.85},
    )
    state: GraphState = {
        "question": "Where was the backup SSL certificate placed?",
        "documents": [doc12],
        "database_path": isolated_pipeline_db,
    }

    result_state = assemble_and_rank_evidence(state)
    surviving_docs = result_state["documents"]

    assert len(surviving_docs) == 1
    unit = surviving_docs[0]

    assert unit.metadata["anchor_message_id"] == 12
    assert unit.metadata["included_message_ids"] == [10, 11, 12, 13]
    assert "[RETRIEVED ANCHOR]" in unit.page_content
    assert "[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9." in unit.page_content
    assert "[ID 11] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?" in unit.page_content
    assert result_state["evidence_mode"] == "dialogue_evidence_unit"


# ---------------------------------------------------------------------
# Test 6: Short time reply and confirmation reply survive evidence selection
# ---------------------------------------------------------------------

def test_short_time_and_confirmation_survive(isolated_pipeline_db: str):
    """Short replies (ID 15: '9pm.' and ID 17: 'Yes, completely clean.') survive through the new path."""
    doc_time = Document(
        page_content="[ID 15] [14/10/2026 09:15 AM] Dev_Lead: 9pm.",
        metadata={"source": "hybrid_search", "message_id": 15, "hybrid_score": 0.70},
    )
    doc_confirm = Document(
        page_content="[ID 17] [14/10/2026 09:19 AM] Dev_Sarah: Yes, completely clean.",
        metadata={"source": "hybrid_search", "message_id": 17, "hybrid_score": 0.75},
    )
    state: GraphState = {
        "question": "Did migration succeed and what time is deployment?",
        "documents": [doc_confirm, doc_time],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    docs = result["documents"]

    assert len(docs) == 2
    anchor_ids = [d.metadata["anchor_message_id"] for d in docs]
    assert 17 in anchor_ids
    assert 15 in anchor_ids

    # Confirmation unit acquired its question (ID 16)
    confirm_unit = next(d for d in docs if d.metadata["anchor_message_id"] == 17)
    assert 16 in confirm_unit.metadata["included_message_ids"]


# ---------------------------------------------------------------------
# Test 7: Contextual similarity signal changes evidence ordering deterministically
# ---------------------------------------------------------------------

def test_contextual_similarity_changes_evidence_ordering(isolated_pipeline_db: str):
    """
    Candidate A (ID 18, cafeteria) has retrieval rank 1.
    Candidate B (ID 12, 'In Vault 9.') has retrieval rank 2.
    When contextual transcript for B strongly matches query, RRF promotes Candidate B to rank 1.
    """
    doc_cafeteria = Document(
        page_content="[ID 18] [14/10/2026 09:25 AM] Dev_Carlos: Cafeteria lunch menu today is vegetable lasagna.",
        metadata={"source": "hybrid_search", "message_id": 18, "hybrid_score": 0.95},
    )
    doc_vault = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12, "hybrid_score": 0.60},
    )

    # Mock embeddings: query matches SSL certificate terms present in ID 12's context
    mock_embed = DeterministicMockEmbeddings({
        "ssl": [1.0, 0.0, 0.0],
        "backup ssl certificate": [1.0, 0.0, 0.0],
        "lasagna": [0.0, 1.0, 0.0],
    })

    state: GraphState = {
        "question": "Where is the backup SSL certificate?",
        "documents": [doc_cafeteria, doc_vault],  # retrieval order: cafeteria #1, vault #2
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state, embedding_function=mock_embed)
    docs = result["documents"]

    assert len(docs) == 2
    # Vault 9 must be promoted to rank 1 due to high contextual transcript similarity
    assert docs[0].metadata["anchor_message_id"] == 12
    assert docs[0].metadata["retrieval_rank"] == 2
    assert docs[0].metadata["contextual_rank"] == 1

    assert docs[1].metadata["anchor_message_id"] == 18
    assert docs[1].metadata["retrieval_rank"] == 1
    assert docs[1].metadata["contextual_rank"] == 2


# ---------------------------------------------------------------------
# Test 8: RRF ranking formula and tie-breaking are deterministic
# ---------------------------------------------------------------------

def test_rrf_ranking_and_tie_breaking_deterministic():
    """RRF fusion formula and tie-breaking must yield identical ordering across repeated runs."""
    doc1 = Document(
        page_content="Transcript A",
        metadata={"anchor_message_id": 101},
    )
    doc2 = Document(
        page_content="Transcript B",
        metadata={"anchor_message_id": 102},
    )

    # Identical similarity to test deterministic tie-breaking
    mock_embed = DeterministicMockEmbeddings({
        "query": [1.0, 0.0],
        "Transcript A": [0.5, 0.5],
        "Transcript B": [0.5, 0.5],
    })

    run1 = rank_dialogue_evidence_units("query", [doc1, doc2], embedding_function=mock_embed)
    run2 = rank_dialogue_evidence_units("query", [doc1, doc2], embedding_function=mock_embed)

    order1 = [d.metadata["anchor_message_id"] for d in run1]
    order2 = [d.metadata["anchor_message_id"] for d in run2]
    assert order1 == order2

    # Verify RRF formula calculation
    fused_expected = round((WEIGHT_RETRIEVAL / (RRF_K_CONTEXTUAL + 1)) + (WEIGHT_CONTEXTUAL / (RRF_K_CONTEXTUAL + 1)), 6)
    assert run1[0].metadata["fused_score"] == fused_expected


# ---------------------------------------------------------------------
# Test 9: Duplicate copies of the same anchor handled correctly
# ---------------------------------------------------------------------

def test_duplicate_input_anchors_handled_correctly(isolated_pipeline_db: str):
    """If the retriever emits the same anchor twice, exactly one evidence unit must be produced."""
    doc12_a = Document(
        page_content="[ID 12] In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )
    doc12_b = Document(
        page_content="[ID 12] In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )
    state: GraphState = {
        "question": "Where is the cert?",
        "documents": [doc12_a, doc12_b],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    docs = result["documents"]
    assert len(docs) == 1
    assert docs[0].metadata["anchor_message_id"] == 12


# ---------------------------------------------------------------------
# Test 10: Distinct anchors with overlapping dialogue windows remain distinct
# ---------------------------------------------------------------------

def test_distinct_anchors_with_overlapping_windows(isolated_pipeline_db: str):
    """Anchors 11 and 12 produce overlapping windows, but remain separate units."""
    doc11 = Document(
        page_content="[ID 11] Where did we store the backup SSL certificate?",
        metadata={"source": "hybrid_search", "message_id": 11},
    )
    doc12 = Document(
        page_content="[ID 12] In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )
    state: GraphState = {
        "question": "Where is the cert?",
        "documents": [doc11, doc12],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    docs = result["documents"]
    assert len(docs) == 2
    assert docs[0].metadata["anchor_message_id"] != docs[1].metadata["anchor_message_id"]


# ---------------------------------------------------------------------
# Test 11: Pre-existing conversation chunks not expanded twice
# ---------------------------------------------------------------------

def test_preexisting_chunk_not_expanded_twice(isolated_pipeline_db: str):
    """Conversation chunks already represent transcript windows and must be preserved as-is."""
    chunk = Document(
        page_content="[ID 10] Checking port access.\n[ID 11] Where did we store the backup SSL cert?",
        metadata={
            "source": "conversation_chunk",
            "chunk_id": 99,
            "start_message_id": 10,
            "end_message_id": 11,
        },
    )
    state: GraphState = {
        "question": "Where is the cert?",
        "documents": [chunk],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    docs = result["documents"]
    assert len(docs) == 1
    assert docs[0].metadata["evidence_unit_status"] == "preserved_chunk"
    assert docs[0].metadata["chunk_id"] == 99


# ---------------------------------------------------------------------
# Test 12: Missing context or malformed metadata safe fallback
# ---------------------------------------------------------------------

def test_missing_context_or_malformed_metadata_safe_fallback(isolated_pipeline_db: str):
    """Ghost IDs or malformed metadata fall back safely without raising exceptions or discarding."""
    ghost = Document(
        page_content="Ghost message without DB row",
        metadata={"source": "hybrid_search", "message_id": 99999},
    )
    malformed = Document(
        page_content="Unparseable message",
        metadata={"source": "hybrid_search", "message_id": "bad_id"},
    )
    state: GraphState = {
        "question": "test",
        "documents": [ghost, malformed],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    docs = result["documents"]
    assert len(docs) == 2
    assert docs[0].metadata["context_expanded"] is False
    assert docs[1].metadata["context_expanded"] is False


# ---------------------------------------------------------------------
# Test 13: Embedding/ranking failure falls back safely
# ---------------------------------------------------------------------

def test_embedding_ranking_failure_falls_back_safely(isolated_pipeline_db: str):
    """If embedding computation fails with an exception, candidates fall back safely to retrieval order."""
    doc12 = Document(
        page_content="[ID 12] In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )
    doc15 = Document(
        page_content="[ID 15] 9pm.",
        metadata={"source": "hybrid_search", "message_id": 15},
    )

    failing_embed = MagicMock()
    failing_embed.embed_query.side_effect = RuntimeError("Embedding service connection error")

    state: GraphState = {
        "question": "test",
        "documents": [doc12, doc15],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state, embedding_function=failing_embed)
    docs = result["documents"]
    assert len(docs) == 2
    assert docs[0].metadata["contextual_ranking_fallback"] is True
    assert docs[0].metadata["contextual_ranking_status"] == "fallback_retrieval_order"
    assert docs[0].metadata["anchor_message_id"] == 12
    assert docs[1].metadata["anchor_message_id"] == 15


# ---------------------------------------------------------------------
# Test 14: Empty result preserves retry or insufficient-evidence path
# ---------------------------------------------------------------------

def test_empty_result_preserves_retry_or_insufficient_evidence():
    """Empty retrieval routes to rewrite_query if rewrites remain, or generate if exhausted."""
    state_can_rewrite: GraphState = {
        "documents": [],
        "rewrite_count": 0,
    }
    assert decide_post_assembly(state_can_rewrite) == "rewrite_query"

    state_exhausted: GraphState = {
        "documents": [],
        "rewrite_count": 2,  # MAX_REWRITES
    }
    assert decide_post_assembly(state_exhausted) == "generate"


# ---------------------------------------------------------------------
# Test 15: Citation IDs and anchor identities available for generation
# ---------------------------------------------------------------------

def test_citation_ids_and_anchor_identities_available(isolated_pipeline_db: str):
    """
    All IDs in assembled dialogue units must be discoverable by extract_available_message_ids()
    and the anchor identity must remain distinguishable in metadata.
    """
    doc12 = Document(
        page_content="[ID 12] In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )
    state: GraphState = {
        "question": "Where is the cert?",
        "documents": [doc12],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    docs = result["documents"]
    assert len(docs) == 1

    available_ids = extract_available_message_ids(docs)
    # Context contains turns [10, 11, 12, 13]
    assert 10 in available_ids
    assert 11 in available_ids
    assert 12 in available_ids
    assert 13 in available_ids

    # Anchor ID is distinct in metadata
    assert docs[0].metadata["anchor_message_id"] == 12


# ---------------------------------------------------------------------
# Test 16: Nearby message cannot overwrite anchor identity
# ---------------------------------------------------------------------

def test_nearby_message_cannot_overwrite_anchor_identity(isolated_pipeline_db: str):
    """Adjacent turns from Dev_Security or Dev_Alex must not overwrite the anchor's sender (Dev_Sam)."""
    doc12 = Document(
        page_content="[ID 12] In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12, "sender": "Dev_Sam"},
    )
    state: GraphState = {
        "question": "test",
        "documents": [doc12],
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    unit = result["documents"][0]

    assert unit.metadata["anchor_sender"] == "Dev_Sam"
    assert unit.metadata["anchor_message_id"] == 12


# ---------------------------------------------------------------------
# Test 17: Experimental top-K limit is respected
# ---------------------------------------------------------------------

def test_experimental_top_k_limit_respected(isolated_pipeline_db: str):
    """When more than TOP_K_EVIDENCE_UNITS candidates are provided, exactly TOP_K_EVIDENCE_UNITS are kept."""
    # Create 8 candidate documents (IDs 10 through 17)
    candidates = [
        Document(page_content=f"[ID {mid}] test message", metadata={"source": "hybrid_search", "message_id": mid})
        for mid in range(10, 18)
    ]
    state: GraphState = {
        "question": "test",
        "documents": candidates,
        "database_path": isolated_pipeline_db,
    }

    result = assemble_and_rank_evidence(state)
    docs = result["documents"]
    assert len(docs) == TOP_K_EVIDENCE_UNITS
    assert len(docs) == 5


# ---------------------------------------------------------------------
# Test 18: No model inference is needed to run tests
# ---------------------------------------------------------------------

def test_no_model_inference_needed():
    """Cosine similarity and RRF fusion run deterministically with pure Python math."""
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert compute_cosine_similarity(v1, v2) == 1.0

    v3 = [0.0, 1.0]
    assert compute_cosine_similarity(v1, v3) == 0.0
