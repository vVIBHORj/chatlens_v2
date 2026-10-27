"""
test_context_expander.py
========================
Deterministic offline test suite for assemble_dialogue_evidence_units in context_expander.py.

Verifies:
  1. Normal anchor with two preceding and one following message.
  2. Anchor at the start of a conversation (boundary condition).
  3. Anchor at the end of a conversation (boundary condition).
  4. Fewer neighboring messages than requested.
  5. Chronological ordering and deterministic timestamp ties (timestamp ASC, id ASC).
  6. Exact message-ID, timestamp, speaker, text, and anchor-metadata preservation.
  7. Multiple speakers and multiline message content.
  8. Overlapping windows from different anchors preserved as distinct evidence units.
  9. Duplicate input anchors deduplicated safely.
  10. Missing anchor rows and malformed metadata failing safely to standalone units.
  11. Context lookup failure and safe fallback behavior.
  12. Existing conversation chunks remaining unexpanded with source identity preserved.
  13. Empty input and omitted database path failing safely.
  14. Input documents remaining completely unchanged after assembly (immutability).
  15. No accidental production-database fallback when database_path is omitted.

SAFETY: Strictly offline; zero Ollama, network, or production database access.
"""

import copy
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from context_expander import assemble_dialogue_evidence_units


# ---------------------------------------------------------------------
# SQLite Isolated Test Fixture
# ---------------------------------------------------------------------

@pytest.fixture
def isolated_chat_db(tmp_path: Path) -> str:
    """
    Creates an isolated temporary SQLite database populated with a synthetic chat transcript:
      ID 10: [2026-10-14 09:08:00] Dev_Security: Checking port access.
      ID 11: [2026-10-14 09:10:00] Dev_Alex: Where did we store the backup SSL certificate?
      ID 12: [2026-10-14 09:11:00] Dev_Sam: In Vault 9.
      ID 13: [2026-10-14 09:12:00] Dev_Alex: Got it, deploying certs now.
      ID 14: [2026-10-14 09:15:00] Dev_Lead: Let us know when deployment succeeds.
      ID 15: [2026-10-14 09:20:00] Dev_Ops: All services green.
    """
    db_path = str(tmp_path / "isolated_test_chat.db")
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
        (14, "2026-10-14T09:15:00", "Dev_Lead", "Let us know when deployment succeeds.", "text", 0.0, "neutral", 0, 38, None, None),
        (15, "2026-10-14T09:20:00", "Dev_Ops", "All services green.", "text", 0.0, "positive", 0, 19, None, None),
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
# Test 1: Normal anchor with two preceding and one following message
# ---------------------------------------------------------------------

def test_normal_anchor_context_window(isolated_chat_db: str):
    """Anchor 12 with before=2, after=1 should assemble messages [10, 11, 12, 13]."""
    anchor_doc = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={
            "source": "hybrid_search",
            "message_id": 12,
            "sender": "Dev_Sam",
            "timestamp": "2026-10-14T09:11:00",
            "hybrid_score": 0.88,
        },
    )

    results = assemble_dialogue_evidence_units(
        [anchor_doc],
        before=2,
        after=1,
        database_path=isolated_chat_db,
    )

    assert len(results) == 1
    unit = results[0]

    # Content structure assertions
    assert "[RETRIEVED ANCHOR]" in unit.page_content
    assert "[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9." in unit.page_content
    assert "[CHRONOLOGICAL CONTEXT]" in unit.page_content

    # Chronological turns presence
    assert "[ID 10] [14/10/2026 09:08 AM] Dev_Security: Checking port access." in unit.page_content
    assert "[ID 11] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?" in unit.page_content
    assert "[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9." in unit.page_content
    assert "[ID 13] [14/10/2026 09:12 AM] Dev_Alex: Got it, deploying certs now." in unit.page_content

    # Metadata assertions
    assert unit.metadata["context_expanded"] is True
    assert unit.metadata["context_method"] == "dialogue_unit"
    assert unit.metadata["evidence_unit_status"] == "expanded"
    assert unit.metadata["anchor_message_id"] == 12
    assert unit.metadata["anchor_sender"] == "Dev_Sam"
    assert unit.metadata["context_messages_before"] == 2
    assert unit.metadata["context_messages_after"] == 1
    assert unit.metadata["context_start_message_id"] == 10
    assert unit.metadata["context_end_message_id"] == 13
    assert unit.metadata["included_message_ids"] == [10, 11, 12, 13]
    # Original metadata preserved
    assert unit.metadata["hybrid_score"] == 0.88
    assert unit.metadata["source"] == "hybrid_search"


# ---------------------------------------------------------------------
# Test 2: Anchor at the start of a conversation (boundary condition)
# ---------------------------------------------------------------------

def test_anchor_at_start_of_conversation(isolated_chat_db: str):
    """Anchor 10 is the first message in the chat; preceding messages must be 0."""
    anchor_doc = Document(
        page_content="[ID 10] [14/10/2026 09:08 AM] Dev_Security: Checking port access.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )

    results = assemble_dialogue_evidence_units(
        [anchor_doc],
        before=2,
        after=2,
        database_path=isolated_chat_db,
    )

    assert len(results) == 1
    unit = results[0]

    assert unit.metadata["context_expanded"] is True
    assert unit.metadata["context_messages_before"] == 0
    assert unit.metadata["context_messages_after"] == 2
    assert unit.metadata["included_message_ids"] == [10, 11, 12]
    assert unit.metadata["context_start_message_id"] == 10
    assert unit.metadata["context_end_message_id"] == 12


# ---------------------------------------------------------------------
# Test 3: Anchor at the end of a conversation (boundary condition)
# ---------------------------------------------------------------------

def test_anchor_at_end_of_conversation(isolated_chat_db: str):
    """Anchor 15 is the last message in the chat; following messages must be 0."""
    anchor_doc = Document(
        page_content="[ID 15] [14/10/2026 09:20 AM] Dev_Ops: All services green.",
        metadata={"source": "hybrid_search", "message_id": 15},
    )

    results = assemble_dialogue_evidence_units(
        [anchor_doc],
        before=2,
        after=2,
        database_path=isolated_chat_db,
    )

    assert len(results) == 1
    unit = results[0]

    assert unit.metadata["context_expanded"] is True
    assert unit.metadata["context_messages_before"] == 2
    assert unit.metadata["context_messages_after"] == 0
    assert unit.metadata["included_message_ids"] == [13, 14, 15]
    assert unit.metadata["context_start_message_id"] == 13
    assert unit.metadata["context_end_message_id"] == 15


# ---------------------------------------------------------------------
# Test 4: Fewer neighboring messages than requested
# ---------------------------------------------------------------------

def test_fewer_neighboring_messages_than_requested(isolated_chat_db: str):
    """Requesting before=10 and after=10 around ID 11 returns all available neighbors without error."""
    anchor_doc = Document(
        page_content="[ID 11] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?",
        metadata={"source": "hybrid_search", "message_id": 11},
    )

    results = assemble_dialogue_evidence_units(
        [anchor_doc],
        before=10,
        after=10,
        database_path=isolated_chat_db,
    )

    assert len(results) == 1
    unit = results[0]

    assert unit.metadata["context_messages_before"] == 1  # only ID 10 exists before
    assert unit.metadata["context_messages_after"] == 4   # IDs 12, 13, 14, 15 exist after
    assert unit.metadata["included_message_ids"] == [10, 11, 12, 13, 14, 15]


# ---------------------------------------------------------------------
# Test 5: Chronological ordering and deterministic timestamp ties
# ---------------------------------------------------------------------

def test_deterministic_timestamp_ties(tmp_path: Path):
    """
    Messages with identical timestamps must be deterministically ordered
    using the repository tie-breaker rule: (timestamp ASC, id ASC).
    """
    db_path = str(tmp_path / "timestamp_tie_chat.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            sender TEXT,
            message TEXT
        )
        """
    )
    # Three messages sharing the exact same timestamp string
    tied_rows = [
        (20, "2026-10-14T10:00:00", "Dev_A", "Message A at 10am"),
        (21, "2026-10-14T10:00:00", "Dev_B", "Message B at 10am (target)"),
        (22, "2026-10-14T10:00:00", "Dev_C", "Message C at 10am"),
        (23, "2026-10-14T10:05:00", "Dev_D", "Later message at 10:05am"),
    ]
    cursor.executemany("INSERT INTO messages (id, timestamp, sender, message) VALUES (?, ?, ?, ?)", tied_rows)
    conn.commit()
    conn.close()

    anchor_doc = Document(
        page_content="[ID 21] [14/10/2026 10:00 AM] Dev_B: Message B at 10am (target)",
        metadata={"source": "hybrid_search", "message_id": 21},
    )

    results = assemble_dialogue_evidence_units(
        [anchor_doc],
        before=1,
        after=1,
        database_path=db_path,
    )

    assert len(results) == 1
    unit = results[0]

    # Preceding tied message must be ID 20 (id < 21 at same timestamp)
    # Following tied message must be ID 22 (id > 21 at same timestamp)
    assert unit.metadata["included_message_ids"] == [20, 21, 22]

    # Ordering in transcript must be strict chronological ascending
    lines = unit.page_content.splitlines()
    context_lines = [l for l in lines if l.startswith("[ID ")]
    assert "[ID 20]" in context_lines[1]
    assert "[ID 21]" in context_lines[2]
    assert "[ID 22]" in context_lines[3]


# ---------------------------------------------------------------------
# Test 6: Exact message-ID, timestamp, speaker, text, and metadata preservation
# ---------------------------------------------------------------------

def test_exact_metadata_and_provenance_preservation(isolated_chat_db: str):
    """Verifies that anchor identity, speaker, and arbitrary retriever metadata are preserved."""
    anchor_doc = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={
            "source": "hybrid_search",
            "message_id": 12,
            "sender": "Dev_Sam",
            "timestamp": "2026-10-14T09:11:00",
            "hybrid_score": 0.9421,
            "evidence_score": 0.8833,
            "semantic_rank": 1,
            "lexical_rank": 3,
            "matched_terms": ["vault", "ssl"],
            "sources": ["dense", "lexical"],
        },
    )

    results = assemble_dialogue_evidence_units(
        [anchor_doc],
        before=1,
        after=1,
        database_path=isolated_chat_db,
    )

    assert len(results) == 1
    unit = results[0]

    # Provenance fields
    assert unit.metadata["anchor_message_id"] == 12
    assert unit.metadata["anchor_sender"] == "Dev_Sam"
    assert unit.metadata["anchor_timestamp"] == "2026-10-14T09:11:00"

    # Preserved retriever metadata
    assert unit.metadata["hybrid_score"] == 0.9421
    assert unit.metadata["evidence_score"] == 0.8833
    assert unit.metadata["semantic_rank"] == 1
    assert unit.metadata["lexical_rank"] == 3
    assert unit.metadata["matched_terms"] == ["vault", "ssl"]
    assert unit.metadata["sources"] == ["dense", "lexical"]


# ---------------------------------------------------------------------
# Test 7: Multiple speakers and multiline message content
# ---------------------------------------------------------------------

def test_multiline_content_and_multiple_speakers(tmp_path: Path):
    """
    Messages containing newlines, markdown code blocks, URLs, and brackets
    must be preserved without loss or escaping errors.
    """
    db_path = str(tmp_path / "multiline_chat.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            sender TEXT,
            message TEXT
        )
        """
    )
    multiline_body = "Here is the config:\n```json\n{\"port\": 8080, \"enabled\": true}\n```\nCheck: https://intranet.corp/ssl"
    rows = [
        (30, "2026-10-14T11:00:00", "Dev_Lead", "Can you paste the deployment config?"),
        (31, "2026-10-14T11:01:00", "Dev_DevOps", multiline_body),
        (32, "2026-10-14T11:02:00", "Dev_Security", "Config approved [Sec-OK]."),
    ]
    cursor.executemany("INSERT INTO messages (id, timestamp, sender, message) VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()

    anchor_doc = Document(
        page_content=f"[ID 31] [14/10/2026 11:01 AM] Dev_DevOps: {multiline_body}",
        metadata={"source": "hybrid_search", "message_id": 31},
    )

    results = assemble_dialogue_evidence_units(
        [anchor_doc],
        before=1,
        after=1,
        database_path=db_path,
    )

    assert len(results) == 1
    unit = results[0]

    assert unit.metadata["included_message_ids"] == [30, 31, 32]
    # Multiline body must appear verbatim in context
    assert "```json\n{\"port\": 8080, \"enabled\": true}\n```" in unit.page_content
    assert "https://intranet.corp/ssl" in unit.page_content
    assert "[Sec-OK]" in unit.page_content


# ---------------------------------------------------------------------
# Test 8: Overlapping windows from different anchors
# ---------------------------------------------------------------------

def test_overlapping_windows_from_different_anchors(isolated_chat_db: str):
    """
    Two distinct anchors (ID 11 and ID 12) with overlapping contextual windows
    must produce two distinct evidence units with their respective anchor IDs preserved.
    """
    doc11 = Document(
        page_content="[ID 11] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?",
        metadata={"source": "hybrid_search", "message_id": 11, "rank": 1},
    )
    doc12 = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12, "rank": 2},
    )

    results = assemble_dialogue_evidence_units(
        [doc11, doc12],
        before=1,
        after=1,
        database_path=isolated_chat_db,
    )

    # Must produce two distinct units
    assert len(results) == 2

    unit1 = results[0]
    unit2 = results[1]

    assert unit1.metadata["anchor_message_id"] == 11
    assert unit1.metadata["included_message_ids"] == [10, 11, 12]
    assert unit1.metadata["rank"] == 1

    assert unit2.metadata["anchor_message_id"] == 12
    assert unit2.metadata["included_message_ids"] == [11, 12, 13]
    assert unit2.metadata["rank"] == 2


# ---------------------------------------------------------------------
# Test 9: Duplicate input anchors
# ---------------------------------------------------------------------

def test_duplicate_input_anchors(isolated_chat_db: str):
    """Identical input anchors must be deduplicated into a single evidence unit."""
    doc12_a = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )
    doc12_b = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )

    results = assemble_dialogue_evidence_units(
        [doc12_a, doc12_b],
        before=2,
        after=1,
        database_path=isolated_chat_db,
    )

    assert len(results) == 1
    assert results[0].metadata["anchor_message_id"] == 12


# ---------------------------------------------------------------------
# Test 10: Missing anchor rows and malformed metadata
# ---------------------------------------------------------------------

def test_missing_anchor_row_and_malformed_metadata(isolated_chat_db: str):
    """
    Anchors with deleted IDs or malformed metadata must not crash the pipeline;
    they must be safely preserved as standalone evidence units.
    """
    missing_doc = Document(
        page_content="[ID 9999] [14/10/2026 09:59 AM] Unknown: Ghost message.",
        metadata={"source": "hybrid_search", "message_id": 9999},
    )
    malformed_doc = Document(
        page_content="Raw text without an ID bracket.",
        metadata={"source": "hybrid_search", "message_id": "not_an_int"},
    )

    results = assemble_dialogue_evidence_units(
        [missing_doc, malformed_doc],
        before=2,
        after=1,
        database_path=isolated_chat_db,
    )

    assert len(results) == 2

    res_missing = results[0]
    assert res_missing.metadata["context_expanded"] is False
    assert res_missing.metadata["evidence_unit_status"] == "unexpanded_anchor_not_found"
    assert res_missing.metadata["anchor_message_id"] == 9999
    assert res_missing.page_content == missing_doc.page_content

    res_malformed = results[1]
    assert res_malformed.metadata["context_expanded"] is False
    assert res_malformed.metadata["evidence_unit_status"] == "unexpanded_malformed_metadata"
    assert res_malformed.page_content == malformed_doc.page_content


# ---------------------------------------------------------------------
# Test 11: Context lookup failure and safe fallback behavior
# ---------------------------------------------------------------------

def test_context_lookup_failure_fallback(isolated_chat_db: str):
    """If SQLite throws an unexpected OperationalError, the anchor must be preserved safely."""
    anchor_doc = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )

    with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("disk I/O error")):
        results = assemble_dialogue_evidence_units(
            [anchor_doc],
            before=2,
            after=1,
            database_path=isolated_chat_db,
        )

    assert len(results) == 1
    unit = results[0]
    assert unit.metadata["context_expanded"] is False
    assert unit.metadata["evidence_unit_status"] == "unexpanded_error"
    assert "disk I/O error" in unit.metadata["expansion_error"]
    assert unit.page_content == anchor_doc.page_content


# ---------------------------------------------------------------------
# Test 12: Existing conversation chunks remaining unexpanded
# ---------------------------------------------------------------------

def test_existing_conversation_chunks_unexpanded(isolated_chat_db: str):
    """Conversation chunks already represent a multi-message window and must not be expanded twice."""
    chunk_doc = Document(
        page_content="[ID 10] Checking port access.\n[ID 11] Where did we store the backup SSL cert?",
        metadata={
            "source": "conversation_chunk",
            "chunk_id": 42,
            "start_message_id": 10,
            "end_message_id": 11,
            "participants": ["Dev_Security", "Dev_Alex"],
        },
    )

    results = assemble_dialogue_evidence_units(
        [chunk_doc],
        before=2,
        after=1,
        database_path=isolated_chat_db,
    )

    assert len(results) == 1
    unit = results[0]

    assert unit.metadata["evidence_unit_status"] == "preserved_chunk"
    assert unit.metadata["context_method"] == "preexisting_chunk"
    assert unit.metadata["chunk_id"] == 42
    assert unit.page_content == chunk_doc.page_content


# ---------------------------------------------------------------------
# Test 13: Empty input and omitted database path failing safely
# ---------------------------------------------------------------------

def test_empty_input_and_omitted_database_path():
    """Empty inputs and missing database_path must fail safely without raising exceptions."""
    assert assemble_dialogue_evidence_units([]) == []
    assert assemble_dialogue_evidence_units(None) == []

    doc = Document(
        page_content="[ID 12] In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )

    # database_path = None
    res_none = assemble_dialogue_evidence_units([doc], database_path=None)
    assert len(res_none) == 1
    assert res_none[0].metadata["evidence_unit_status"] == "unexpanded_missing_database_path"
    assert res_none[0].metadata["context_expanded"] is False

    # database_path = ""
    res_empty = assemble_dialogue_evidence_units([doc], database_path="")
    assert len(res_empty) == 1
    assert res_empty[0].metadata["evidence_unit_status"] == "unexpanded_missing_database_path"


# ---------------------------------------------------------------------
# Test 14: Input documents remaining unchanged after assembly
# ---------------------------------------------------------------------

def test_input_documents_remain_unchanged(isolated_chat_db: str):
    """Input Document instances must not be mutated in-place by the assembly function."""
    original_metadata = {
        "source": "hybrid_search",
        "message_id": 12,
        "score": 0.85,
        "tags": ["prod", "crypto"],
    }
    original_content = "[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9."

    input_doc = Document(
        page_content=original_content,
        metadata=copy.deepcopy(original_metadata),
    )

    results = assemble_dialogue_evidence_units(
        [input_doc],
        before=2,
        after=1,
        database_path=isolated_chat_db,
    )

    # Output document is transformed
    assert results[0].metadata["context_expanded"] is True

    # Input document must remain 100% untouched
    assert input_doc.page_content == original_content
    assert input_doc.metadata == original_metadata
    assert "context_expanded" not in input_doc.metadata
    assert "included_message_ids" not in input_doc.metadata


# ---------------------------------------------------------------------
# Test 15: No accidental production-database fallback
# ---------------------------------------------------------------------

def test_no_accidental_production_database_fallback(isolated_chat_db: str):
    """
    Verifies that calling assemble_dialogue_evidence_units with an omitted or empty
    database path never touches SQLite or the production database path.
    """
    doc = Document(
        page_content="[ID 12] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 12},
    )

    with patch("sqlite3.connect") as mock_connect:
        assemble_dialogue_evidence_units([doc], database_path=None)
        mock_connect.assert_not_called()

        assemble_dialogue_evidence_units([doc], database_path="")
        mock_connect.assert_not_called()

        assemble_dialogue_evidence_units([doc], database_path="   ")
        mock_connect.assert_not_called()

    # And with an explicit isolated path, sqlite3.connect connects only to that path
    with patch("sqlite3.connect", wraps=sqlite3.connect) as monitored_connect:
        assemble_dialogue_evidence_units([doc], database_path=isolated_chat_db)
        monitored_connect.assert_called_once_with(isolated_chat_db)
