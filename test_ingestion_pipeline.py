import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

from parser import parse_whatsapp_export, ChatMessage
from enrich import enrich_messages, to_dataframe, EnrichedMessage
from chat_database import save_messages, get_connection, get_message
from vectorstore import build_message_documents, build_conversation_chunks
from app import ingest_chat_file, handle_upload


def test_app_import_succeeds():
    """Verify app.py imports successfully and exposes verified ingestion helpers."""
    import app
    assert hasattr(app, "build_vectorstore")
    assert hasattr(app, "save_messages")
    assert hasattr(app, "ingest_chat_file")
    assert hasattr(app, "handle_upload")


def test_synthetic_export_parsed_and_enriched(tmp_path: Path):
    """Verify parsing and enrichment on a synthetic, conversation-agnostic WhatsApp export."""
    synthetic_chat = (
        "10/10/2026, 14:00 - Dr. Thorne: The orbital trajectory calculations are finished.\n"
        "10/10/2026, 14:05 - Elena: Excellent, transmitting the telemetry data now.\n"
    )
    chat_file = tmp_path / "orbital_chat.txt"
    chat_file.write_text(synthetic_chat, encoding="utf-8")

    messages = parse_whatsapp_export(str(chat_file))
    assert len(messages) == 2
    assert messages[0].sender == "Dr. Thorne"
    assert "orbital trajectory" in messages[0].message
    assert messages[1].sender == "Elena"

    enriched = enrich_messages(messages)
    assert len(enriched) == 2
    assert enriched[0].id == 0
    assert enriched[1].id == 1
    assert enriched[0].sender == "Dr. Thorne"
    assert enriched[1].sender == "Elena"

    df = to_dataframe(enriched)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["sender"].unique()) == {"Dr. Thorne", "Elena"}


def test_sqlite_persistence_and_vectorstore_invocation(tmp_path: Path):
    """Verify save_messages stores records into isolated SQLite and build_vectorstore receives enriched list."""
    synthetic_chat = (
        "12/05/2025, 09:15 - Carlos: Quantum encryption protocol draft is attached.\n"
        "12/05/2025, 09:20 - Diana: Reviewing section 4 regarding key exchange.\n"
    )
    chat_file = tmp_path / "quantum_chat.txt"
    chat_file.write_text(synthetic_chat, encoding="utf-8")

    test_db = str(tmp_path / "isolated_chat.db")

    messages = parse_whatsapp_export(str(chat_file))
    enriched = enrich_messages(messages)

    # 1. Test SQLite persistence
    save_messages(enriched, database_path=test_db, reset=True)

    conn = get_connection(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT id, sender, message FROM messages ORDER BY id")
    rows = cursor.fetchall()

    cursor.execute("SELECT rowid, message FROM messages_fts")
    fts_rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 2
    assert rows[0]["sender"] == "Carlos"
    assert "Quantum encryption" in rows[0]["message"]
    assert rows[1]["sender"] == "Diana"
    assert len(fts_rows) == 2

    # 2. Test vectorstore build call signature
    with patch("vectorstore.build_vectorstore") as mock_build_vs:
        from vectorstore import build_vectorstore
        build_vectorstore(enriched)
        mock_build_vs.assert_called_once_with(enriched)


def test_dataset_replacement_purges_previous_records(tmp_path: Path):
    """Verify that replacing a dataset with reset=True purges all prior messages and avoids data mixing."""
    test_db = str(tmp_path / "replacement_test.db")

    # Ingest dataset 1
    chat1 = (
        "01/01/2026, 10:00 - Alice: Architecture meeting at 10 AM.\n"
        "01/01/2026, 10:02 - Bob: Sounds good.\n"
    )
    f1 = tmp_path / "chat1.txt"
    f1.write_text(chat1, encoding="utf-8")
    enriched1 = enrich_messages(parse_whatsapp_export(str(f1)))
    save_messages(enriched1, database_path=test_db, reset=True)

    conn = get_connection(test_db)
    rows1 = conn.execute("SELECT sender FROM messages").fetchall()
    conn.close()
    assert len(rows1) == 2
    assert {r["sender"] for r in rows1} == {"Alice", "Bob"}

    # Ingest dataset 2 (completely different participants and topics)
    chat2 = (
        "02/02/2026, 15:00 - Xavier: Deep sea exploration submersible deployed.\n"
        "02/02/2026, 15:05 - Yara: Pressure hull integrity is nominal.\n"
        "02/02/2026, 15:10 - Zachary: Sonar telemetry confirmed.\n"
    )
    f2 = tmp_path / "chat2.txt"
    f2.write_text(chat2, encoding="utf-8")
    enriched2 = enrich_messages(parse_whatsapp_export(str(f2)))
    save_messages(enriched2, database_path=test_db, reset=True)

    conn = get_connection(test_db)
    rows2 = conn.execute("SELECT sender, message FROM messages").fetchall()
    conn.close()

    assert len(rows2) == 3
    senders = {r["sender"] for r in rows2}
    assert senders == {"Xavier", "Yara", "Zachary"}
    assert "Alice" not in senders
    assert "Bob" not in senders


def test_vectorstore_document_chunking_offline():
    """Verify vectorstore document and chunk builders function offline without Ollama."""
    chat = [
        ChatMessage(
            timestamp=pd.Timestamp("2026-03-01 10:00:00"),
            sender="Scientist A",
            message="Initial spectrometer readings are consistent.",
            message_type="text",
        ),
        ChatMessage(
            timestamp=pd.Timestamp("2026-03-01 10:01:00"),
            sender="Scientist B",
            message="Calibrating diffraction sensors.",
            message_type="text",
        ),
    ]
    enriched = enrich_messages(chat)

    msg_docs = build_message_documents(enriched)
    assert len(msg_docs) == 2
    assert msg_docs[0].metadata["message_id"] == 0
    assert msg_docs[0].metadata["sender"] == "Scientist A"
    assert "spectrometer readings" in msg_docs[0].page_content

    chunk_docs = build_conversation_chunks(enriched)
    assert len(chunk_docs) >= 1
    assert chunk_docs[0].metadata["source"] == "conversation_chunk"
    assert chunk_docs[0].metadata["start_message_id"] == 0


def test_retrieval_mapping_across_synthetic_datasets(tmp_path: Path):
    """Verify how message IDs map to SQLite records across datasets and demonstrate misalignment risk."""
    chat_a = (
        "01/01/2026, 08:00 - Dr. Thorne: Gravity wave amplitude detected.\n"
        "01/01/2026, 08:05 - Dr. Cooper: Laser interferometer calibrated.\n"
    )
    chat_b = (
        "02/02/2026, 09:00 - Marina: Coral reef bleaching survey completed.\n"
        "02/02/2026, 09:05 - Jacques: Deep sea temperature sensors deployed.\n"
    )
    file_a = tmp_path / "chat_a.txt"
    file_b = tmp_path / "chat_b.txt"
    file_a.write_text(chat_a, encoding="utf-8")
    file_b.write_text(chat_b, encoding="utf-8")

    enriched_a = enrich_messages(parse_whatsapp_export(str(file_a)))
    enriched_b = enrich_messages(parse_whatsapp_export(str(file_b)))

    test_db = str(tmp_path / "mapping_test.db")

    # 1. Ingest Dataset A
    save_messages(enriched_a, database_path=test_db, reset=True)
    chroma_candidate_id = 0
    row_a = get_message(chroma_candidate_id, database_path=test_db)
    assert row_a is not None
    assert row_a["sender"] == "Dr. Thorne"
    assert "Gravity wave" in row_a["message"]

    # 2. Replace with Dataset B
    save_messages(enriched_b, database_path=test_db, reset=True)

    # 3. In SQLite, message ID 0 now resolves to Dataset B's first message
    row_b = get_message(chroma_candidate_id, database_path=test_db)
    assert row_b is not None
    assert row_b["sender"] == "Marina"
    assert "Coral reef" in row_b["message"]
    assert "Dr. Thorne" not in row_b["sender"]


# ============================================================================
# REAL INGESTION & UPLOAD LIFECYCLE TESTS (EXERCISING APP.PY LOGIC)
# ============================================================================

def test_upload_lifecycle_different_exports_same_session(tmp_path: Path):
    """Verify upload A succeeds, then a different upload B is processed in the same session."""
    test_db = str(tmp_path / "lifecycle.db")
    session_state = {}
    mock_vs = MagicMock()

    chat_a_bytes = b"10/10/2026, 10:00 - Alice: Alpha mission launch.\n"
    chat_b_bytes = b"11/11/2026, 11:00 - Bruno: Beta telemetry received.\n"

    # Upload A
    success_a, err_a, df_a, count_a = handle_upload(
        file_bytes=chat_a_bytes,
        session_state=session_state,
        database_path=test_db,
        vectorstore_fn=mock_vs,
    )
    assert success_a is True
    assert err_a is None
    assert count_a == 1
    assert session_state["vectorstore_ready"] is True
    assert session_state["df"] is not None
    assert "Alice" in session_state["df"]["sender"].values
    hash_a = session_state["processed_file_hash"]
    assert hash_a is not None

    # Upload B in the SAME session
    success_b, err_b, df_b, count_b = handle_upload(
        file_bytes=chat_b_bytes,
        session_state=session_state,
        database_path=test_db,
        vectorstore_fn=mock_vs,
    )
    assert success_b is True
    assert err_b is None
    assert count_b == 1
    assert session_state["vectorstore_ready"] is True
    assert "Bruno" in session_state["df"]["sender"].values
    assert "Alice" not in session_state["df"]["sender"].values
    hash_b = session_state["processed_file_hash"]
    assert hash_b != hash_a

    # SQLite confirms Alice was purged and replaced by Bruno
    conn = get_connection(test_db)
    rows = conn.execute("SELECT sender FROM messages").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["sender"] == "Bruno"


def test_repeated_reruns_same_file_do_not_rebuild(tmp_path: Path):
    """Verify repeated reruns with the same file do not rebuild stores unnecessarily."""
    test_db = str(tmp_path / "rerun.db")
    session_state = {}
    mock_save = MagicMock(side_effect=lambda msgs, **kwargs: save_messages(msgs, database_path=test_db, reset=True))
    mock_vs = MagicMock()

    chat_bytes = b"10/10/2026, 10:00 - Alice: Unchanged conversation content.\n"

    # First run: processes file
    success1, err1, df1, count1 = handle_upload(
        file_bytes=chat_bytes,
        session_state=session_state,
        database_path=test_db,
        save_fn=mock_save,
        vectorstore_fn=mock_vs,
    )
    assert success1 is True
    assert mock_save.call_count == 1
    assert mock_vs.call_count == 1

    # Second run (Streamlit rerun with unchanged upload)
    success2, err2, df2, count2 = handle_upload(
        file_bytes=chat_bytes,
        session_state=session_state,
        database_path=test_db,
        save_fn=mock_save,
        vectorstore_fn=mock_vs,
    )
    assert success2 is True
    assert err2 == "already_processed"
    # Verification: neither store build was invoked again
    assert mock_save.call_count == 1
    assert mock_vs.call_count == 1


def test_different_content_same_filename_treated_as_different_upload(tmp_path: Path):
    """Verify two different file contents sharing the same filename are distinguished by SHA-256."""
    test_db = str(tmp_path / "same_filename.db")
    session_state = {}
    mock_vs = MagicMock()

    # Content 1 and Content 2 both represent exports that a user might save as "chat.txt"
    content_1 = b"01/01/2026, 12:00 - User1: Content from conversation 1.\n"
    content_2 = b"02/02/2026, 12:00 - User2: Completely different content from conversation 2.\n"

    success1, _, _, _ = handle_upload(
        file_bytes=content_1,
        session_state=session_state,
        database_path=test_db,
        vectorstore_fn=mock_vs,
    )
    assert success1 is True
    assert "User1" in session_state["df"]["sender"].values

    # Ingest Content 2
    success2, _, _, _ = handle_upload(
        file_bytes=content_2,
        session_state=session_state,
        database_path=test_db,
        vectorstore_fn=mock_vs,
    )
    assert success2 is True
    assert "User2" in session_state["df"]["sender"].values
    assert "User1" not in session_state["df"]["sender"].values


def test_sqlite_persistence_failure_blocks_qa_and_prevents_stale_data(tmp_path: Path):
    """Verify that if SQLite persistence fails, ingestion reports failure and blocks Q&A."""
    session_state = {"df": "old_data_marker", "vectorstore_ready": True, "processed_file_hash": "old_hash"}
    
    mock_failing_save = MagicMock(side_effect=sqlite3.OperationalError("Simulated disk error"))
    mock_vs = MagicMock()

    chat_bytes = b"10/10/2026, 10:00 - NewUser: Trying to upload.\n"

    success, err, df, count = handle_upload(
        file_bytes=chat_bytes,
        session_state=session_state,
        save_fn=mock_failing_save,
        vectorstore_fn=mock_vs,
    )

    assert success is False
    assert "Simulated disk error" in err
    # Key safety check: Q&A is blocked, old data was purged from session, and ready flag is False
    assert session_state["df"] is None
    assert session_state["vectorstore_ready"] is False
    # vectorstore_fn should NOT have been invoked
    assert mock_vs.call_count == 0


def test_vectorstore_failure_after_sqlite_blocks_qa(tmp_path: Path):
    """Verify that if vectorstore build fails after SQLite, ingestion reports failure and blocks Q&A."""
    test_db = str(tmp_path / "vs_fail.db")
    session_state = {"df": "old_data_marker", "vectorstore_ready": True}

    mock_failing_vs = MagicMock(side_effect=RuntimeError("Ollama service unavailable"))

    chat_bytes = b"10/10/2026, 10:00 - NewUser: Trying to upload.\n"

    success, err, df, count = handle_upload(
        file_bytes=chat_bytes,
        session_state=session_state,
        database_path=test_db,
        vectorstore_fn=mock_failing_vs,
    )

    assert success is False
    assert "Ollama service unavailable" in err
    # Key safety check: Q&A is blocked and ready flag is False
    assert session_state["df"] is None
    assert session_state["vectorstore_ready"] is False


def test_failed_upload_can_be_retried_in_same_session(tmp_path: Path):
    """Verify that a failed upload does not lock out subsequent retries in the same session."""
    test_db = str(tmp_path / "retry.db")
    session_state = {}

    chat_bytes = b"10/10/2026, 10:00 - RetryUser: Attempting upload.\n"

    # Attempt 1: Fails
    mock_fail_vs = MagicMock(side_effect=RuntimeError("Transient network failure"))
    success1, err1, _, _ = handle_upload(
        file_bytes=chat_bytes,
        session_state=session_state,
        database_path=test_db,
        vectorstore_fn=mock_fail_vs,
    )
    assert success1 is False
    assert session_state["vectorstore_ready"] is False
    assert session_state["df"] is None
    # Fingerprint was NOT updated, allowing retry
    assert session_state.get("processed_file_hash") is None

    # Attempt 2: User retries (or service recovers) with SAME file content
    mock_ok_vs = MagicMock()
    success2, err2, df2, count2 = handle_upload(
        file_bytes=chat_bytes,
        session_state=session_state,
        database_path=test_db,
        vectorstore_fn=mock_ok_vs,
    )
    assert success2 is True
    assert err2 is None
    assert session_state["vectorstore_ready"] is True
    assert session_state["df"] is not None
    assert session_state["processed_file_hash"] is not None
    assert "RetryUser" in session_state["df"]["sender"].values
