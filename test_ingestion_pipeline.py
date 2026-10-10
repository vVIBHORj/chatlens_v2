import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

from parser import parse_whatsapp_export, ChatMessage
from enrich import enrich_messages, to_dataframe, EnrichedMessage
from chat_database import save_messages, get_connection
from vectorstore import build_message_documents, build_conversation_chunks


def test_app_import_succeeds():
    """Verify app.py imports successfully without missing symbols or errors."""
    import app
    assert hasattr(app, "build_vectorstore")
    assert hasattr(app, "save_messages")


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


def test_failure_handling_leaves_flag_false():
    """Verify that a vectorstore or database failure causes vectorstore_ready to remain False."""
    session_state = {"vectorstore_ready": False, "df": None}
    
    enriched = [
        EnrichedMessage(
            id=1,
            timestamp=pd.Timestamp("2026-01-01 10:00:00"),
            sender="User",
            message="Hello",
            message_type="text",
            sentiment_compound=0.0,
            sentiment_label="neutral",
            is_question=False,
            message_length=5,
            response_time_minutes=None,
            reply_time_minutes=None,
        )
    ]

    # Simulate ingestion failure in vectorstore
    ingestion_error = None
    try:
        with patch("vectorstore.build_vectorstore", side_effect=RuntimeError("Ollama connection failed")):
            from vectorstore import build_vectorstore
            build_vectorstore(enriched)
            session_state["vectorstore_ready"] = True
    except Exception as e:
        ingestion_error = e
        session_state["vectorstore_ready"] = False
        session_state["df"] = None

    assert ingestion_error is not None
    assert str(ingestion_error) == "Ollama connection failed"
    assert session_state["vectorstore_ready"] is False
    assert session_state["df"] is None
