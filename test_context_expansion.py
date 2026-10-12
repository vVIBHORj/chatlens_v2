import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from parser import parse_whatsapp_export
from enrich import enrich_messages
from chat_database import save_messages, get_chronological_context
import chat_database
from rag_graph import (
    expand_context,
    validate_citations,
    extract_available_message_ids,
    GraphState,
    CONTEXT_BEFORE,
    CONTEXT_AFTER,
)


@pytest.fixture
def synthetic_chat_db(tmp_path: Path, monkeypatch):
    """
    Creates an isolated temporary SQLite database populated with a synthetic
    WhatsApp chat containing:
      ID 0: Question about coolant canisters
      ID 1: Short reply "Sub-level 4."
      ID 2: Question about maintenance window
      ID 3: Time reply "9pm"
      ID 4: Self-contained factual message about reactor core temperature
      ID 5: Self-contained factual message about telemetry reports
      ID 6-8: Distractor messages (system diagnostics, cafeteria, USB cables)
      ID 9: Short reply "Yes" at the boundary end
    """
    db_path = str(tmp_path / "test_chat.db")
    raw_chat = (
        "10/10/2026, 14:00 - Dr. Thorne: Where did we store the backup coolant canisters?\n"
        "10/10/2026, 14:02 - Elena: Sub-level 4.\n"
        "10/10/2026, 14:05 - Dr. Thorne: What time does the maintenance window begin?\n"
        "10/10/2026, 14:06 - Elena: 9pm\n"
        "10/10/2026, 14:10 - Dr. Thorne: The primary reactor core temperature is stabilized at 310 Kelvin.\n"
        "10/10/2026, 14:15 - Elena: We finished logging all telemetry reports for shift beta.\n"
        "10/10/2026, 14:20 - System: Unrelated system diagnostic routine completed without warning.\n"
        "10/10/2026, 14:25 - Carlos: Cafeteria menu today has vegetable lasagna.\n"
        "10/10/2026, 14:30 - Diana: Anyone have spare USB cables in the breakroom?\n"
        "10/10/2026, 14:35 - Elena: Yes\n"
    )
    chat_file = tmp_path / "synthetic_chat.txt"
    chat_file.write_text(raw_chat, encoding="utf-8")

    messages = parse_whatsapp_export(str(chat_file))
    enriched = enrich_messages(messages)
    save_messages(enriched, database_path=db_path, reset=True)

    # Monkeypatch default DATABASE_PATH so calls default to this isolated database
    monkeypatch.setattr(chat_database, "DATABASE_PATH", db_path)
    return db_path


def make_graph_state(
    documents,
    primary_intent: str = "semantic_rag",
    question: str = "Test question",
    db_path: str = None,
) -> GraphState:
    state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": documents,
        "generation": "",
        "rewrite_count": 0,
        "grounded": False,
        "semantic_grounded": False,
        "df": None,
        "scope": "specific",
        "primary_intent": primary_intent,
        "secondary_intents": [],
        "query_spec": None,
        "citation_validation": None,
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "available_citation_ids": [],
    }
    if db_path:
        state["database_path"] = db_path
    return state


# =====================================================================
# Case 1: Short reply acquires preceding question
# =====================================================================

def test_short_reply_acquires_preceding_question(synthetic_chat_db):
    """
    A short reply such as 'Sub-level 4.' cannot be interpreted on its own.
    When retrieved under semantic_rag, it must gain its preceding question (ID 0).
    """
    doc_short = Document(
        page_content="[ID 1] [10/10/2026 02:02 PM] Elena: Sub-level 4.",
        metadata={
            "source": "hybrid_search",
            "message_id": 1,
            "sender": "Elena",
            "timestamp": "10/10/2026 02:02 PM",
        },
    )
    state = make_graph_state(
        documents=[doc_short],
        primary_intent="semantic_rag",
        question="Where did we store the backup coolant canisters?",
        db_path=synthetic_chat_db,
    )

    result = expand_context(state)
    expanded_docs = result["documents"]

    assert len(expanded_docs) == 1
    expanded = expanded_docs[0]

    # Must be marked as expanded
    assert expanded.metadata.get("context_expanded") is True
    # Preceding question must be present in the expanded context
    assert "[CHRONOLOGICAL CONTEXT]" in expanded.page_content
    assert "[ID 0]" in expanded.page_content
    assert "Where did we store the backup coolant canisters?" in expanded.page_content
    # Anchor must still be present
    assert "[RETRIEVED ANCHOR]" in expanded.page_content
    assert "[ID 1]" in expanded.page_content


# =====================================================================
# Case 2: Time reply gains conversational context
# =====================================================================

def test_time_reply_acquires_conversational_context(synthetic_chat_db):
    """
    A short time reply such as '9pm' requires conversational context
    to establish what activity or event it refers to (ID 2).
    """
    doc_time = Document(
        page_content="[ID 3] [10/10/2026 02:06 PM] Elena: 9pm",
        metadata={
            "source": "hybrid_search",
            "message_id": 3,
            "sender": "Elena",
            "timestamp": "10/10/2026 02:06 PM",
        },
    )
    state = make_graph_state(
        documents=[doc_time],
        primary_intent="semantic_rag",
        question="What time does the maintenance window begin?",
        db_path=synthetic_chat_db,
    )

    result = expand_context(state)
    expanded_docs = result["documents"]

    assert len(expanded_docs) == 1
    expanded = expanded_docs[0]

    assert expanded.metadata.get("context_expanded") is True
    assert "[CHRONOLOGICAL CONTEXT]" in expanded.page_content
    assert "[ID 2]" in expanded.page_content
    assert "What time does the maintenance window begin?" in expanded.page_content


# =====================================================================
# Case 3: Self-contained factual message remains usable without bloat
# =====================================================================

def test_self_contained_factual_message_not_unnecessarily_expanded(synthetic_chat_db):
    """
    A standalone, self-contained factual message has full subject and predicate.
    It should remain usable without pulling in unrelated neighboring messages.
    """
    doc_fact = Document(
        page_content=(
            "[ID 4] [10/10/2026 02:10 PM] Dr. Thorne: "
            "The primary reactor core temperature is stabilized at 310 Kelvin."
        ),
        metadata={
            "source": "hybrid_search",
            "message_id": 4,
            "sender": "Dr. Thorne",
            "timestamp": "10/10/2026 02:10 PM",
        },
    )
    state = make_graph_state(
        documents=[doc_fact],
        primary_intent="semantic_rag",
        question="What is the reactor core temperature?",
        db_path=synthetic_chat_db,
    )

    result = expand_context(state)
    expanded_docs = result["documents"]

    assert len(expanded_docs) == 1
    expanded = expanded_docs[0]

    # Self-contained message must NOT be expanded with unnecessary neighbors
    assert expanded.metadata.get("context_expanded") is False
    assert expanded.metadata.get("context_method") == "retrieved_anchor"
    assert "[CHRONOLOGICAL CONTEXT]" not in expanded.page_content
    assert "[ID 3]" not in expanded.page_content
    assert "[ID 5]" not in expanded.page_content


# =====================================================================
# Case 4: Chronological boundaries (first/last messages)
# =====================================================================

def test_chronological_boundaries_safe(synthetic_chat_db):
    """
    Expansion at conversation boundaries (first message or last message)
    must behave safely without index errors or inventing message IDs.
    """
    # 4a: First message in conversation (ID 0)
    doc_first = Document(
        page_content="[ID 0] [10/10/2026 02:00 PM] Dr. Thorne: Where did we store the backup coolant canisters?",
        metadata={
            "source": "hybrid_search",
            "message_id": 0,
            "sender": "Dr. Thorne",
            "timestamp": "10/10/2026 02:00 PM",
        },
    )
    state_first = make_graph_state(
        documents=[doc_first],
        primary_intent="semantic_rag",
        db_path=synthetic_chat_db,
    )
    result_first = expand_context(state_first)
    assert len(result_first["documents"]) == 1

    # 4b: Last message in conversation (ID 9: 'Yes' - short reply at boundary)
    doc_last = Document(
        page_content="[ID 9] [10/10/2026 02:35 PM] Elena: Yes",
        metadata={
            "source": "hybrid_search",
            "message_id": 9,
            "sender": "Elena",
            "timestamp": "10/10/2026 02:35 PM",
        },
    )
    state_last = make_graph_state(
        documents=[doc_last],
        primary_intent="semantic_rag",
        db_path=synthetic_chat_db,
    )
    result_last = expand_context(state_last)
    assert len(result_last["documents"]) == 1
    expanded_last = result_last["documents"][0]
    # Bounded safely without messages after it
    assert expanded_last.metadata.get("context_expanded") is True
    assert "[ID 8]" in expanded_last.page_content
    assert "[ID 9]" in expanded_last.page_content


# =====================================================================
# Case 5: Distractor control
# =====================================================================

def test_distractor_control_avoids_unrelated_history(synthetic_chat_db):
    """
    Context expansion for semantic queries must NOT indiscriminately
    pull in large amounts of unrelated history. Messages further away
    (e.g. ID 6, 7, 8) must not be included when expanding ID 1.
    """
    doc_short = Document(
        page_content="[ID 1] [10/10/2026 02:02 PM] Elena: Sub-level 4.",
        metadata={
            "source": "hybrid_search",
            "message_id": 1,
            "sender": "Elena",
            "timestamp": "10/10/2026 02:02 PM",
        },
    )
    state = make_graph_state(
        documents=[doc_short],
        primary_intent="semantic_rag",
        db_path=synthetic_chat_db,
    )

    result = expand_context(state)
    expanded = result["documents"][0]

    # Immediate context (ID 0) is included
    assert "[ID 0]" in expanded.page_content

    # Distractor messages further away in the chat must NOT be included
    assert "[ID 6]" not in expanded.page_content
    assert "[ID 7]" not in expanded.page_content
    assert "[ID 8]" not in expanded.page_content
    assert "lasagna" not in expanded.page_content.lower()
    assert "cables" not in expanded.page_content.lower()


# =====================================================================
# Case 6: Citation integrity for chronological neighbors
# =====================================================================

def test_citation_integrity_recognizes_neighbor_message_ids(synthetic_chat_db):
    """
    When a neighbor message (e.g. ID 0) is included in the expanded context,
    its actual message ID must be recognized as available by validate_citations().
    """
    doc_short = Document(
        page_content="[ID 1] [10/10/2026 02:02 PM] Elena: Sub-level 4.",
        metadata={
            "source": "hybrid_search",
            "message_id": 1,
            "sender": "Elena",
            "timestamp": "10/10/2026 02:02 PM",
        },
    )
    state = make_graph_state(
        documents=[doc_short],
        primary_intent="semantic_rag",
        question="Where did we store the backup coolant canisters?",
        db_path=synthetic_chat_db,
    )

    result = expand_context(state)
    expanded_doc = result["documents"][0]

    # Answer cites both the preceding question (ID 0) and the reply (ID 1)
    answer = "The backup coolant canisters were stored in Sub-level 4 [ID 0, ID 1]."
    val = validate_citations(answer, [expanded_doc])

    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert set(val.valid_cited_ids) == {0, 1}
    assert val.invalid_cited_ids == []
    assert 0 in val.available_ids
    assert 1 in val.available_ids


# =====================================================================
# Case 7: Existing CONTEXT intent remains intact
# =====================================================================

def test_existing_context_intent_behavior_intact(synthetic_chat_db):
    """
    The existing behavior for primary_intent == 'context' must remain intact,
    using CONTEXT_BEFORE=10 and CONTEXT_AFTER=10.
    """
    doc = Document(
        page_content="[ID 4] [10/10/2026 02:10 PM] Dr. Thorne: The primary reactor core temperature is stabilized at 310 Kelvin.",
        metadata={
            "source": "hybrid_search",
            "message_id": 4,
            "sender": "Dr. Thorne",
            "timestamp": "10/10/2026 02:10 PM",
        },
    )
    state = make_graph_state(
        documents=[doc],
        primary_intent="context",
        question="What was happening around the reactor stabilization?",
        db_path=synthetic_chat_db,
    )

    result = expand_context(state)
    expanded = result["documents"][0]

    assert expanded.metadata.get("context_expanded") is True
    assert expanded.metadata.get("context_messages_before") == CONTEXT_BEFORE
    assert expanded.metadata.get("context_messages_after") == CONTEXT_AFTER
    assert expanded.metadata.get("context_method") == "chronological"
    assert "[CHRONOLOGICAL CONTEXT]" in expanded.page_content


# =====================================================================
# Case 8: Failure handling (missing record, DB error)
# =====================================================================

def test_failure_handling_missing_records_and_db_error(tmp_path: Path):
    """
    Context fetch failures (nonexistent message_id or DB error) must not
    corrupt graph state, raise unhandled exceptions, or invent message IDs.
    """
    # 8a: Nonexistent message ID in DB
    doc_missing = Document(
        page_content="[ID 9999] [10/10/2026] Unknown: Short text",
        metadata={
            "source": "hybrid_search",
            "message_id": 9999,
            "sender": "Unknown",
        },
    )
    state_missing = make_graph_state(
        documents=[doc_missing],
        primary_intent="semantic_rag",
    )
    # Must not raise an exception
    result_missing = expand_context(state_missing)
    assert len(result_missing["documents"]) == 1
    # Does not invent IDs
    avail_ids = extract_available_message_ids(result_missing["documents"])
    assert 9999 in avail_ids
    assert len(avail_ids) == 1

    # 8b: Database error / locked DB simulation
    with patch("rag_graph.get_chronological_context", side_effect=sqlite3.OperationalError("Database disk image is malformed")):
        state_err = make_graph_state(
            documents=[doc_missing],
            primary_intent="semantic_rag",
        )
        result_err = expand_context(state_err)
        assert len(result_err["documents"]) == 1
        assert result_err["documents"][0].metadata.get("context_expanded") is False
