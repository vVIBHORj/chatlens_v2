"""
test_same_minute_context.py

Deterministic offline regression test suite for Task 62:
Same-Minute Chronological Context Ordering in chat_database.py.

Verifies:
1. Earlier message with an older timestamp is returned.
2. Earlier sibling with the same timestamp and smaller ID is returned.
3. Later message with a newer timestamp is returned.
4. Later sibling with the same timestamp and larger ID is returned.
5. Multiple messages sharing the same minute are ordered by ID.
6. 'before' limit selects the closest preceding messages correctly.
7. 'after' limit selects the closest following messages correctly.
8. Before-context results are returned in chronological order.
9. After-context results are returned in chronological order.
10. Target message is not included in either context result.
11. Messages with NULL timestamps are excluded consistently.
12. Equal timestamps across a date boundary are handled correctly.
13. Explicit custom database_path is respected.
14. Existing default-path callers remain backward compatible.
15. Empty before/after windows behave correctly.
16. Key release build scenario (IDs 100, 101, 102 at 14:30) with target at middle, start, and end.
"""

import sqlite3
from pathlib import Path
import pytest

from chat_database import get_chronological_context


def create_test_db(tmp_path: Path, filename: str, rows) -> str:
    """Helper to create an isolated temporary SQLite database populated with messages."""
    db_path = str(tmp_path / filename)
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
    cursor.execute("CREATE INDEX idx_messages_timestamp ON messages(timestamp)")
    cursor.execute("CREATE INDEX idx_messages_sender ON messages(sender)")

    for row in rows:
        # Pad row to match table schema
        padded = list(row)
        while len(padded) < 11:
            padded.append(None)
        cursor.execute(
            """
            INSERT INTO messages (
                id, timestamp, sender, message, message_type,
                sentiment_compound, sentiment_label, is_question,
                message_length, response_time_minutes, reply_time_minutes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            padded,
        )
    conn.commit()
    conn.close()
    return db_path


# =============================================================================
# 1-4. Preceding and following siblings (older, newer, same timestamp)
# =============================================================================
def test_1_to_4_preceding_and_following_siblings(tmp_path: Path):
    rows = [
        (10, "2026-10-31T14:28:00", "Alice", "Older timestamp message"),
        (11, "2026-10-31T14:30:00", "Bob", "Same minute, smaller ID (earlier sibling)"),
        (12, "2026-10-31T14:30:00", "Charlie", "Target message"),
        (13, "2026-10-31T14:30:00", "David", "Same minute, larger ID (later sibling)"),
        (14, "2026-10-31T14:32:00", "Eve", "Newer timestamp message"),
    ]
    db = create_test_db(tmp_path, "siblings.db", rows)

    res = get_chronological_context(message_id=12, before=5, after=5, database_path=db)
    ids = [r["id"] for r in res]

    # Verify all 5 messages are returned in exact ascending order:
    # 10 (older timestamp), 11 (same minute earlier sibling), 12 (target),
    # 13 (same minute later sibling), 14 (newer timestamp)
    assert ids == [10, 11, 12, 13, 14]


# =============================================================================
# 5. Multiple messages sharing the same minute are ordered by ID
# =============================================================================
def test_5_multiple_messages_same_minute_ordered_by_id(tmp_path: Path):
    rows = [
        (50, "2026-10-31T12:00:00", "UserA", "Msg 50"),
        (51, "2026-10-31T12:00:00", "UserB", "Msg 51"),
        (52, "2026-10-31T12:00:00", "UserC", "Msg 52"),
        (53, "2026-10-31T12:00:00", "UserD", "Msg 53"),
        (54, "2026-10-31T12:00:00", "UserE", "Msg 54"),
    ]
    db = create_test_db(tmp_path, "same_minute_5.db", rows)

    res = get_chronological_context(message_id=52, before=10, after=10, database_path=db)
    ids = [r["id"] for r in res]
    assert ids == [50, 51, 52, 53, 54]


# =============================================================================
# 6 & 7. 'before' and 'after' limits select the closest messages correctly
# =============================================================================
def test_6_and_7_before_and_after_limits(tmp_path: Path):
    rows = [
        (1, "2026-10-31T10:00:00", "A", "Far before"),
        (2, "2026-10-31T10:01:00", "A", "Mid before"),
        (3, "2026-10-31T10:02:00", "A", "Close before (older ts)"),
        (4, "2026-10-31T10:03:00", "A", "Closest before (same minute sibling)"),
        (5, "2026-10-31T10:03:00", "A", "TARGET"),
        (6, "2026-10-31T10:03:00", "A", "Closest after (same minute sibling)"),
        (7, "2026-10-31T10:04:00", "A", "Close after (newer ts)"),
        (8, "2026-10-31T10:05:00", "A", "Mid after"),
        (9, "2026-10-31T10:06:00", "A", "Far after"),
    ]
    db = create_test_db(tmp_path, "limits.db", rows)

    # Ask for closest 2 before and closest 2 after
    res = get_chronological_context(message_id=5, before=2, after=2, database_path=db)
    ids = [r["id"] for r in res]

    # Preceding closest 2: ID 3 (10:02) and ID 4 (10:03, id 4 < 5)
    # Following closest 2: ID 6 (10:03, id 6 > 5) and ID 7 (10:04)
    # Combined with target ID 5:
    assert ids == [3, 4, 5, 6, 7]


# =============================================================================
# 8 & 9. Before-context and After-context are returned in chronological order
# =============================================================================
def test_8_and_9_before_and_after_chronological_order(tmp_path: Path):
    rows = [
        (201, "2026-10-31T15:00:00", "Alice", "B1"),
        (202, "2026-10-31T15:01:00", "Bob", "B2"),
        (203, "2026-10-31T15:01:00", "Charlie", "B3"),
        (204, "2026-10-31T15:02:00", "Target", "Target"),
        (205, "2026-10-31T15:02:00", "David", "A1"),
        (206, "2026-10-31T15:03:00", "Eve", "A2"),
    ]
    db = create_test_db(tmp_path, "order_check.db", rows)

    res = get_chronological_context(message_id=204, before=3, after=2, database_path=db)
    timestamps = [r["timestamp"] for r in res]
    ids = [r["id"] for r in res]

    # Both timestamps and IDs must be non-decreasing
    for i in range(len(res) - 1):
        assert (timestamps[i], ids[i]) < (timestamps[i + 1], ids[i + 1])
    assert ids == [201, 202, 203, 204, 205, 206]


# =============================================================================
# 10. Target message is not included in either context sub-query result
# =============================================================================
def test_10_target_message_not_duplicated_or_misplaced(tmp_path: Path):
    rows = [
        (301, "2026-10-31T16:00:00", "A", "Sibling before"),
        (302, "2026-10-31T16:00:00", "Target", "Target"),
        (303, "2026-10-31T16:00:00", "B", "Sibling after"),
    ]
    db = create_test_db(tmp_path, "no_dup.db", rows)

    res = get_chronological_context(message_id=302, before=10, after=10, database_path=db)
    ids = [r["id"] for r in res]

    # Exactly 3 elements, Target appears exactly once in the center
    assert ids == [301, 302, 303]
    assert ids.count(302) == 1


# =============================================================================
# 11. Messages with NULL timestamps are excluded consistently
# =============================================================================
def test_11_null_timestamps_excluded(tmp_path: Path):
    rows = [
        (401, "2026-10-31T17:00:00", "A", "Valid before"),
        (402, None, "NullGuy", "Corrupt message with NULL timestamp"),
        (403, "2026-10-31T17:05:00", "Target", "Target"),
        (404, None, "NullGuy2", "Another corrupt message with NULL timestamp"),
        (405, "2026-10-31T17:10:00", "B", "Valid after"),
    ]
    db = create_test_db(tmp_path, "null_ts.db", rows)

    res = get_chronological_context(message_id=403, before=5, after=5, database_path=db)
    ids = [r["id"] for r in res]

    # NULL timestamp messages must be excluded from chronological context
    assert 402 not in ids
    assert 404 not in ids
    assert ids == [401, 403, 405]


# =============================================================================
# 12. Equal timestamps across a date boundary are handled correctly
# =============================================================================
def test_12_equal_timestamps_across_date_boundary(tmp_path: Path):
    rows = [
        (501, "2026-10-31T23:59:00", "A", "Late Oct 31"),
        (502, "2026-11-01T00:00:00", "B", "Midnight Nov 1, sibling 1"),
        (503, "2026-11-01T00:00:00", "Target", "Midnight Nov 1, TARGET"),
        (504, "2026-11-01T00:00:00", "C", "Midnight Nov 1, sibling 2"),
        (505, "2026-11-01T00:01:00", "D", "Early Nov 1"),
    ]
    db = create_test_db(tmp_path, "date_boundary.db", rows)

    res = get_chronological_context(message_id=503, before=5, after=5, database_path=db)
    ids = [r["id"] for r in res]
    assert ids == [501, 502, 503, 504, 505]


# =============================================================================
# 13. Explicit custom database_path is respected
# =============================================================================
def test_13_explicit_custom_database_path_respected(tmp_path: Path):
    rows_a = [(1, "2026-10-31T10:00:00", "A", "In DB A")]
    rows_b = [
        (1, "2026-10-31T10:00:00", "B", "In DB B - msg 1"),
        (2, "2026-10-31T10:00:00", "B", "In DB B - TARGET"),
    ]
    db_a = create_test_db(tmp_path, "custom_a.db", rows_a)
    db_b = create_test_db(tmp_path, "custom_b.db", rows_b)

    res = get_chronological_context(message_id=2, before=2, after=2, database_path=db_b)
    ids = [r["id"] for r in res]
    assert ids == [1, 2]
    assert res[0]["sender"] == "B"


# =============================================================================
# 14. Existing default-path callers remain backward compatible
# =============================================================================
def test_14_default_path_compatibility(tmp_path: Path, monkeypatch):
    rows = [
        (10, "2026-10-31T11:00:00", "User", "Target only"),
    ]
    db = create_test_db(tmp_path, "default_test.db", rows)
    import chat_database
    monkeypatch.setattr(chat_database, "DATABASE_PATH", db)

    # Call without database_path keyword argument
    res = get_chronological_context(message_id=10, before=2, after=2)
    assert len(res) == 1
    assert res[0]["id"] == 10


# =============================================================================
# 15. Empty before/after windows behave correctly
# =============================================================================
def test_15_empty_windows_behavior(tmp_path: Path):
    rows = [
        (1, "2026-10-31T12:00:00", "A", "Before"),
        (2, "2026-10-31T12:00:00", "Target", "Target"),
        (3, "2026-10-31T12:00:00", "B", "After"),
    ]
    db = create_test_db(tmp_path, "empty_windows.db", rows)

    # before=0, after=0 -> only target
    res_none = get_chronological_context(message_id=2, before=0, after=0, database_path=db)
    assert [r["id"] for r in res_none] == [2]

    # before=1, after=0 -> before + target
    res_before_only = get_chronological_context(message_id=2, before=1, after=0, database_path=db)
    assert [r["id"] for r in res_before_only] == [1, 2]

    # before=0, after=1 -> target + after
    res_after_only = get_chronological_context(message_id=2, before=0, after=1, database_path=db)
    assert [r["id"] for r in res_after_only] == [2, 3]


# =============================================================================
# 16. Key Scenario: Release Build Dialogue (IDs 100, 101, 102 at 14:30)
# =============================================================================
def test_16_key_scenario_release_build_dialogue(tmp_path: Path):
    """
    Scenario:
      ID 100 — 14:30 — "Did the release build succeed?"
      ID 101 — 14:30 — "Yes, completely clean."
      ID 102 — 14:30 — "I will update the deployment log."

    Target ID 101:
      - Before-context must contain ID 100.
      - After-context must contain ID 102.
      - The final ordering must be 100, 101, 102 when combined.

    Also tests target at the beginning (ID 100) and at the end (ID 102).
    """
    rows = [
        (99, "2026-10-31T14:20:00", "Lead", "Previous discussion"),
        (100, "2026-10-31T14:30:00", "Alice", "Did the release build succeed?"),
        (101, "2026-10-31T14:30:00", "Bob", "Yes, completely clean."),
        (102, "2026-10-31T14:30:00", "Charlie", "I will update the deployment log."),
        (103, "2026-10-31T14:40:00", "Lead", "Later wrap-up message"),
    ]
    db = create_test_db(tmp_path, "release_build.db", rows)

    # 1. Target in the middle: ID 101
    res_101 = get_chronological_context(message_id=101, before=1, after=1, database_path=db)
    ids_101 = [r["id"] for r in res_101]
    assert ids_101 == [100, 101, 102]
    assert res_101[0]["message"] == "Did the release build succeed?"
    assert res_101[1]["message"] == "Yes, completely clean."
    assert res_101[2]["message"] == "I will update the deployment log."

    # 2. Target at the beginning of the same-minute sequence: ID 100
    res_100 = get_chronological_context(message_id=100, before=1, after=2, database_path=db)
    ids_100 = [r["id"] for r in res_100]
    assert ids_100 == [99, 100, 101, 102]

    # 3. Target at the end of the same-minute sequence: ID 102
    res_102 = get_chronological_context(message_id=102, before=2, after=1, database_path=db)
    ids_102 = [r["id"] for r in res_102]
    assert ids_102 == [100, 101, 102, 103]

