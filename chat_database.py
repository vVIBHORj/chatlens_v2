
"""
chat_database.py

Stores the complete enriched WhatsApp conversation in SQLite.

Architecture:

    WhatsApp TXT
         ↓
    parser.py
         ↓
    enrich.py
         ↓
    EnrichedMessage
         ↓
    SQLite
       ├── messages
       └── messages_fts
         ↓
    Exact search / keyword search / context expansion

SQLite is the source of truth for the complete conversation.

Chroma is used separately as the semantic retrieval index.

Requirements:
    Python standard library only for this file.
"""


import sqlite3
import os

from pathlib import Path
from typing import List, Optional, Dict, Any

from enrich import EnrichedMessage


# =====================================================================
# Configuration
# =====================================================================

DATABASE_PATH = "./chat_data.db"


# =====================================================================
# Database connection
# =====================================================================

def get_connection(
    database_path: str = DATABASE_PATH,
) -> sqlite3.Connection:
    """
    Create and configure a SQLite connection.
    """

    connection = sqlite3.connect(
        database_path
    )

    connection.row_factory = sqlite3.Row

    # Good default for reliability.
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# =====================================================================
# Create database
# =====================================================================

def create_database(
    database_path: str = DATABASE_PATH,
) -> None:
    """
    Create the SQLite database schema.

    Creates:

        messages
        messages_fts

    plus indexes and triggers required to keep the FTS index
    synchronized with the messages table.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    # -----------------------------------------------------------------
    # Main messages table
    # -----------------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (

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

    # -----------------------------------------------------------------
    # Normal indexes
    # -----------------------------------------------------------------

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp
        ON messages(timestamp)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_sender
        ON messages(sender)
        """
    )

    # -----------------------------------------------------------------
    # FTS5 external-content table
    # -----------------------------------------------------------------

    cursor.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(
            message,
            sender,
            content='messages',
            content_rowid='id'
        )
        """
    )

    # -----------------------------------------------------------------
    # INSERT trigger
    # -----------------------------------------------------------------

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_ai
        AFTER INSERT ON messages
        BEGIN

            INSERT INTO messages_fts(
                rowid,
                message,
                sender
            )
            VALUES (
                new.id,
                new.message,
                new.sender
            );

        END;
        """
    )

    # -----------------------------------------------------------------
    # DELETE trigger
    # -----------------------------------------------------------------

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_ad
        AFTER DELETE ON messages
        BEGIN

            INSERT INTO messages_fts(
                messages_fts,
                rowid,
                message,
                sender
            )
            VALUES (
                'delete',
                old.id,
                old.message,
                old.sender
            );

        END;
        """
    )

    # -----------------------------------------------------------------
    # UPDATE trigger
    # -----------------------------------------------------------------

    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_au
        AFTER UPDATE ON messages
        BEGIN

            INSERT INTO messages_fts(
                messages_fts,
                rowid,
                message,
                sender
            )
            VALUES (
                'delete',
                old.id,
                old.message,
                old.sender
            );

            INSERT INTO messages_fts(
                rowid,
                message,
                sender
            )
            VALUES (
                new.id,
                new.message,
                new.sender
            );

        END;
        """
    )

    connection.commit()

    connection.close()


# =====================================================================
# Reset database
# =====================================================================

def reset_database(
    database_path: str = DATABASE_PATH,
) -> None:
    """
    Completely remove the existing SQLite database.

    This is intentionally stronger than:

        DELETE FROM messages

    because an FTS5 external-content index can become inconsistent
    if the previous database was already malformed.

    The next call to create_database() creates everything cleanly.
    """

    database_path_obj = Path(
        database_path
    )

    if not database_path_obj.exists():

        print(
            "No existing SQLite database found."
        )

        return

    print(
        f"Removing existing database: "
        f"{database_path_obj}"
    )

    try:

        database_path_obj.unlink()

        print(
            "✓ Existing SQLite database removed."
        )

    except PermissionError as e:

        raise RuntimeError(
            "Could not remove chat_data.db. "
            "Make sure no other program is using it."
        ) from e


# =====================================================================
# Rebuild FTS index
# =====================================================================

def rebuild_fts_index(
    database_path: str = DATABASE_PATH,
) -> None:
    """
    Rebuild the FTS5 index from the messages table.

    This should be called after bulk inserting messages.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO messages_fts(
                messages_fts
            )
            VALUES ('rebuild')
            """
        )

        connection.commit()

    except sqlite3.DatabaseError as e:

        connection.rollback()

        raise RuntimeError(
            f"Failed to rebuild SQLite FTS index: {e}"
        ) from e

    finally:

        connection.close()

    print(
        "✓ SQLite FTS index rebuilt."
    )


# =====================================================================
# FTS integrity check
# =====================================================================

def verify_fts_index(
    database_path: str = DATABASE_PATH,
) -> bool:
    """
    Verify that the FTS5 index is internally consistent.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO messages_fts(
                messages_fts
            )
            VALUES ('integrity-check')
            """
        )

        connection.commit()

        print(
            "✓ SQLite FTS integrity check passed."
        )

        return True

    except sqlite3.DatabaseError as e:

        print(
            f"❌ SQLite FTS integrity check failed: {e}"
        )

        return False

    finally:

        connection.close()


# =====================================================================
# Save messages
# =====================================================================

def save_messages(
    messages: List[EnrichedMessage],
    database_path: str = DATABASE_PATH,
    reset: bool = True,
) -> None:
    """
    Save all enriched WhatsApp messages into SQLite.

    If reset=True:

        1. Remove the old database completely.
        2. Create a fresh database.
        3. Insert all messages.
        4. Rebuild FTS.
        5. Verify integrity.

    This avoids carrying forward a corrupted FTS index.
    """

    if not messages:

        raise ValueError(
            "No enriched messages were provided."
        )

    # -----------------------------------------------------------------
    # Reset completely if requested
    # -----------------------------------------------------------------

    if reset:

        reset_database(
            database_path
        )

    # -----------------------------------------------------------------
    # Create clean schema
    # -----------------------------------------------------------------

    create_database(
        database_path
    )

    # -----------------------------------------------------------------
    # Open connection
    # -----------------------------------------------------------------

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    # -----------------------------------------------------------------
    # Prepare rows
    # -----------------------------------------------------------------

    rows = []

    for message in messages:

        rows.append(
            (
                message.id,

                (
                    message.timestamp.isoformat()
                    if message.timestamp
                    else None
                ),

                message.sender,

                message.message,

                message.message_type,

                message.sentiment_compound,

                message.sentiment_label,

                int(message.is_question),

                message.message_length,

                message.response_time_minutes,

                message.reply_time_minutes,
            )
        )

    # -----------------------------------------------------------------
    # Insert in one transaction
    # -----------------------------------------------------------------

    try:

        cursor.executemany(
            """
            INSERT INTO messages (

                id,
                timestamp,
                sender,
                message,
                message_type,
                sentiment_compound,
                sentiment_label,
                is_question,
                message_length,
                response_time_minutes,
                reply_time_minutes

            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            rows,
        )

        connection.commit()

    except sqlite3.DatabaseError as e:

        connection.rollback()

        connection.close()

        raise RuntimeError(
            f"Failed to save messages to SQLite: {e}"
        ) from e

    connection.close()

    print(
        f"✓ Saved {len(rows)} messages to SQLite."
    )

    # -----------------------------------------------------------------
    # Rebuild FTS after bulk insert
    # -----------------------------------------------------------------

    rebuild_fts_index(
        database_path
    )

    # -----------------------------------------------------------------
    # Verify FTS
    # -----------------------------------------------------------------

    if not verify_fts_index(
        database_path
    ):

        raise RuntimeError(
            "SQLite FTS index failed integrity verification."
        )


# =====================================================================
# Search messages using FTS5
# =====================================================================
# =====================================================================
# Query preprocessing
# =====================================================================

# These are common question/grammar words that usually add little
# value to lexical retrieval.
#
# IMPORTANT:
# Do not make this list too aggressive.
# WhatsApp contains Hindi, Hinglish, slang, names, typos, etc.
# We want to preserve unusual/user-specific vocabulary.
LEXICAL_STOPWORDS = {
    # English question words
    "what",
    "whats",
    "what's",
    "who",
    "whom",
    "whose",
    "which",
    "where",
    "when",
    "why",
    "how",

    # Common auxiliary / grammar words
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",

    # Pronouns / generic words
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",

    # Common question framing
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",

    "about",
    "discuss",
    "discussed",
    "discussing",
    "tell",
    "told",

    # Common auxiliary/modal words
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",

    # Common temporal/question framing
    "did",
    "have",
    "has",
    "had",
}


def extract_lexical_terms(query: str) -> List[str]:
    """
    Extract useful lexical search terms from a natural-language query.

    The goal is NOT to perform sophisticated NLP.

    We simply:
        1. normalize whitespace
        2. remove obvious punctuation around words
        3. remove common question/grammar words
        4. preserve meaningful user vocabulary

    This is intentionally conservative because WhatsApp chats may
    contain Hindi, Hinglish, slang, names, abbreviations, typos, etc.

    Example:

        "What laptop was EXODIA considering buying?"

    becomes approximately:

        ["laptop", "EXODIA", "considering", "buying"]
    """

    if not query or not query.strip():
        return []

    import re

    raw_words = query.strip().split()

    terms = []

    for word in raw_words:

        # Remove punctuation surrounding a word while preserving
        # internal characters such as apostrophes.
        cleaned = re.sub(
            r"^[^\w@#']+|[^\w@#']+$",
            "",
            word,
            flags=re.UNICODE,
        )

        if not cleaned:
            continue

        normalized = cleaned.lower()

        if normalized in LEXICAL_STOPWORDS:
            continue

        terms.append(cleaned)

    # Remove duplicate terms while preserving original order.
    unique_terms = []
    seen = set()

    for term in terms:

        key = term.lower()

        if key in seen:
            continue

        seen.add(key)
        unique_terms.append(term)

    return unique_terms

# =====================================================================
# Search messages using FTS5
# =====================================================================

def search_messages(
    query: str,
    limit: int = 20,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Search messages using SQLite FTS5.

    The natural-language query is first converted into a small set
    of meaningful lexical terms.

    Example:

        "What laptop was EXODIA considering buying?"

    becomes approximately:

        "laptop" OR "EXODIA" OR "considering" OR "buying"

    This prevents generic question words such as "what", "was",
    "who", "the", etc. from dominating retrieval.

    The original query itself is NOT modified elsewhere in the RAG
    pipeline. This preprocessing applies only to lexical search.
    """

    if not query or not query.strip():
        return []

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    # -----------------------------------------------------------------
    # Extract meaningful lexical terms
    # -----------------------------------------------------------------

    terms = extract_lexical_terms(query)

    if not terms:
        connection.close()
        return []

    # -----------------------------------------------------------------
    # Prepare safe FTS terms
    # -----------------------------------------------------------------

    safe_terms = []

    for term in terms:

        cleaned = term.replace(
            '"',
            '""'
        )

        safe_terms.append(
            f'"{cleaned}"'
        )

    fts_query = " OR ".join(
        safe_terms
    )

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    try:

        cursor.execute(
            """
            SELECT
                messages.*,
                messages_fts.rank AS fts_rank

            FROM messages_fts

            JOIN messages
                ON messages.id = messages_fts.rowid

            WHERE messages_fts MATCH ?

            ORDER BY messages_fts.rank

            LIMIT ?
            """,
            (
                fts_query,
                int(limit),
            ),
        )

        rows = cursor.fetchall()

    except sqlite3.DatabaseError as e:

        connection.close()

        raise RuntimeError(
            f"SQLite keyword search failed: {e}"
        ) from e

    connection.close()

    return rows


# =====================================================================
# Get message by ID
# =====================================================================

def get_message(
    message_id: int,
    database_path: str = DATABASE_PATH,
) -> Optional[sqlite3.Row]:
    """
    Retrieve one message by its stable ID.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE id = ?
        """,
        (
            int(message_id),
        ),
    )

    row = cursor.fetchone()

    connection.close()

    return row

# =====================================================================
# Get chronological neighbors
# =====================================================================

def get_chronological_context(
    message_id: int,
    before: int = 40,
    after: int = 40,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Retrieve messages chronologically surrounding a target message.

    IMPORTANT:
        Message IDs are stable identifiers only.
        They are NOT assumed to represent chronological order.

    The target message's timestamp is used to locate its real
    chronological neighbors.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    # -------------------------------------------------------------
    # Get target message
    # -------------------------------------------------------------

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE id = ?
        """,
        (
            int(message_id),
        ),
    )

    target = cursor.fetchone()

    if target is None:
        connection.close()
        return []

    target_timestamp = target["timestamp"]

    if not target_timestamp:
        connection.close()
        return [target]

    # -------------------------------------------------------------
    # Messages BEFORE target
    # -------------------------------------------------------------

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE
            timestamp IS NOT NULL
            AND timestamp < ?
        ORDER BY
            timestamp DESC,
            id DESC
        LIMIT ?
        """,
        (
            target_timestamp,
            int(before),
        ),
    )

    before_rows = cursor.fetchall()

    # -------------------------------------------------------------
    # Messages AFTER target
    # -------------------------------------------------------------

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE
            timestamp IS NOT NULL
            AND timestamp > ?
        ORDER BY
            timestamp ASC,
            id ASC
        LIMIT ?
        """,
        (
            target_timestamp,
            int(after),
        ),
    )

    after_rows = cursor.fetchall()

    connection.close()

    # -------------------------------------------------------------
    # Combine and sort chronologically
    # -------------------------------------------------------------

    rows = (
        list(reversed(before_rows))
        + [target]
        + list(after_rows)
    )

    rows.sort(
        key=lambda row: (
            row["timestamp"] or "",
            row["id"],
        )
    )

    return rows

# =====================================================================
# Get surrounding context
# =====================================================================

def get_context(
    message_id: int,
    before: int = 10,
    after: int = 10,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Retrieve messages surrounding a target message.

    Example:

        message_id = 100
        before = 10
        after = 10

    approximately returns:

        90 ... 100 ... 110
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    start_id = max(
        0,
        int(message_id) - int(before),
    )

    end_id = (
        int(message_id) + int(after)
    )

    cursor.execute(
        """
        SELECT *
        FROM messages

        WHERE id BETWEEN ? AND ?

        ORDER BY id ASC
        """,
        (
            start_id,
            end_id,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# =====================================================================
# Get exact message range
# =====================================================================

def get_message_range(
    start_id: int,
    end_id: int,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Retrieve an exact inclusive range of messages.
    """

    if start_id > end_id:

        start_id, end_id = (
            end_id,
            start_id,
        )

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM messages

        WHERE id BETWEEN ? AND ?

        ORDER BY id ASC
        """,
        (
            int(start_id),
            int(end_id),
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# =====================================================================
# Database statistics
# =====================================================================

def get_database_stats(
    database_path: str = DATABASE_PATH,
) -> Dict[str, Any]:
    """
    Return basic database statistics.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    # -----------------------------------------------------------------
    # Total messages
    # -----------------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM messages
        """
    )

    total_messages = cursor.fetchone()[0]

    # -----------------------------------------------------------------
    # Total senders
    # -----------------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(DISTINCT sender)
        FROM messages

        WHERE sender IS NOT NULL
        """
    )

    total_senders = cursor.fetchone()[0]

    # -----------------------------------------------------------------
    # Date range
    # -----------------------------------------------------------------

    cursor.execute(
        """
        SELECT
            MIN(timestamp),
            MAX(timestamp)

        FROM messages
        """
    )

    first_timestamp, last_timestamp = (
        cursor.fetchone()
    )

    # -----------------------------------------------------------------
    # Questions
    # -----------------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM messages

        WHERE is_question = 1
        """
    )

    total_questions = cursor.fetchone()[0]

    # -----------------------------------------------------------------
    # Media
    # -----------------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM messages

        WHERE message_type = 'media'
        """
    )

    total_media = cursor.fetchone()[0]

    connection.close()

    return {
        "total_messages": total_messages,
        "total_senders": total_senders,
        "total_questions": total_questions,
        "total_media": total_media,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


# =====================================================================
# Get messages by sender
# =====================================================================

def get_messages_by_sender(
    sender: str,
    limit: Optional[int] = None,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Retrieve messages belonging to a specific sender.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    if limit is None:

        cursor.execute(
            """
            SELECT *
            FROM messages

            WHERE sender = ?

            ORDER BY id ASC
            """,
            (
                sender,
            ),
        )

    else:

        cursor.execute(
            """
            SELECT *
            FROM messages

            WHERE sender = ?

            ORDER BY id ASC

            LIMIT ?
            """,
            (
                sender,
                int(limit),
            ),
        )

    rows = cursor.fetchall()

    connection.close()

    return rows


# =====================================================================
# Get messages between IDs
# =====================================================================

def get_messages_between(
    start_id: int,
    end_id: int,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Alias for get_message_range().

    Kept for readability in future retrieval code.
    """

    return get_message_range(
        start_id=start_id,
        end_id=end_id,
        database_path=database_path,
    )


# =====================================================================
# Format rows as transcript
# =====================================================================

def format_rows_as_transcript(
    rows: List[sqlite3.Row],
) -> str:
    """
    Convert SQLite rows into a readable WhatsApp-style transcript.

    Example:

        [ID 102] [18/09/2022 08:48 PM] Vibhor: Hello
    """

    lines = []

    for row in rows:

        timestamp = row["timestamp"]

        sender = row["sender"]

        message = row["message"]

        try:

            from datetime import datetime

            dt = datetime.fromisoformat(
                timestamp
            )

            timestamp_text = dt.strftime(
                "%d/%m/%Y %I:%M %p"
            )

        except Exception:

            timestamp_text = (
                timestamp
                if timestamp
                else ""
            )

        lines.append(
            f"[ID {row['id']}] "
            f"[{timestamp_text}] "
            f"{sender}: "
            f"{message}"
        )

    return "\n".join(
        lines
    )


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    from parser import parse_whatsapp_export
    from enrich import enrich_messages

    print("=" * 70)
    print("ChatLens-RAG SQLite Database Test")
    print("=" * 70)

    # -------------------------------------------------------------
    # Parse
    # -------------------------------------------------------------

    print(
        "\nParsing WhatsApp export..."
    )

    msgs = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"Parsed {len(msgs)} messages."
    )

    # -------------------------------------------------------------
    # Enrich
    # -------------------------------------------------------------

    print(
        "\nEnriching messages..."
    )

    enriched = enrich_messages(
        msgs
    )

    print(
        f"Enriched {len(enriched)} messages."
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    print(
        "\nSaving database..."
    )

    save_messages(
        enriched,
        reset=True,
    )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    print(
        "\nDatabase statistics:"
    )

    stats = get_database_stats()

    for key, value in stats.items():

        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------
    # Keyword search
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Testing keyword search: sports"
    )

    print(
        "=" * 70
    )

    results = search_messages(
        "sports",
        limit=10,
    )

    if not results:

        print(
            "No keyword matches found."
        )

    else:

        for row in results:

            print(
                f"[ID {row['id']}] "
                f"{row['timestamp']} "
                f"- "
                f"{row['sender']}: "
                f"{row['message']}"
            )

    # -------------------------------------------------------------
    # Context retrieval
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Testing context retrieval"
    )

    print(
        "=" * 70
    )

    if enriched:

        test_id = enriched[
            min(
                10,
                len(enriched) - 1,
            )
        ].id

        print(
            f"Target message ID: {test_id}"
        )

        context = get_context(
            test_id,
            before=3,
            after=3,
        )

        print(
            "\nContext:"
        )

        print(
            format_rows_as_transcript(
                context
            )
        )

    # -------------------------------------------------------------
    # Final integrity check
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Final database verification"
    )

    print(
        "=" * 70
    )

    final_stats = get_database_stats()

    print(
        f"Messages: "
        f"{final_stats['total_messages']}"
    )

    print(
        f"Senders: "
        f"{final_stats['total_senders']}"
    )

    print(
        f"Questions: "
        f"{final_stats['total_questions']}"
    )

    print(
        f"Media: "
        f"{final_stats['total_media']}"
    )

    print(
        f"First message: "
        f"{final_stats['first_timestamp']}"
    )

    print(
        f"Last message: "
        f"{final_stats['last_timestamp']}"
    )

    print(
        "\n✓ SQLite test completed successfully."
    )

