from __future__ import annotations
import copy
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

from langchain_core.documents import Document

from chat_database import (
    get_message,
    get_chronological_context,
    get_connection,
)

# ============================================================
# CONFIG
# ============================================================

# How far we initially inspect around an anchor.
# This is only an optimization/fallback — chronological
# ordering is determined by timestamps.
ID_LOOKAROUND = 100

# Normal conversational gap.
NORMAL_GAP_MINUTES = 180

# Absolute boundary. Never cross this gap.
HARD_GAP_HOURS = 12

# Minimum number of messages we try to keep around an anchor.
MIN_CONTEXT_MESSAGES = 5

# Maximum messages allowed in one context region.
MAX_CONTEXT_MESSAGES = 60

# Maximum number of regions returned.
MAX_REGIONS = 8

# Number of consecutive blank/system-like messages tolerated.
MAX_EMPTY_MESSAGES = 3


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TimelineMessage:
    message_id: int
    timestamp: Optional[datetime]
    sender: str
    message: str
    raw: Any = None


@dataclass
class ContextRegion:
    messages: List[TimelineMessage]
    anchor_ids: List[int]
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def start_id(self) -> Optional[int]:
        if not self.messages:
            return None
        return self.messages[0].message_id

    @property
    def end_id(self) -> Optional[int]:
        if not self.messages:
            return None
        return self.messages[-1].message_id


# ============================================================
# TIMESTAMP PARSING
# ============================================================

def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    text = str(value).strip()

    if not text:
        return None

    # ISO timestamps
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M",
        "%d-%m-%Y %H:%M",
        "%d-%m-%y %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue

    return None


# ============================================================
# MESSAGE NORMALIZATION
# ============================================================

def normalize_message(row: Any) -> Optional[TimelineMessage]:
    if row is None:
        return None

    # ---------------------------------------------------------
    # SQLite Row
    # ---------------------------------------------------------
    if isinstance(row, sqlite3.Row):
        message_id = row["id"]
        timestamp = row["timestamp"]
        sender = row["sender"]
        message = row["message"]

    # ---------------------------------------------------------
    # Dictionary
    # ---------------------------------------------------------
    elif isinstance(row, dict):
        message_id = row.get("id", row.get("message_id"))
        timestamp = row.get(
            "timestamp",
            row.get("datetime", row.get("date")),
        )
        sender = row.get("sender", "")
        message = row.get("message", row.get("text", ""))

    # ---------------------------------------------------------
    # Generic object
    # ---------------------------------------------------------
    else:
        message_id = getattr(
            row,
            "id",
            getattr(row, "message_id", None),
        )

        timestamp = getattr(
            row,
            "timestamp",
            getattr(
                row,
                "datetime",
                getattr(row, "date", None),
            ),
        )

        sender = getattr(row, "sender", "")
        message = getattr(
            row,
            "message",
            getattr(row, "text", ""),
        )

    if message_id is None:
        return None

    timestamp = parse_timestamp(timestamp)

    return TimelineMessage(
        message_id=int(message_id),
        timestamp=timestamp,
        sender=str(sender or ""),
        message=str(message or ""),
        raw=row,
    )


# ============================================================
# FETCH ANCHOR
# ============================================================

def fetch_anchor(anchor_id: int) -> Optional[TimelineMessage]:
    try:
        row = get_message(anchor_id)
    except Exception:
        return None

    return normalize_message(row)


# ============================================================
# FETCH CHRONOLOGICAL DATASET
# ============================================================
def fetch_candidate_timeline(
    anchor_id: int,
    lookaround: int = ID_LOOKAROUND,
) -> List[TimelineMessage]:

    try:
        rows = get_chronological_context(
            message_id=anchor_id,
            before=40,
            after=40,
        )
    except Exception as e:
        print(
            f"Failed to fetch chronological context "
            f"for anchor {anchor_id}: {e}"
        )
        return []

    messages = []

    for row in rows:

        msg = normalize_message(row)

        if msg is None:
            continue

        if msg.timestamp is None:
            continue

        messages.append(msg)

    # -------------------------------------------------------------
    # CRITICAL:
    # Sort by actual timestamp.
    # Never sort by message ID.
    # -------------------------------------------------------------

    messages.sort(
        key=lambda x: (
            x.timestamp,
            x.message_id,
        )
    )

    return messages


# ============================================================
# FIND ANCHOR POSITION CHRONOLOGICALLY
# ============================================================

def find_anchor_index(
    messages: List[TimelineMessage],
    anchor_id: int,
) -> Optional[int]:

    for index, message in enumerate(messages):
        if message.message_id == anchor_id:
            return index

    return None


# ============================================================
# GAP LOGIC
# ============================================================

def gap_minutes(
    earlier: TimelineMessage,
    later: TimelineMessage,
) -> Optional[float]:

    if earlier.timestamp is None or later.timestamp is None:
        return None

    return (later.timestamp - earlier.timestamp).total_seconds() / 60.0


def is_hard_boundary(
    earlier: TimelineMessage,
    later: TimelineMessage,
) -> bool:

    gap = gap_minutes(earlier, later)

    if gap is None:
        return True

    return gap >= HARD_GAP_HOURS * 60


def is_normal_boundary(
    earlier: TimelineMessage,
    later: TimelineMessage,
) -> bool:

    gap = gap_minutes(earlier, later)

    if gap is None:
        return True

    return gap >= NORMAL_GAP_MINUTES


# ============================================================
# EMPTY MESSAGE HANDLING
# ============================================================

def is_empty_message(message: TimelineMessage) -> bool:
    text = message.message.strip()

    return (
        not text
        or text in {
            "<Media omitted>",
            "This message was deleted",
            "You deleted this message",
        }
    )


# ============================================================
# EXPAND ONE ANCHOR
# ============================================================

def expand_single_anchor(
    anchor_id: int,
    min_messages: int = MIN_CONTEXT_MESSAGES,
    max_messages: int = MAX_CONTEXT_MESSAGES,
) -> Optional[ContextRegion]:

    timeline = fetch_candidate_timeline(anchor_id)

    if not timeline:
        return None

    anchor_index = find_anchor_index(timeline, anchor_id)

    if anchor_index is None:
        return None

    anchor = timeline[anchor_index]

    selected = [anchor]

    # --------------------------------------------------------
    # Expand backwards
    # --------------------------------------------------------

    empty_count = 0

    i = anchor_index - 1

    while i >= 0 and len(selected) < max_messages:

        current = timeline[i]
        later = timeline[i + 1]

        # Never cross a huge temporal boundary.
        if is_hard_boundary(current, later):
            break

        # Normal gap is allowed only while we still need
        # minimum context.
        if is_normal_boundary(current, later):
            if len(selected) >= min_messages:
                break

        if is_empty_message(current):
            empty_count += 1

            if empty_count > MAX_EMPTY_MESSAGES:
                break
        else:
            empty_count = 0

        selected.insert(0, current)
        i -= 1

    # --------------------------------------------------------
    # Expand forwards
    # --------------------------------------------------------

    empty_count = 0

    i = anchor_index + 1

    while i < len(timeline) and len(selected) < max_messages:

        previous = timeline[i - 1]
        current = timeline[i]

        if is_hard_boundary(previous, current):
            break

        if is_normal_boundary(previous, current):
            if len(selected) >= min_messages:
                break

        if is_empty_message(current):
            empty_count += 1

            if empty_count > MAX_EMPTY_MESSAGES:
                break
        else:
            empty_count = 0

        selected.append(current)
        i += 1

    # Final chronological ordering.
    selected.sort(
        key=lambda x: (
            x.timestamp,
            x.message_id,
        )
    )

    return ContextRegion(
        messages=selected,
        anchor_ids=[anchor_id],
        score=0.0,
        reasons=["true chronological expansion"],
    )


# ============================================================
# CANDIDATE HELPERS
# ============================================================

def get_candidate_id(candidate: Any) -> Optional[int]:

    # Direct integer anchor ID.
    if isinstance(candidate, int):
        return candidate

    if isinstance(candidate, dict):
        value = candidate.get(
            "message_id",
            candidate.get("id"),
        )
    else:
        value = getattr(
            candidate,
            "message_id",
            getattr(candidate, "id", None),
        )

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def get_candidate_score(candidate: Any) -> float:

    if isinstance(candidate, dict):
        value = candidate.get("score", 0.0)
    else:
        value = getattr(candidate, "score", 0.0)

    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# REGION TEMPORAL OVERLAP
# ============================================================

def regions_temporally_connected(
    a: ContextRegion,
    b: ContextRegion,
) -> bool:

    if not a.messages or not b.messages:
        return False

    a_start = a.messages[0].timestamp
    a_end = a.messages[-1].timestamp

    b_start = b.messages[0].timestamp
    b_end = b.messages[-1].timestamp

    if None in (a_start, a_end, b_start, b_end):
        return False

    # Normalize ordering.
    if b_start < a_start:
        a, b = b, a
        a_start = a.messages[0].timestamp
        a_end = a.messages[-1].timestamp
        b_start = b.messages[0].timestamp
        b_end = b.messages[-1].timestamp

    gap = (b_start - a_end).total_seconds() / 60.0

    # Overlap or close conversational continuity.
    return gap < NORMAL_GAP_MINUTES


# ============================================================
# MERGE REGIONS
# ============================================================

def merge_regions(
    regions: List[ContextRegion],
) -> List[ContextRegion]:

    if not regions:
        return []

    # Sort chronologically.
    regions.sort(
        key=lambda region: (
            region.messages[0].timestamp
            if region.messages
            else datetime.max
        )
    )

    merged: List[ContextRegion] = []

    for region in regions:

        if not merged:
            merged.append(region)
            continue

        previous = merged[-1]

        if regions_temporally_connected(previous, region):

            combined_messages = {
                message.message_id: message
                for message in previous.messages
            }

            for message in region.messages:
                combined_messages[message.message_id] = message

            messages = list(combined_messages.values())

            messages.sort(
                key=lambda x: (
                    x.timestamp,
                    x.message_id,
                )
            )

            merged[-1] = ContextRegion(
                messages=messages,
                anchor_ids=sorted(
                    set(previous.anchor_ids + region.anchor_ids)
                ),
                score=max(
                    previous.score,
                    region.score,
                ),
                reasons=sorted(
                    set(
                        previous.reasons
                        + region.reasons
                        + ["temporally connected regions merged"]
                    )
                ),
            )

        else:
            merged.append(region)

    return merged


# ============================================================
# MAIN EXPANSION
# ============================================================

def expand_candidates(
    candidates: Iterable[Any],
    max_regions: int = MAX_REGIONS,
) -> List[ContextRegion]:

    candidate_list = list(candidates)

    if not candidate_list:
        return []

    regions: List[ContextRegion] = []

    for candidate in candidate_list:

        anchor_id = get_candidate_id(candidate)

        if anchor_id is None:
            continue

        score = get_candidate_score(candidate)

        region = expand_single_anchor(anchor_id)

        if region is None:
            continue

        region.score = score

        regions.append(region)

    # Merge only when timestamps actually connect.
    regions = merge_regions(regions)

    # Recalculate score after merging.
    for region in regions:

        anchor_scores = []

        for candidate in candidate_list:

            candidate_id = get_candidate_id(candidate)

            if candidate_id in region.anchor_ids:
                anchor_scores.append(
                    get_candidate_score(candidate)
                )

        if anchor_scores:
            region.score = (
                max(anchor_scores)
                + 0.10 * max(0, len(anchor_scores) - 1)
            )

            if len(anchor_scores) > 1:
                region.reasons.append(
                    "multiple retrieval anchors"
                )

    # Highest-value regions first.
    regions.sort(
        key=lambda region: region.score,
        reverse=True,
    )

    return regions[:max_regions]


# ============================================================
# FORMAT FOR LLM
# ============================================================

def format_context_regions(
    regions: List[ContextRegion],
) -> str:

    if not regions:
        return ""

    output = []

    for region_index, region in enumerate(regions, start=1):

        output.append(
            f"===== CONTEXT REGION {region_index} ====="
        )

        for message in region.messages:

            timestamp = (
                message.timestamp.isoformat()
                if message.timestamp
                else "unknown-time"
            )

            output.append(
                f"[{message.message_id}] "
                f"[{timestamp}] "
                f"{message.sender}: "
                f"{message.message}"
            )

        output.append(
            f"===== END CONTEXT REGION {region_index} ====="
        )

    return "\n".join(output)


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_context_regions(
    regions: List[ContextRegion],
) -> None:

    print()
    print("=" * 80)
    print(
        f"TRUE CHRONOLOGICAL CONTEXT REGIONS: {len(regions)}"
    )
    print("=" * 80)

    for index, region in enumerate(regions, start=1):

        print()
        print(f"REGION #{index}")

        if region.messages:

            print(
                f"IDs: "
                f"{region.messages[0].message_id}"
                f" → "
                f"{region.messages[-1].message_id}"
            )

            print(
                f"Time: "
                f"{region.messages[0].timestamp}"
                f" → "
                f"{region.messages[-1].timestamp}"
            )

        print(
            f"Anchors: {region.anchor_ids}"
        )

        print(
            f"Messages: {len(region.messages)}"
        )

        print(
            f"Score: {region.score:.3f}"
        )

        print(
            "Reasons: "
            + ", ".join(region.reasons)
        )

        print("-" * 80)

        for message in region.messages:

            timestamp = (
                message.timestamp.isoformat()
                if message.timestamp
                else "unknown"
            )

            print(
                f"[ID {message.message_id}] "
                f"[{timestamp}] "
                f"{message.sender}: "
                f"{message.message}"
            )


# ============================================================
# DIALOGUE EVIDENCE UNIT ASSEMBLY
# ============================================================

def _extract_document_message_id(document: Document) -> Optional[int]:
    """
    Extract integer message ID from a Document's metadata or page_content header.
    Returns None if no valid ID can be resolved.
    """
    if not document:
        return None

    if hasattr(document, "metadata") and document.metadata:
        mid = document.metadata.get("message_id")
        if mid is not None:
            try:
                return int(mid)
            except (ValueError, TypeError):
                pass

    if hasattr(document, "page_content") and document.page_content:
        m = re.search(r"\[(?:ID|message_id)\s*=?\s*(\d+)\]", document.page_content, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, TypeError):
                pass

    return None


def _format_evidence_timestamp(raw_timestamp: Any) -> str:
    """Format timestamp string into readable ISO format or preserve original text."""
    if not raw_timestamp:
        return ""
    text = str(raw_timestamp).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y %I:%M %p")
    except Exception:
        return text


def _format_evidence_message_line(row: Any) -> str:
    """Format a single message row into canonical [ID <id>] [<timestamp>] <sender>: <text> representation."""
    msg_id = row["id"]
    raw_ts = row["timestamp"]
    ts_str = _format_evidence_timestamp(raw_ts)
    sender = row["sender"] or ""
    body = row["message"] or ""
    return f"[ID {msg_id}] [{ts_str}] {sender}: {body}"


def assemble_dialogue_evidence_units(
    anchor_documents: Optional[List[Document]],
    before: int = 2,
    after: int = 1,
    database_path: Optional[str] = None,
) -> List[Document]:
    """
    Converts retrieved chat-message anchors into bounded chronological dialogue
    evidence units before relevance filtering.

    Preserves:
      - Original message text, sender, timestamp, and message ID.
      - Exact anchor identity to distinguish retrieved targets from contextual neighbors.
      - Surrounding turns in deterministic chronological order (timestamp ASC, id ASC).
      - Input documents without in-place mutation.
      - Full citation provenance: contextual inclusion does not alter author/statement mapping.

    Safety:
      - Requires an explicit database_path; never falls back to the production database.
      - Fails open by safely preserving unexpanded standalone units if lookup fails or DB path is missing.
      - Preserves existing conversation-chunk documents without redundant expansion.
      - Deduplicates identical input anchors while preserving distinct overlapping windows.
    """
    if not anchor_documents or not isinstance(anchor_documents, (list, tuple)):
        return []

    before_count = max(0, int(before))
    after_count = max(0, int(after))

    evidence_units: List[Document] = []
    seen_anchors: Set[Any] = set()

    # Safety: do not attempt context expansion if database_path is omitted or empty.
    # We must NEVER fall back to production chat_data.db.
    can_query_db = bool(database_path and isinstance(database_path, str) and database_path.strip())
    db_exists = can_query_db and os.path.exists(database_path)

    for doc in anchor_documents:
        if not doc or not isinstance(doc, Document):
            continue

        doc_meta = dict(doc.metadata) if doc.metadata else {}
        source = doc_meta.get("source", "unknown")
        chunk_id = doc_meta.get("chunk_id")
        msg_id = _extract_document_message_id(doc)

        # -------------------------------------------------------------
        # Deduplication of identical anchors
        # -------------------------------------------------------------
        if source == "conversation_chunk" or chunk_id is not None:
            anchor_key = ("chunk", chunk_id if chunk_id is not None else doc.page_content[:120])
        elif msg_id is not None:
            anchor_key = ("message", msg_id)
        else:
            anchor_key = ("raw", doc.page_content)

        if anchor_key in seen_anchors:
            continue
        seen_anchors.add(anchor_key)

        # -------------------------------------------------------------
        # 1. Conversation Chunks: preserve as-is without second expansion
        # -------------------------------------------------------------
        if source == "conversation_chunk" or chunk_id is not None:
            chunk_meta = copy.deepcopy(doc_meta)
            chunk_meta["context_expanded"] = False
            chunk_meta["context_method"] = "preexisting_chunk"
            chunk_meta["evidence_unit_status"] = "preserved_chunk"
            chunk_meta["anchor_type"] = "conversation_chunk"
            evidence_units.append(
                Document(
                    page_content=doc.page_content,
                    metadata=chunk_meta,
                )
            )
            continue

        # -------------------------------------------------------------
        # 2. Malformed Metadata: no valid integer message ID
        # -------------------------------------------------------------
        if msg_id is None:
            malformed_meta = copy.deepcopy(doc_meta)
            malformed_meta["context_expanded"] = False
            malformed_meta["context_method"] = "standalone_unexpanded"
            malformed_meta["evidence_unit_status"] = "unexpanded_malformed_metadata"
            malformed_meta["included_message_ids"] = []
            evidence_units.append(
                Document(
                    page_content=doc.page_content,
                    metadata=malformed_meta,
                )
            )
            continue

        # -------------------------------------------------------------
        # 3. Missing or Nonexistent Database Path: Fail Safely
        # -------------------------------------------------------------
        if not can_query_db:
            missing_db_meta = copy.deepcopy(doc_meta)
            missing_db_meta["context_expanded"] = False
            missing_db_meta["context_method"] = "standalone_unexpanded"
            missing_db_meta["evidence_unit_status"] = "unexpanded_missing_database_path"
            missing_db_meta["expansion_error"] = "database_path was omitted or empty; expansion aborted to prevent production fallback"
            missing_db_meta["anchor_message_id"] = msg_id
            missing_db_meta["included_message_ids"] = [msg_id]
            evidence_units.append(
                Document(
                    page_content=doc.page_content,
                    metadata=missing_db_meta,
                )
            )
            continue

        if not db_exists:
            no_db_meta = copy.deepcopy(doc_meta)
            no_db_meta["context_expanded"] = False
            no_db_meta["context_method"] = "standalone_unexpanded"
            no_db_meta["evidence_unit_status"] = "unexpanded_database_not_found"
            no_db_meta["expansion_error"] = f"Database file does not exist: {database_path}"
            no_db_meta["anchor_message_id"] = msg_id
            no_db_meta["included_message_ids"] = [msg_id]
            evidence_units.append(
                Document(
                    page_content=doc.page_content,
                    metadata=no_db_meta,
                )
            )
            continue

        # -------------------------------------------------------------
        # 4. Resolve Anchor & Retrieve Chronological Dialogue Unit
        # -------------------------------------------------------------
        try:
            conn = sqlite3.connect(database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Target anchor
            cursor.execute(
                """
                SELECT id, timestamp, sender, message
                FROM messages
                WHERE id = ?
                """,
                (msg_id,),
            )
            target_row = cursor.fetchone()

            if target_row is None:
                conn.close()
                missing_anchor_meta = copy.deepcopy(doc_meta)
                missing_anchor_meta["context_expanded"] = False
                missing_anchor_meta["context_method"] = "standalone_unexpanded"
                missing_anchor_meta["evidence_unit_status"] = "unexpanded_anchor_not_found"
                missing_anchor_meta["expansion_error"] = f"Message ID {msg_id} not found in database"
                missing_anchor_meta["anchor_message_id"] = msg_id
                missing_anchor_meta["included_message_ids"] = [msg_id]
                evidence_units.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=missing_anchor_meta,
                    )
                )
                continue

            target_ts = target_row["timestamp"]
            has_timestamp = bool(target_ts and str(target_ts).strip())

            # Retrieve preceding messages (deterministic timestamp + id tie-breaker)
            if has_timestamp and before_count > 0:
                cursor.execute(
                    """
                    SELECT id, timestamp, sender, message
                    FROM messages
                    WHERE
                        timestamp IS NOT NULL
                        AND (
                            timestamp < ?
                            OR (timestamp = ? AND id < ?)
                        )
                    ORDER BY
                        timestamp DESC,
                        id DESC
                    LIMIT ?
                    """,
                    (target_ts, target_ts, msg_id, before_count),
                )
                before_rows = list(reversed(cursor.fetchall()))
            elif before_count > 0:
                cursor.execute(
                    """
                    SELECT id, timestamp, sender, message
                    FROM messages
                    WHERE id < ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (msg_id, before_count),
                )
                before_rows = list(reversed(cursor.fetchall()))
            else:
                before_rows = []

            # Retrieve following messages (deterministic timestamp + id tie-breaker)
            if has_timestamp and after_count > 0:
                cursor.execute(
                    """
                    SELECT id, timestamp, sender, message
                    FROM messages
                    WHERE
                        timestamp IS NOT NULL
                        AND (
                            timestamp > ?
                            OR (timestamp = ? AND id > ?)
                        )
                    ORDER BY
                        timestamp ASC,
                        id ASC
                    LIMIT ?
                    """,
                    (target_ts, target_ts, msg_id, after_count),
                )
                after_rows = list(cursor.fetchall())
            elif after_count > 0:
                cursor.execute(
                    """
                    SELECT id, timestamp, sender, message
                    FROM messages
                    WHERE id > ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (msg_id, after_count),
                )
                after_rows = list(cursor.fetchall())
            else:
                after_rows = []

            conn.close()

            # Assemble chronological messages and remove duplicate IDs within unit
            raw_combined = before_rows + [target_row] + after_rows
            seen_unit_ids: Set[int] = set()
            deduped_rows = []
            for r in raw_combined:
                rid = int(r["id"])
                if rid not in seen_unit_ids:
                    seen_unit_ids.add(rid)
                    deduped_rows.append(r)

            # Deterministic sorting rule: (timestamp ASC, id ASC)
            deduped_rows.sort(
                key=lambda r: (
                    str(r["timestamp"] or ""),
                    int(r["id"]),
                )
            )

            # Format evidence unit content
            context_lines = [_format_evidence_message_line(r) for r in deduped_rows]
            anchor_text = doc.page_content.strip()

            unit_page_content = (
                "[RETRIEVED ANCHOR]\n"
                f"{anchor_text}\n\n"
                "[CHRONOLOGICAL CONTEXT]\n"
                + "\n".join(context_lines)
            )

            # Build evidence unit metadata preserving anchor identity and citation provenance
            unit_meta = copy.deepcopy(doc_meta)
            unit_meta["context_expanded"] = True
            unit_meta["context_method"] = "dialogue_unit"
            unit_meta["evidence_unit_status"] = "expanded"
            unit_meta["anchor_message_id"] = msg_id
            unit_meta["anchor_sender"] = target_row["sender"] or ""
            unit_meta["anchor_timestamp"] = target_row["timestamp"]
            unit_meta["context_messages_before"] = len(before_rows)
            unit_meta["context_messages_after"] = len(after_rows)
            unit_meta["context_start_message_id"] = int(deduped_rows[0]["id"])
            unit_meta["context_end_message_id"] = int(deduped_rows[-1]["id"])
            unit_meta["included_message_ids"] = [int(r["id"]) for r in deduped_rows]

            evidence_units.append(
                Document(
                    page_content=unit_page_content,
                    metadata=unit_meta,
                )
            )

        except Exception as e:
            fallback_meta = copy.deepcopy(doc_meta)
            fallback_meta["context_expanded"] = False
            fallback_meta["context_method"] = "standalone_unexpanded"
            fallback_meta["evidence_unit_status"] = "unexpanded_error"
            fallback_meta["expansion_error"] = str(e)
            fallback_meta["anchor_message_id"] = msg_id
            fallback_meta["included_message_ids"] = [msg_id]
            evidence_units.append(
                Document(
                    page_content=doc.page_content,
                    metadata=fallback_meta,
                )
            )

    return evidence_units


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    test_anchors = [
        167,
        170,
        175,
        924,
        2141,
        4721,
    ]

    regions = expand_candidates(
        test_anchors,
        max_regions=MAX_REGIONS,
    )

    print_context_regions(regions)

    print()
    print("=" * 80)
    print("LLM CONTEXT PREVIEW")
    print("=" * 80)
    print()

    print(
        format_context_regions(regions)
    )