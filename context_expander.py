from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chat_database import get_message, get_message_range


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

    if isinstance(row, dict):
        message_id = row.get("id", row.get("message_id"))
        timestamp = row.get("timestamp", row.get("datetime", row.get("date")))
        sender = row.get("sender", "")
        message = row.get("message", row.get("text", ""))

    else:
        message_id = getattr(row, "id", getattr(row, "message_id", None))
        timestamp = getattr(
            row,
            "timestamp",
            getattr(row, "datetime", getattr(row, "date", None)),
        )
        sender = getattr(row, "sender", "")
        message = getattr(row, "message", getattr(row, "text", ""))

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

    start_id = max(1, anchor_id - lookaround)
    end_id = anchor_id + lookaround

    try:
        rows = get_message_range(start_id, end_id)
    except Exception:
        return []

    messages = []

    for row in rows:
        msg = normalize_message(row)

        if msg is None:
            continue

        if msg.timestamp is None:
            continue

        messages.append(msg)

    # CRITICAL:
    # IDs are NOT used to determine chronological order.
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