"""
parser.py
Parses a raw WhatsApp chat export (.txt) into structured records with
file-level date order inference (DD/MM vs MM/DD).

WhatsApp export line formats vary slightly by platform/locale, e.g.:
  12/1/24, 9:03 AM - John: Hey, are we still on for lunch today?
  1/12/2024, 21:03 - John: Hey there
  [12/1/24, 9:03:12 AM] John: Hey there   (iOS sometimes wraps in brackets)

Multi-line messages (no timestamp prefix) are appended to the previous message.
System messages ("X joined using this group's invite link", "Messages are
end-to-end encrypted", etc.) are tagged as message_type="system" and excluded
from downstream analysis by default.

Task 61 Update:
Introduces file-level date-order inference to resolve the classic DD/MM vs MM/DD
defect. Pins a single date order for the entire file rather than parsing lines
independently.
"""

import re
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


# Matches the start-of-line timestamp + sender pattern across common export formats.
# Handles optional [brackets], optional seconds, optional AM/PM, and both '-' and
# the unicode en-dash some exports use.
LINE_PATTERN = re.compile(
    r"""^\[?
    (?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)
    (?:\]\s*(?:[-\u2013]\s*)?|\]?\s*[-\u2013]\s*)
    (?P<sender>[^:]+):\s
    (?P<message>.*)$""",
    re.VERBOSE,
)

# System messages have no ": " sender/message split (no one to attribute to).
SYSTEM_LINE_PATTERN = re.compile(
    r"""^\[?
    (?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)
    (?:\]\s*(?:[-\u2013]\s*)?|\]?\s*[-\u2013]\s*)
    (?P<message>[^:]*)$""",
    re.VERBOSE,
)

# Date formats partitioned by order
DMY_DATE_FORMATS = [
    "%d/%m/%y %I:%M %p",
    "%d/%m/%Y %I:%M %p",
    "%d/%m/%y %H:%M",
    "%d/%m/%Y %H:%M",
    "%d/%m/%y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]

MDY_DATE_FORMATS = [
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%y %H:%M",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %I:%M:%S %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]

# Preserved legacy global list (MDY before DMY) for backward compatibility
DATE_FORMATS = [
    "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p",
    "%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p",
    "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M",
    "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M",
    "%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
]


class DateOrderConfidence:
    """Explicit confidence status outcomes for file-level date-order inference."""
    DMY_CONFIDENT = "DMY_CONFIDENT"
    MDY_CONFIDENT = "MDY_CONFIDENT"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_DATE_EVIDENCE = "INSUFFICIENT_DATE_EVIDENCE"


@dataclass
class DateInferenceResult:
    """Detailed audit diagnostic object for date-order inference."""
    detected_order: Optional[str]          # "DMY" | "MDY" | None
    confidence: str                        # Member of DateOrderConfidence
    resolved_order: str                    # "DMY" | "MDY" (the order used for parsing)
    decisive_dmy_count: int = 0
    decisive_mdy_count: int = 0
    ambiguous_count: int = 0
    total_dates: int = 0
    chronological_dmy_plausible: bool = True
    chronological_mdy_plausible: bool = True
    dmy_reversals: int = 0
    mdy_reversals: int = 0
    override_applied: bool = False
    warning: Optional[str] = None
    reason: str = ""


@dataclass
class ChatMessage:
    timestamp: Optional[datetime]
    sender: Optional[str]
    message: str
    message_type: str  # "text" | "media" | "system"


class ChatMessageList(list):
    """
    Subclass of list holding ChatMessage objects with attached date-inference diagnostics.
    Provides 100% backward compatibility for all existing list consumers.
    """
    def __init__(
        self,
        messages: Optional[List[ChatMessage]] = None,
        date_inference: Optional[DateInferenceResult] = None,
    ):
        super().__init__(messages or [])
        self.date_inference = date_inference


def _parse_datetime(
    date_str: str,
    time_str: str,
    date_order: Optional[str] = None,
) -> Optional[datetime]:
    """
    Parse a date and time string using either an explicit date order ('DMY' or 'MDY')
    or the legacy format list if date_order is omitted.
    """
    raw = f"{date_str} {time_str}".replace("\u202f", " ").strip()

    if date_order:
        norm = date_order.strip().upper()
        if norm == "DMY":
            formats = DMY_DATE_FORMATS
        elif norm == "MDY":
            formats = MDY_DATE_FORMATS
        else:
            raise ValueError(f"Unknown date_order: {date_order!r}. Expected 'DMY' or 'MDY'.")
    else:
        formats = DATE_FORMATS

    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _classify_message(text: str) -> str:
    if "<Media omitted>" in text or "image omitted" in text.lower() or "video omitted" in text.lower():
        return "media"
    return "text"


def _count_chronological_reversals(times: List[datetime]) -> int:
    """
    Count the number of times timestamp i+1 is strictly earlier than timestamp i.
    In a valid WhatsApp export, messages appear in non-decreasing chronological order.
    A reversal indicates an impossible or corrupted chronological sequence.
    """
    if len(times) < 2:
        return 0
    reversals = 0
    for i in range(len(times) - 1):
        if times[i + 1] < times[i]:
            reversals += 1
    return reversals


def infer_date_order(
    lines_or_text: Union[List[str], str],
    date_order: str = "auto",
) -> DateInferenceResult:
    """
    Infers one consistent file-level date order (DMY vs MDY) across an entire export.

    Algorithm:
    1. Collect valid date tokens from matching WhatsApp message and system lines.
    2. Identify decisive evidence:
       - First numeric component > 12 -> DD/MM (DMY decisive)
       - Second numeric component > 12 -> MM/DD (MDY decisive)
    3. Verify consistency of decisive observations:
       - Both DMY and MDY decisive tokens present -> CONFLICTING_EVIDENCE
       - Only DMY decisive tokens -> DMY_CONFIDENT
       - Only MDY decisive tokens -> MDY_CONFIDENT
    4. If no decisive tokens (all components <= 12):
       - Test chronological consistency under both DMY and MDY interpretations.
       - If only one produces a valid, non-decreasing sequence -> CONFIDENT via secondary evidence.
       - If both remain chronologically plausible -> AMBIGUOUS.
    5. Fallback policy:
       - If date_order is 'auto' and file is AMBIGUOUS, CONFLICTING_EVIDENCE, or
         INSUFFICIENT_DATE_EVIDENCE, falls back to legacy default 'MDY' with an explicit warning.
       - If explicit 'DMY' or 'MDY' override is supplied, respects override.
    """
    if not isinstance(date_order, str):
        raise ValueError(f"Invalid date_order type: {type(date_order)}. Expected str ('auto', 'DMY', 'MDY').")

    valid_order = date_order.strip().upper()
    if valid_order not in ("AUTO", "DMY", "MDY"):
        raise ValueError(f"Invalid date_order: {date_order!r}. Expected 'auto', 'DMY', or 'MDY'.")

    if isinstance(lines_or_text, str):
        lines = lines_or_text.splitlines()
    else:
        lines = lines_or_text

    date_observations: List[Tuple[str, str, int, int, int]] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue

        match = LINE_PATTERN.match(line)
        if match and len(match.group("sender").strip()) > 60:
            match = None
        if not match:
            match = SYSTEM_LINE_PATTERN.match(line)
        if not match:
            continue

        date_str = match.group("date")
        time_str = match.group("time")

        parts = date_str.split("/")
        if len(parts) == 3:
            try:
                p1, p2, yr = int(parts[0]), int(parts[1]), int(parts[2])
                date_observations.append((date_str, time_str, p1, p2, yr))
            except ValueError:
                continue

    decisive_dmy = 0
    decisive_mdy = 0
    ambiguous = 0
    invalid = 0

    for _, _, p1, p2, _ in date_observations:
        if p1 > 12 and 1 <= p2 <= 12:
            decisive_dmy += 1
        elif p2 > 12 and 1 <= p1 <= 12:
            decisive_mdy += 1
        elif 1 <= p1 <= 12 and 1 <= p2 <= 12:
            ambiguous += 1
        else:
            invalid += 1

    total_dates = len(date_observations)
    dmy_reversals = 0
    mdy_reversals = 0
    dmy_plausible = True
    mdy_plausible = True

    detected_order: Optional[str] = None
    confidence: str = DateOrderConfidence.INSUFFICIENT_DATE_EVIDENCE
    reason: str = ""

    if total_dates == 0:
        detected_order = None
        confidence = DateOrderConfidence.INSUFFICIENT_DATE_EVIDENCE
        reason = "No valid timestamped message lines found in chat export."
    elif decisive_dmy > 0 and decisive_mdy > 0:
        detected_order = None
        confidence = DateOrderConfidence.CONFLICTING_EVIDENCE
        reason = (
            f"Conflicting decisive date evidence detected: {decisive_dmy} tokens indicate "
            f"DMY (first component > 12), and {decisive_mdy} tokens indicate MDY (second component > 12)."
        )
    elif decisive_dmy > 0 and decisive_mdy == 0:
        detected_order = "DMY"
        confidence = DateOrderConfidence.DMY_CONFIDENT
        reason = f"Consistent decisive DMY evidence detected ({decisive_dmy} tokens with day > 12 in first position)."
    elif decisive_mdy > 0 and decisive_dmy == 0:
        detected_order = "MDY"
        confidence = DateOrderConfidence.MDY_CONFIDENT
        reason = f"Consistent decisive MDY evidence detected ({decisive_mdy} tokens with day > 12 in second position)."
    else:
        # All valid tokens have p1 <= 12 and p2 <= 12.
        # Evaluate secondary evidence: Chronological Consistency
        dmy_times: List[datetime] = []
        mdy_times: List[datetime] = []

        for d_str, t_str, _, _, _ in date_observations:
            t_dmy = _parse_datetime(d_str, t_str, date_order="DMY")
            if t_dmy:
                dmy_times.append(t_dmy)
            t_mdy = _parse_datetime(d_str, t_str, date_order="MDY")
            if t_mdy:
                mdy_times.append(t_mdy)

        dmy_reversals = _count_chronological_reversals(dmy_times)
        mdy_reversals = _count_chronological_reversals(mdy_times)

        dmy_plausible = (dmy_reversals == 0) and (len(dmy_times) > 0)
        mdy_plausible = (mdy_reversals == 0) and (len(mdy_times) > 0)

        if dmy_plausible and not mdy_plausible:
            detected_order = "DMY"
            confidence = DateOrderConfidence.DMY_CONFIDENT
            reason = (
                f"Secondary chronological evidence: DMY interpretation is consistent (0 reversals), "
                f"while MDY has {mdy_reversals} chronological reversal(s)."
            )
        elif mdy_plausible and not dmy_plausible:
            detected_order = "MDY"
            confidence = DateOrderConfidence.MDY_CONFIDENT
            reason = (
                f"Secondary chronological evidence: MDY interpretation is consistent (0 reversals), "
                f"while DMY has {dmy_reversals} chronological reversal(s)."
            )
        else:
            detected_order = None
            confidence = DateOrderConfidence.AMBIGUOUS
            reason = (
                f"All date components <= 12 ({ambiguous} ambiguous tokens) and both DMY and MDY "
                "interpretations are chronologically plausible."
            )

    # Resolution and override handling
    override_applied = False
    warning: Optional[str] = None

    if valid_order in ("DMY", "MDY"):
        resolved_order = valid_order
        override_applied = True
        if detected_order is not None and detected_order != resolved_order:
            warning = (
                f"Explicit date_order='{resolved_order}' overrides inferred detected_order='{detected_order}'."
            )
            warnings.warn(warning, UserWarning, stacklevel=2)
    else:  # AUTO
        if detected_order is not None:
            resolved_order = detected_order
        else:
            # Fall back to legacy default (MDY)
            resolved_order = "MDY"
            if confidence == DateOrderConfidence.AMBIGUOUS:
                warning = (
                    "WhatsApp chat export date order is ambiguous (all date components <= 12 and both interpretations "
                    "are chronologically plausible). Falling back to legacy default 'MDY'. Specify date_order='DMY' "
                    "or date_order='MDY' to override."
                )
                warnings.warn(warning, UserWarning, stacklevel=2)
            elif confidence == DateOrderConfidence.CONFLICTING_EVIDENCE:
                warning = (
                    "WhatsApp chat export contains conflicting date evidence (both DD > 12 and MM > 12 observed). "
                    "Falling back to legacy default 'MDY'. Specify date_order='DMY' or date_order='MDY' to override."
                )
                warnings.warn(warning, UserWarning, stacklevel=2)
            elif confidence == DateOrderConfidence.INSUFFICIENT_DATE_EVIDENCE:
                warning = (
                    "WhatsApp chat export contains insufficient date evidence. Falling back to legacy default 'MDY'."
                )
                warnings.warn(warning, UserWarning, stacklevel=2)

    return DateInferenceResult(
        detected_order=detected_order,
        confidence=confidence,
        resolved_order=resolved_order,
        decisive_dmy_count=decisive_dmy,
        decisive_mdy_count=decisive_mdy,
        ambiguous_count=ambiguous,
        total_dates=total_dates,
        chronological_dmy_plausible=dmy_plausible,
        chronological_mdy_plausible=mdy_plausible,
        dmy_reversals=dmy_reversals,
        mdy_reversals=mdy_reversals,
        override_applied=override_applied,
        warning=warning,
        reason=reason,
    )


def _parse_lines(
    lines: List[str],
    date_order: str = "auto",
) -> ChatMessageList:
    """Internal helper to parse pre-read lines using file-level date inference."""
    inference = infer_date_order(lines, date_order=date_order)
    resolved_order = inference.resolved_order

    messages: List[ChatMessage] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue

        match = LINE_PATTERN.match(line)
        # Sender names are short by convention; a very long "sender" capture
        # (e.g. > 60 chars) almost always means the line had no real sender.
        if match and len(match.group("sender").strip()) > 60:
            match = None

        if match:
            dt = _parse_datetime(match.group("date"), match.group("time"), date_order=resolved_order)
            sender = match.group("sender").strip()
            text = match.group("message").strip()
            messages.append(
                ChatMessage(
                    timestamp=dt,
                    sender=sender,
                    message=text,
                    message_type=_classify_message(text),
                )
            )
            continue

        sys_match = SYSTEM_LINE_PATTERN.match(line)
        if sys_match:
            dt = _parse_datetime(sys_match.group("date"), sys_match.group("time"), date_order=resolved_order)
            messages.append(
                ChatMessage(
                    timestamp=dt,
                    sender=None,
                    message=sys_match.group("message").strip(),
                    message_type="system",
                )
            )
            continue

        # No timestamp match -> continuation of the previous multi-line message
        if messages:
            messages[-1].message += "\n" + line.strip()

    return ChatMessageList(messages, date_inference=inference)


def parse_whatsapp_text(
    text: str,
    date_order: str = "auto",
) -> ChatMessageList:
    """Parse raw WhatsApp export text into a ChatMessageList with file-level date inference."""
    lines = text.splitlines(keepends=True)
    return _parse_lines(lines, date_order=date_order)


def parse_whatsapp_text_with_diagnostics(
    text: str,
    date_order: str = "auto",
) -> Tuple[ChatMessageList, DateInferenceResult]:
    """Parse raw WhatsApp export text, returning messages and the date-inference diagnostic object."""
    messages = parse_whatsapp_text(text, date_order=date_order)
    return messages, messages.date_inference  # type: ignore


def parse_whatsapp_export(
    filepath: str,
    date_order: str = "auto",
) -> ChatMessageList:
    """
    Parse a WhatsApp .txt export into a ChatMessageList records with file-level date inference.
    Preserves 100% backward compatibility as a standard list subclass.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return _parse_lines(lines, date_order=date_order)


def parse_whatsapp_export_with_diagnostics(
    filepath: str,
    date_order: str = "auto",
) -> Tuple[ChatMessageList, DateInferenceResult]:
    """Parse a WhatsApp .txt export, returning messages and the date-inference diagnostic object."""
    messages = parse_whatsapp_export(filepath, date_order=date_order)
    return messages, messages.date_inference  # type: ignore


def to_dict_list(messages: List[ChatMessage]) -> List[dict]:
    out = []
    for m in messages:
        d = asdict(m)
        d["timestamp"] = m.timestamp.isoformat() if m.timestamp else None
        out.append(d)
    return out


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_chat.txt"
    parsed, diag = parse_whatsapp_export_with_diagnostics(path)
    print(f"Date inference: detected={diag.detected_order}, confidence={diag.confidence}, resolved={diag.resolved_order}")
    print(f"Parsed {len(parsed)} messages\n")
    for m in parsed[:5]:
        print(m)
