"""
test_whatsapp_date_parser.py

Deterministic offline regression test suite for Task 61:
File-Level WhatsApp Date Format Detection and Chronology Integrity.

Verifies:
1. DMY export containing 04/05/24 and 13/05/24.
2. MDY export containing 05/04/24 and 05/13/24.
3. Fully ambiguous DMY dates where both components are 12 or below.
4. Fully ambiguous MDY dates where both components are 12 or below.
5. A file with contradictory decisive date evidence.
6. Chronological consistency selecting one unique plausible interpretation.
7. Both interpretations remaining chronologically plausible.
8. Month-boundary chronological ordering.
9. Dates spanning a year boundary.
10. Two-digit and four-digit years.
11. 12-hour and 24-hour time formats.
12. Explicit date_order="DMY" override.
13. Explicit date_order="MDY" override.
14. Invalid date tokens and malformed lines.
15. Existing parser behavior for supported WhatsApp export formats.
16. Section 6 Chronology Integrity Validation (DMY May 4-15 11-day elapsed vs 40-day bug, and MDY).
"""

from datetime import datetime, timedelta
import pytest

from parser import (
    ChatMessage,
    ChatMessageList,
    DateInferenceResult,
    DateOrderConfidence,
    infer_date_order,
    parse_whatsapp_text,
    parse_whatsapp_text_with_diagnostics,
    _parse_datetime,
)


# =============================================================================
# 1. DMY export containing 04/05/24 and 13/05/24
# =============================================================================
def test_1_dmy_decisive_export():
    chat = (
        "04/05/24, 09:00 - Alice: Project kickoff\n"
        "13/05/24, 17:30 - Bob: Sprint review\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    assert diag.confidence == DateOrderConfidence.DMY_CONFIDENT
    assert diag.detected_order == "DMY"
    assert diag.resolved_order == "DMY"
    assert diag.decisive_dmy_count == 1
    assert diag.decisive_mdy_count == 0

    # 04/05/24 must be May 4, 2024, NOT April 5
    assert messages[0].timestamp == datetime(2024, 5, 4, 9, 0)
    assert messages[1].timestamp == datetime(2024, 5, 13, 17, 30)


# =============================================================================
# 2. MDY export containing 05/04/24 and 05/13/24
# =============================================================================
def test_2_mdy_decisive_export():
    chat = (
        "05/04/24, 09:00 - Alice: Project kickoff\n"
        "05/13/24, 17:30 - Bob: Sprint review\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    assert diag.confidence == DateOrderConfidence.MDY_CONFIDENT
    assert diag.detected_order == "MDY"
    assert diag.resolved_order == "MDY"
    assert diag.decisive_mdy_count == 1
    assert diag.decisive_dmy_count == 0

    # 05/04/24 must be May 4, 2024
    assert messages[0].timestamp == datetime(2024, 5, 4, 9, 0)
    assert messages[1].timestamp == datetime(2024, 5, 13, 17, 30)


# =============================================================================
# 3. Fully ambiguous DMY dates where both components are 12 or below
# =============================================================================
def test_3_ambiguous_dmy_dates_surfaces_warning():
    chat = (
        "01/02/24, 10:00 - Alice: Meeting morning\n"
        "01/02/24, 14:00 - Bob: Afternoon sync\n"
    )
    with pytest.warns(UserWarning, match="ambiguous"):
        messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    assert diag.confidence == DateOrderConfidence.AMBIGUOUS
    assert diag.detected_order is None
    # Falls back to legacy default (MDY)
    assert diag.resolved_order == "MDY"
    assert diag.ambiguous_count == 2
    assert diag.chronological_dmy_plausible is True
    assert diag.chronological_mdy_plausible is True


# =============================================================================
# 4. Fully ambiguous MDY dates where both components are 12 or below
# =============================================================================
def test_4_ambiguous_mdy_dates():
    chat = (
        "02/01/24, 10:00 - Alice: Hello\n"
        "02/01/24, 10:05 - Bob: Hi there\n"
    )
    with pytest.warns(UserWarning, match="ambiguous"):
        messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    assert diag.confidence == DateOrderConfidence.AMBIGUOUS
    assert diag.detected_order is None
    assert diag.resolved_order == "MDY"
    assert diag.total_dates == 2


# =============================================================================
# 5. A file with contradictory decisive date evidence
# =============================================================================
def test_5_contradictory_decisive_date_evidence():
    chat = (
        "15/05/24, 10:00 - Alice: European format line (day 15 > 12)\n"
        "05/20/24, 11:00 - Bob: US format line (day 20 > 12 in second position)\n"
    )
    with pytest.warns(UserWarning, match="conflicting"):
        messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    assert diag.confidence == DateOrderConfidence.CONFLICTING_EVIDENCE
    assert diag.detected_order is None
    assert diag.decisive_dmy_count == 1
    assert diag.decisive_mdy_count == 1
    # Does not flip date order back and forth; uses resolved default
    assert diag.resolved_order == "MDY"
    # Line 1 (15/05/24) cannot parse under MDY (month 15 invalid) -> timestamp is None, preserving safety
    assert messages[0].timestamp is None
    # Line 2 (05/20/24) parses under MDY
    assert messages[1].timestamp == datetime(2024, 5, 20, 11, 0)


# =============================================================================
# 6. Chronological consistency selecting one unique plausible interpretation
# =============================================================================
def test_6_chronological_consistency_uniquely_selects_order():
    # Sequence: May 1 -> May 2 -> June 1 (under MDY)
    # Under DMY: Jan 5 -> Feb 5 -> Jan 6 (reversal from Feb 5 to Jan 6!)
    chat = (
        "05/01/24, 10:00 - Alice: Day 1\n"
        "05/02/24, 10:00 - Alice: Day 2\n"
        "06/01/24, 10:00 - Alice: Day 3\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    assert diag.confidence == DateOrderConfidence.MDY_CONFIDENT
    assert diag.detected_order == "MDY"
    assert diag.resolved_order == "MDY"
    assert diag.chronological_mdy_plausible is True
    assert diag.chronological_dmy_plausible is False
    assert diag.dmy_reversals >= 1
    assert diag.mdy_reversals == 0

    assert messages[0].timestamp == datetime(2024, 5, 1, 10, 0)
    assert messages[1].timestamp == datetime(2024, 5, 2, 10, 0)
    assert messages[2].timestamp == datetime(2024, 6, 1, 10, 0)


# =============================================================================
# 7. Both interpretations remaining chronologically plausible
# =============================================================================
def test_7_both_interpretations_chronologically_plausible():
    chat = (
        "03/04/24, 09:00 - Alice: Start\n"
        "03/04/24, 09:30 - Bob: In progress\n"
    )
    diag = infer_date_order(chat)
    assert diag.confidence == DateOrderConfidence.AMBIGUOUS
    assert diag.chronological_dmy_plausible is True
    assert diag.chronological_mdy_plausible is True


# =============================================================================
# 8. Month-boundary chronological ordering
# =============================================================================
def test_8_month_boundary_chronological_ordering():
    # DMY sequence: Feb 11 -> Feb 12 -> Mar 1 (increasing across month boundary)
    # Under MDY: Nov 2 -> Dec 2 -> Jan 3 (backward jump of 11 months from Dec 2 to Jan 3!)
    chat = (
        "11/02/24, 10:00 - Alice: Feb 11\n"
        "12/02/24, 10:00 - Alice: Feb 12\n"
        "01/03/24, 10:00 - Alice: Mar 1\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    assert diag.confidence == DateOrderConfidence.DMY_CONFIDENT
    assert diag.detected_order == "DMY"
    assert diag.resolved_order == "DMY"
    assert diag.chronological_dmy_plausible is True
    assert diag.chronological_mdy_plausible is False
    assert diag.mdy_reversals >= 1

    assert messages[0].timestamp == datetime(2024, 2, 11, 10, 0)
    assert messages[1].timestamp == datetime(2024, 2, 12, 10, 0)
    assert messages[2].timestamp == datetime(2024, 3, 1, 10, 0)


# =============================================================================
# 9. Dates spanning a year boundary
# =============================================================================
def test_9_year_boundary_spanning():
    chat = (
        "31/12/23, 23:50 - Alice: Countdown\n"
        "01/01/24, 00:05 - Bob: Happy New Year!\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="auto")

    # 31/12 is decisive DMY
    assert diag.confidence == DateOrderConfidence.DMY_CONFIDENT
    assert diag.resolved_order == "DMY"

    assert messages[0].timestamp == datetime(2023, 12, 31, 23, 50)
    assert messages[1].timestamp == datetime(2024, 1, 1, 0, 5)
    assert messages[1].timestamp - messages[0].timestamp == timedelta(minutes=15)


# =============================================================================
# 10. Two-digit and four-digit years
# =============================================================================
def test_10_two_digit_and_four_digit_years():
    chat_2digit = "15/05/24, 12:00 - Alice: 2-digit year\n"
    chat_4digit = "15/05/2024, 12:00 - Alice: 4-digit year\n"

    msg_2, _ = parse_whatsapp_text_with_diagnostics(chat_2digit)
    msg_4, _ = parse_whatsapp_text_with_diagnostics(chat_4digit)

    assert msg_2[0].timestamp == datetime(2024, 5, 15, 12, 0)
    assert msg_4[0].timestamp == datetime(2024, 5, 15, 12, 0)


# =============================================================================
# 11. 12-hour and 24-hour time formats
# =============================================================================
def test_11_time_formats_12h_24h_and_seconds():
    chat = (
        "15/05/24, 9:03 AM - Alice: 12-hour unpadded\n"
        "15/05/24, 09:03 PM - Bob: 12-hour padded PM\n"
        "15/05/24, 21:03 - Charlie: 24-hour\n"
        "15/05/24, 21:03:45 - Dave: 24-hour with seconds\n"
        "15/05/24, 09:03:45 AM - Eve: 12-hour with seconds\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat)
    assert diag.resolved_order == "DMY"

    assert messages[0].timestamp == datetime(2024, 5, 15, 9, 3)
    assert messages[1].timestamp == datetime(2024, 5, 15, 21, 3)
    assert messages[2].timestamp == datetime(2024, 5, 15, 21, 3)
    assert messages[3].timestamp == datetime(2024, 5, 15, 21, 3, 45)
    assert messages[4].timestamp == datetime(2024, 5, 15, 9, 3, 45)


# =============================================================================
# 12. Explicit date_order="DMY" override
# =============================================================================
def test_12_explicit_dmy_override():
    # Ambiguous chat
    chat = "05/04/24, 10:00 - Alice: Ambiguous date\n"
    messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="DMY")

    assert diag.resolved_order == "DMY"
    assert diag.override_applied is True
    # Under DMY: May 4, 2024
    assert messages[0].timestamp == datetime(2024, 4, 5, 10, 0)


# =============================================================================
# 13. Explicit date_order="MDY" override
# =============================================================================
def test_13_explicit_mdy_override():
    # Ambiguous chat
    chat = "05/04/24, 10:00 - Alice: Ambiguous date\n"
    messages, diag = parse_whatsapp_text_with_diagnostics(chat, date_order="MDY")

    assert diag.resolved_order == "MDY"
    assert diag.override_applied is True
    # Under MDY: May 4, 2024
    assert messages[0].timestamp == datetime(2024, 5, 4, 10, 0)


# =============================================================================
# 14. Invalid date tokens and malformed lines
# =============================================================================
def test_14_invalid_date_tokens_and_malformed_lines():
    chat = (
        "15/05/24, 10:00 - Alice: Valid message\n"
        "35/40/24, 11:00 - Bob: Impossible date numbers\n"
        "Not a timestamp at all line\n"
        "15/05/24, 12:00 - Alice: Another valid message\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat)

    assert diag.confidence == DateOrderConfidence.DMY_CONFIDENT
    assert diag.resolved_order == "DMY"
    # 3 messages parsed: message 2 has timestamp=None due to impossible date numbers,
    # and the non-timestamp line appends to message 2 as continuation text
    assert len(messages) == 3
    assert messages[0].timestamp == datetime(2024, 5, 15, 10, 0)
    assert messages[1].timestamp is None
    assert "Impossible date numbers\nNot a timestamp at all line" in messages[1].message
    assert messages[2].timestamp == datetime(2024, 5, 15, 12, 0)


# =============================================================================
# 15. Existing parser behavior for supported WhatsApp export formats
# =============================================================================
def test_15_supported_whatsapp_export_formats():
    chat = (
        "[15/05/24, 14:00:00] Alice: Bracketed timestamp with seconds\n"
        "15/05/24, 14:05 \u2013 Bob: Line with unicode en-dash\n"
        "15/05/24, 14:10 - Messages and calls are end-to-end encrypted.\n"
        "15/05/24, 14:15 - Alice: Multi-line message line 1\n"
        "Continuation line 2\n"
        "Continuation line 3\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(chat)

    assert len(messages) == 4
    # Message 1
    assert messages[0].sender == "Alice"
    assert messages[0].timestamp == datetime(2024, 5, 15, 14, 0, 0)
    # Message 2
    assert messages[1].sender == "Bob"
    # Message 3: System message
    assert messages[2].sender is None
    assert messages[2].message_type == "system"
    # Message 4: Multi-line
    assert messages[3].sender == "Alice"
    assert "line 1\nContinuation line 2\nContinuation line 3" in messages[3].message


# =============================================================================
# 16. Chronology Integrity Validation (Section 6)
# =============================================================================
def test_16_chronology_integrity_dmy_11_days_not_40_days():
    """
    Validates that a synthetic DMY export spanning May 4-15, 2024 correctly
    computes an 11-day elapsed time difference rather than 40 days.
    """
    synthetic_dmy_chat = (
        "04/05/24, 10:00 - Alice: Project roadmap initialized.\n"
        "08/05/24, 12:30 - Bob: Core architecture components deployed.\n"
        "15/05/24, 15:45 - Alice: Final release verification completed.\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(synthetic_dmy_chat)

    # 1. 15/05/24 provides decisive evidence for DMY
    assert diag.confidence == DateOrderConfidence.DMY_CONFIDENT
    assert diag.resolved_order == "DMY"

    # 2. 04/05/24 parses as May 4, 2024 (NOT April 5)
    t_start = messages[0].timestamp
    assert t_start == datetime(2024, 5, 4, 10, 0)

    # 3. 15/05/24 parses as May 15, 2024
    t_end = messages[2].timestamp
    assert t_end == datetime(2024, 5, 15, 15, 45)

    # 4. Elapsed difference is 11 days (plus 5h45m), NOT 40 days
    delta = t_end - t_start
    assert delta.days == 11
    assert delta == timedelta(days=11, hours=5, minutes=45)

    # 5. Original message ordering is preserved
    assert messages[0].sender == "Alice"
    assert messages[1].sender == "Bob"
    assert messages[2].sender == "Alice"


def test_16b_chronology_integrity_mdy():
    """Validates corresponding MDY chronology integrity with unambiguous dates."""
    synthetic_mdy_chat = (
        "05/04/24, 10:00 - Alice: Project roadmap initialized.\n"
        "05/08/24, 12:30 - Bob: Core architecture components deployed.\n"
        "05/15/24, 15:45 - Alice: Final release verification completed.\n"
    )
    messages, diag = parse_whatsapp_text_with_diagnostics(synthetic_mdy_chat)

    assert diag.confidence == DateOrderConfidence.MDY_CONFIDENT
    assert diag.resolved_order == "MDY"

    t_start = messages[0].timestamp
    t_end = messages[2].timestamp

    assert t_start == datetime(2024, 5, 4, 10, 0)
    assert t_end == datetime(2024, 5, 15, 15, 45)

    delta = t_end - t_start
    assert delta.days == 11
    assert delta == timedelta(days=11, hours=5, minutes=45)
