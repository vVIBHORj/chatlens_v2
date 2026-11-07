"""
test_evidence_scope_guard.py — Task 69 Adversarial Deterministic Test Suite

Comprehensive tests for the 20 required adversarial scenarios:
1. Exact query echoed as a question.
2. Query paraphrased as a question.
3. Question followed by a direct factual answer.
4. Question followed by an unrelated statement.
5. Short answer with no question mark.
6. Multi-message dialogue containing both question and answer.
7. Production query with development-only evidence.
8. Production query with staging-only evidence.
9. Production query with explicit production evidence.
10. Candidate mentioning both dev and production.
11. Environment mismatch expressed in a filename.
12. Environment mismatch expressed in natural language.
13. Ambiguous environment wording.
14. Query with no explicit environment.
15. Candidate with no explicit environment.
16. Joint-support evidence from CASE-09.
17. Terse answers from CASE-01 and CASE-02.
18. CASE-04 qualifier-bearing answer.
19. Empty candidate text.
20. Malformed metadata.

Runs in <1 second with zero external dependencies (no Ollama, no DB, no network).
"""

import pytest
from evidence_scope_guard import (
    GuardDecision,
    GuardResult,
    evaluate_evidence_scope,
    extract_environments,
    extract_content_keywords,
    is_interrogative_message,
    is_plausible_answering_reply,
)


def test_01_exact_query_echoed_as_question():
    """Scenario 1: Exact query echoed as a question."""
    query = "Who approved the firewall port change?"
    candidate = "[ID 115] [2026-10-14 05:04 PM] Dev_Security: Who approved the firewall port change?"
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_QUERY_ECHO
    assert "rule_query_echo" in res.rule


def test_02_query_paraphrased_as_question():
    """Scenario 2: Query paraphrased as a question."""
    query = "Who approved the firewall port change?"
    candidate = "[ID 115] [2026-10-14 05:04 PM] Dev_Security: Does anyone know who approved the firewall port change yesterday?"
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_QUERY_ECHO


def test_03_question_followed_by_direct_factual_answer():
    """Scenario 3: Question followed by a direct factual answer."""
    query = "Where was the backup SSL certificate placed?"
    candidate = (
        "[ID 101] [2026-10-14 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?\n"
        "[ID 102] [2026-10-14 09:11 AM] Dev_Sam: In Vault 9."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS
    assert res.rule == "rule_query_echo_with_answer"


def test_04_question_followed_by_unrelated_statement():
    """Scenario 4: Question followed by an unrelated statement."""
    query = "Who approved the firewall port change?"
    candidate = (
        "[ID 115] [2026-10-14 05:04 PM] Dev_Security: Who approved the firewall port change yesterday?\n"
        "[ID 116] [2026-10-14 05:05 PM] Dev_Dana: I think the weather is going to be stormy this evening."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_QUERY_ECHO
    assert res.rule == "rule_query_echo_unanswered_banter"


def test_05_short_answer_with_no_question_mark():
    """Scenario 5: Short answer with no question mark."""
    query = "What time is the production deployment window scheduled for?"
    candidate = "[ID 104] [2026-10-14 02:31 PM] Dev_Lead: 9pm."
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_06_multimessage_dialogue_containing_both_question_and_answer():
    """Scenario 6: Multi-message dialogue containing both question and answer."""
    query = "Did the database migration script succeed on staging?"
    candidate = (
        "[CHRONOLOGICAL CONTEXT]\n"
        "[ID 0] [14/10/2026 08:50 AM] Dev_Security: Checking port access and daily operational status.\n"
        "[ID 107] [14/10/2026 04:19 PM] Dev_Chris: Did the database migration script finish successfully on staging?\n"
        "[ID 108] [14/10/2026 04:20 PM] Dev_Sarah: Yes, completely clean."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_07_production_query_with_development_only_evidence():
    """Scenario 7: Production query with development-only evidence."""
    query = "Where is the production JWT secret stored?"
    candidate = (
        "[ID 122] Dev_Test: Where did we save the dev environment test JWT token?\n"
        "[ID 123] Dev_Auth: In secrets.dev.json."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_ENVIRONMENT_MISMATCH
    assert res.rule == "rule_env_mismatch_prod_vs_nonprod"


def test_08_production_query_with_staging_only_evidence():
    """Scenario 8: Production query with staging-only evidence."""
    query = "Where is the production database configuration located?"
    candidate = "[ID 201] Dev_Ops: The database configuration is in staging.env."
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_ENVIRONMENT_MISMATCH
    assert res.rule == "rule_env_mismatch_prod_vs_nonprod"


def test_09_production_query_with_explicit_production_evidence():
    """Scenario 9: Production query with explicit production evidence."""
    query = "What time is the production deployment window scheduled for?"
    candidate = (
        "[ID 103] Dev_Ops: When are we kicking off the production deployment window tonight?\n"
        "[ID 104] Dev_Lead: 9pm."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_10_candidate_mentioning_both_dev_and_production():
    """Scenario 10: Candidate mentioning both dev and production."""
    query = "Where is the production JWT secret stored?"
    candidate = (
        "[ID 250] Dev_Sec: We moved the keys from dev to production, now in production.yaml."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_11_environment_mismatch_expressed_in_filename():
    """Scenario 11: Environment mismatch expressed in a filename."""
    query = "Where is the production database credentials file?"
    candidate = "[ID 123] Dev_Auth: In secrets.dev.json."
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_ENVIRONMENT_MISMATCH


def test_12_environment_mismatch_expressed_in_natural_language():
    """Scenario 12: Environment mismatch expressed in natural language."""
    query = "What is the production database host address?"
    candidate = "[ID 130] Dev_Cloud: The development cluster host is 10.0.1.5."
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_ENVIRONMENT_MISMATCH


def test_13_ambiguous_environment_wording():
    """Scenario 13: Ambiguous environment wording."""
    query = "Where is the production API key stored?"
    candidate = "[ID 140] Dev_Lead: We are migrating keys between dev and production environments."
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.NEEDS_REVIEW
    assert res.rule == "rule_env_ambiguous_wording"


def test_14_query_with_no_explicit_environment():
    """Scenario 14: Query with no explicit environment."""
    query = "Where was the backup SSL certificate placed?"
    candidate = "[ID 102] Dev_Sam: In Vault 9."
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_15_candidate_with_no_explicit_environment():
    """Scenario 15: Candidate with no explicit environment."""
    query = "What time is the production deployment window scheduled for?"
    candidate = "[ID 104] Dev_Lead: 9pm."
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_16_joint_support_evidence_from_case09():
    """Scenario 16: Joint-support evidence from CASE-09."""
    query = "Which server rack is experiencing power fluctuation?"
    candidate = (
        "[CHRONOLOGICAL CONTEXT]\n"
        "[ID 117] [14/10/2026 01:40 PM] Dev_Monitor: Rack C4 power sensor is throwing warnings.\n"
        "[ID 118] [14/10/2026 01:41 PM] Dev_Ops: Can anyone confirm on site?\n"
        "[ID 119] [14/10/2026 01:42 PM] Dev_Facilities: Confirmed, heavy voltage drops."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_17_terse_answers_from_case01_and_case02():
    """Scenario 17: Terse answers from CASE-01 and CASE-02."""
    # CASE-01
    q1 = "Where was the backup SSL certificate placed?"
    c1 = "[ID 102] Dev_Sam: In Vault 9."
    assert evaluate_evidence_scope(q1, c1).decision == GuardDecision.PASS

    # CASE-02
    q2 = "What time is the production deployment window scheduled for?"
    c2 = "[ID 104] Dev_Lead: 9pm."
    assert evaluate_evidence_scope(q2, c2).decision == GuardDecision.PASS


def test_18_case04_qualifier_bearing_answer():
    """Scenario 18: CASE-04 qualifier-bearing answer."""
    query = "Did the database migration script succeed on staging?"
    candidate = (
        "[ID 107] Dev_Chris: Did the database migration script finish successfully on staging?\n"
        "[ID 108] Dev_Sarah: Yes, completely clean."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS


def test_19_empty_candidate_text():
    """Scenario 19: Empty candidate text."""
    query = "Where was the backup SSL certificate placed?"
    assert evaluate_evidence_scope(query, "").decision == GuardDecision.PASS
    assert evaluate_evidence_scope(query, "   ").decision == GuardDecision.PASS


def test_20_malformed_metadata():
    """Scenario 20: Malformed metadata."""
    query = "Where was the backup SSL certificate placed?"
    cand = "[ID 102] Dev_Sam: In Vault 9."
    # None, malformed list, int instead of dict
    assert evaluate_evidence_scope(query, cand, anchor_message_id="?", included_message_ids=None, metadata=None).decision == GuardDecision.PASS
    assert evaluate_evidence_scope(query, cand, anchor_message_id=None, included_message_ids="bad_type", metadata=123).decision == GuardDecision.PASS


# ============================================================================
# TASK 74 REGRESSION TESTS: PHASE 1 (ENVIRONMENT DETECTION AMBIGUITY)
# ============================================================================

def test_21_query_with_dev_role_does_not_infer_environment():
    """Phase 1: 'dev' or 'developers' referring to role must not trigger development environment mismatch."""
    q1 = "Which dev approved the release?"
    c1 = (
        "[RETRIEVED ANCHOR]\n"
        "[ID 10] [2026-10-14 10:00 AM] Alice: Production release was approved by Bob."
    )
    res1 = evaluate_evidence_scope(q1, c1)
    # Must NOT reject as environment mismatch
    assert res1.decision != GuardDecision.REJECT_ENVIRONMENT_MISMATCH
    assert "development" not in res1.details["query_envs"]

    q2 = "Which developers reviewed the production release?"
    res2 = evaluate_evidence_scope(q2, c1)
    assert res2.decision != GuardDecision.REJECT_ENVIRONMENT_MISMATCH
    assert "development" not in res2.details["query_envs"]
    assert "production" in res2.details["query_envs"]

    q3 = "Ask the lead dev about the status."
    c3 = "[ID 11] Charlie: Staging deploy is green."
    res3 = evaluate_evidence_scope(q3, c3)
    assert res3.decision != GuardDecision.REJECT_ENVIRONMENT_MISMATCH
    assert "development" not in res3.details["query_envs"]


def test_22_query_with_contextual_dev_environment_identified():
    """Phase 1: 'dev' accompanied by environment context correctly identifies development."""
    q1 = "Which secrets are in the dev environment?"
    envs1 = extract_environments(q1)
    assert "development" in envs1

    q2 = "Which branch contains the dev configuration?"
    envs2 = extract_environments(q2)
    assert "development" in envs2

    q3 = "Where is the dev cluster deployed?"
    envs3 = extract_environments(q3)
    assert "development" in envs3

    q4 = "Where is the development secret stored?"
    envs4 = extract_environments(q4)
    assert "development" in envs4

    # Development query against production-only candidate must still trigger mismatch
    c_prod = "[ID 20] Alice: The production secret is stored in Vault."
    res_mismatch = evaluate_evidence_scope(q1, c_prod)
    assert res_mismatch.decision == GuardDecision.REJECT_ENVIRONMENT_MISMATCH


# ============================================================================
# TASK 74 REGRESSION TESTS: PHASE 2 (STANDALONE RETRIEVED-ANCHOR PARSING)
# ============================================================================

def test_23_standalone_retrieved_anchor_question_echo_rejected():
    """Phase 2: Question-only candidate under [RETRIEVED ANCHOR] without context is rejected."""
    query = "Who approved the firewall port change?"
    candidate = (
        "[RETRIEVED ANCHOR]\n"
        "[ID 115] [2026-10-14 05:04 PM] Dev_Security: Who approved the firewall port change yesterday?"
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.REJECT_QUERY_ECHO
    assert "rule_query_echo" in res.rule
    assert res.details["has_query_echo"] is True


def test_24_standalone_retrieved_anchor_factual_answer_passed():
    """Phase 2: Factual answer under [RETRIEVED ANCHOR] without context passes guard."""
    query = "Where was the backup SSL certificate placed?"
    candidate = (
        "[RETRIEVED ANCHOR]\n"
        "[ID 102] [2026-10-14 09:11 AM] Dev_Sam: In Vault 9."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS
    assert res.rule == "rule_default_pass"


def test_25_standalone_retrieved_anchor_question_and_reply():
    """Phase 2: Multi-message dialogue under [RETRIEVED ANCHOR] passes guard."""
    query = "Where was the backup SSL certificate placed?"
    candidate = (
        "[RETRIEVED ANCHOR]\n"
        "[ID 101] [2026-10-14 09:10 AM] Alice: Where was the SSL certificate placed?\n"
        "[ID 102] [2026-10-14 09:11 AM] Bob: In Vault 9."
    )
    res = evaluate_evidence_scope(query, candidate)
    assert res.decision == GuardDecision.PASS
    assert res.details["has_answer_reply"] is True


def test_26_empty_and_malformed_candidate_parsing():
    """Phase 2: Empty and malformed candidate inputs parse safely."""
    from evidence_scope_guard import parse_candidate_messages
    assert parse_candidate_messages("") == []
    assert parse_candidate_messages("   \n\n  ") == []
    assert parse_candidate_messages("[RETRIEVED ANCHOR]\n") == []
    assert parse_candidate_messages("Random non-message header lines\n---") == [{'id': None, 'sender': None, 'text': 'Random non-message header lines', 'raw_line': 'Random non-message header lines'}]

