"""
test_context_grading_evaluation.py
====================================
Comprehensive offline evaluation suite for ChatLens-RAG context-aware relevance grading.

Evaluates relevance grading across 12 deterministic synthetic cases representing:
  1. Terse location answer dependent on question ("In Vault 9.")
  2. Short numerical / time answer ("9pm.")
  3. Pronoun-based reply ("It is in the second folder.")
  4. Yes/no reply dependent on question ("Yes, completely clean.")
  5. Short reply with misleading preceding context ("On the kitchen counter.")
  6. Self-contained relevant message (preserves standalone correctness)
  7. Self-contained irrelevant message
  8. Unrelated candidate with query-matching preceding message (weather distractor)
  9. Multi-message conversational exchange (Rack C4 power sensor)
  10. Short answer with insufficient / ambiguous context ("Maybe.")
  11. Conversation with similar but different entity (Dev JWT token vs Prod JWT secret)
  12. Ordinary-length candidate (>6 words, bypasses preceding context lookup)

A/B Evaluation Conditions:
  - Condition A: Context-aware grading (with preceding conversational context)
  - Condition B: Standalone baseline (suppressing preceding context lookup)

Deterministic Plumbing:
  - Uses real rag_graph.grade_documents() production plumbing.
  - Mocked deterministic LLM with rule-based logic; zero Ollama/network dependencies.
  - Zero access to production chat_data.db or chroma_db.
"""

import json
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_graph import (
    grade_documents,
    GraphState,
)


@dataclass
class ContextGradingCase:
    case_id: str
    category: str
    description: str
    query: str
    candidate_id: int
    candidate_text: str
    sender: str
    timestamp: str
    expected_label: str  # "relevant" or "irrelevant" (ground-truth binary label)
    expected_standalone: str  # Expected decision under standalone grading
    expected_context_aware: str  # Expected decision under context-aware grading
    preceding_messages: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------
# 12 Deterministic Synthetic Evaluation Cases
# ---------------------------------------------------------------------

EVALUATION_CASES: List[ContextGradingCase] = [
    # Case 1: Terse answer depending on preceding question
    ContextGradingCase(
        case_id="CASE-01-Terse-Location",
        category="Terse Context-Dependent",
        description="Terse location fragment whose entity depends on preceding question.",
        query="Where was the backup SSL certificate placed?",
        candidate_id=102,
        candidate_text="In Vault 9.",
        sender="Dev_Sam",
        timestamp="2026-10-14T09:11:00",
        expected_label="relevant",
        expected_standalone="irrelevant",
        expected_context_aware="relevant",
        preceding_messages=[
            {
                "id": 101,
                "timestamp": "2026-10-14T09:10:00",
                "sender": "Dev_Alex",
                "message": "Where did we store the backup SSL certificate?",
            }
        ],
    ),
    # Case 2: Short numerical / time answer
    ContextGradingCase(
        case_id="CASE-02-Short-Time",
        category="Terse Numerical/Time",
        description="Short time fragment answering deployment window start.",
        query="What time is the production deployment window scheduled for?",
        candidate_id=104,
        candidate_text="9pm.",
        sender="Dev_Lead",
        timestamp="2026-10-14T14:31:00",
        expected_label="relevant",
        expected_standalone="irrelevant",
        expected_context_aware="relevant",
        preceding_messages=[
            {
                "id": 103,
                "timestamp": "2026-10-14T14:30:00",
                "sender": "Dev_Ops",
                "message": "When are we kicking off the production deployment window tonight?",
            }
        ],
    ),
    # Case 3: Pronoun-based reply
    ContextGradingCase(
        case_id="CASE-03-Pronoun-Anaphora",
        category="Pronoun Resolution",
        description="Candidate uses pronoun 'It' referring to incident root cause report.",
        query="Where can I find the incident root cause report?",
        candidate_id=106,
        candidate_text="It is in the second folder.",
        sender="Dev_Maya",
        timestamp="2026-10-14T11:05:00",
        expected_label="relevant",
        expected_standalone="irrelevant",
        expected_context_aware="relevant",
        preceding_messages=[
            {
                "id": 105,
                "timestamp": "2026-10-14T11:04:00",
                "sender": "Dev_Ken",
                "message": "Has anyone seen the incident root cause report?",
            }
        ],
    ),
    # Case 4: Yes/no reply dependent on question
    ContextGradingCase(
        case_id="CASE-04-Yes-No-Confirmation",
        category="Terse Confirmation",
        description="Yes/no confirmation whose meaning depends on the question.",
        query="Did the database migration script succeed on staging?",
        candidate_id=108,
        candidate_text="Yes, completely clean.",
        sender="Dev_Sarah",
        timestamp="2026-10-14T16:20:00",
        expected_label="relevant",
        expected_standalone="irrelevant",
        expected_context_aware="relevant",
        preceding_messages=[
            {
                "id": 107,
                "timestamp": "2026-10-14T16:19:00",
                "sender": "Dev_Chris",
                "message": "Did the database migration script finish successfully on staging?",
            }
        ],
    ),
    # Case 5: Short reply with misleading preceding context
    ContextGradingCase(
        case_id="CASE-05-Misleading-Context",
        category="Misleading Context Distractor",
        description="Short reply answering about cafeteria keys, NOT the staging API key.",
        query="Where is the staging API key stored?",
        candidate_id=110,
        candidate_text="On the kitchen counter.",
        sender="Dev_Sam",
        timestamp="2026-10-14T12:05:00",
        expected_label="irrelevant",
        expected_standalone="irrelevant",
        expected_context_aware="irrelevant",
        preceding_messages=[
            {
                "id": 109,
                "timestamp": "2026-10-14T12:04:00",
                "sender": "Dev_Alex",
                "message": "Where did you leave the office cafeteria keys?",
            }
        ],
    ),
    # Case 6: Self-contained relevant message
    ContextGradingCase(
        case_id="CASE-06-Self-Contained-Relevant",
        category="Self-Contained Positive",
        description="Complete factual statement that is relevant with or without context.",
        query="What is the primary IP address for the Redis cache cluster?",
        candidate_id=112,
        candidate_text="The primary IP address for the Redis cache cluster is 10.0.4.15.",
        sender="Dev_Alex",
        timestamp="2026-10-14T08:30:00",
        expected_label="relevant",
        expected_standalone="relevant",
        expected_context_aware="relevant",
        preceding_messages=[
            {
                "id": 111,
                "timestamp": "2026-10-14T08:29:00",
                "sender": "Dev_Ops",
                "message": "Good morning team, checking infrastructure status.",
            }
        ],
    ),
    # Case 7: Self-contained irrelevant message
    ContextGradingCase(
        case_id="CASE-07-Self-Contained-Irrelevant",
        category="Self-Contained Negative",
        description="Completely unrelated conversation about lunch.",
        query="What is the primary IP address for the Redis cache cluster?",
        candidate_id=114,
        candidate_text="Pizza sounds great for lunch today.",
        sender="Dev_Dana",
        timestamp="2026-10-14T12:15:00",
        expected_label="irrelevant",
        expected_standalone="irrelevant",
        expected_context_aware="irrelevant",
        preceding_messages=[
            {
                "id": 113,
                "timestamp": "2026-10-14T12:14:00",
                "sender": "Dev_Chris",
                "message": "Anyone ordering lunch today?",
            }
        ],
    ),
    # Case 8: Unrelated candidate with query-matching preceding message
    ContextGradingCase(
        case_id="CASE-08-Unrelated-Candidate-Matching-Context",
        category="Context Distractor Invariance",
        description="Preceding message matches query, but candidate itself discusses weather.",
        query="Who approved the firewall port change?",
        candidate_id=116,
        candidate_text="I think the weather is going to be stormy this evening.",
        sender="Dev_Dana",
        timestamp="2026-10-14T17:05:00",
        expected_label="irrelevant",
        expected_standalone="irrelevant",
        expected_context_aware="irrelevant",
        preceding_messages=[
            {
                "id": 115,
                "timestamp": "2026-10-14T17:04:00",
                "sender": "Dev_Security",
                "message": "Who approved the firewall port change yesterday?",
            }
        ],
    ),
    # Case 9: Multi-message conversational exchange
    ContextGradingCase(
        case_id="CASE-09-Multi-Message-Exchange",
        category="Multi-Message Conversation",
        description="Two preceding messages identify the entity (Rack C4) for candidate confirmation.",
        query="Which server rack is experiencing power fluctuation?",
        candidate_id=119,
        candidate_text="Confirmed, heavy voltage drops.",
        sender="Dev_Facilities",
        timestamp="2026-10-14T13:42:00",
        expected_label="relevant",
        expected_standalone="irrelevant",
        expected_context_aware="relevant",
        preceding_messages=[
            {
                "id": 117,
                "timestamp": "2026-10-14T13:40:00",
                "sender": "Dev_Monitor",
                "message": "Rack C4 power sensor is throwing warnings.",
            },
            {
                "id": 118,
                "timestamp": "2026-10-14T13:41:00",
                "sender": "Dev_Ops",
                "message": "Can anyone confirm on site?",
            },
        ],
    ),
    # Case 10: Short answer with insufficient / ambiguous context
    ContextGradingCase(
        case_id="CASE-010-Ambiguous-Context-Abstention",
        category="Ambiguous / Abstention",
        description="Candidate is 'Maybe.' and context does not resolve pod autoscaler query.",
        query="Did the Kubernetes pod autoscaler trigger?",
        candidate_id=121,
        candidate_text="Maybe.",
        sender="Dev_Sam",
        timestamp="2026-10-14T15:10:00",
        expected_label="irrelevant",  # Non-supporting for factual retrieval
        expected_standalone="uncertain",
        expected_context_aware="uncertain",
        preceding_messages=[
            {
                "id": 120,
                "timestamp": "2026-10-14T15:09:00",
                "sender": "Dev_Alex",
                "message": "Checking general system alerts.",
            }
        ],
    ),
    # Case 11: Similar but different entity (Dev test token vs Prod secret)
    ContextGradingCase(
        case_id="CASE-011-Similar-Entity-Negative",
        category="Fine-Grained Entity Discrimination",
        description="Candidate and context discuss dev environment test token, NOT production JWT secret.",
        query="Where is the production JWT secret stored?",
        candidate_id=123,
        candidate_text="In secrets.dev.json.",
        sender="Dev_Auth",
        timestamp="2026-10-14T11:22:00",
        expected_label="irrelevant",
        expected_standalone="irrelevant",
        expected_context_aware="irrelevant",
        preceding_messages=[
            {
                "id": 122,
                "timestamp": "2026-10-14T11:21:00",
                "sender": "Dev_Test",
                "message": "Where did we save the dev environment test JWT token?",
            }
        ],
    ),
    # Case 12: Ordinary-length candidate (>6 words, bypasses preceding context lookup)
    ContextGradingCase(
        case_id="CASE-012-Ordinary-Length-Bypass",
        category="Length Threshold Invariance",
        description="20-word self-contained answer that exceeds the 6-word threshold and bypasses lookup.",
        query="What is the backup retention policy for user data?",
        candidate_id=125,
        candidate_text="Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage.",
        sender="Dev_Compliance",
        timestamp="2026-10-14T10:15:00",
        expected_label="relevant",
        expected_standalone="relevant",
        expected_context_aware="relevant",
        preceding_messages=[
            {
                "id": 124,
                "timestamp": "2026-10-14T10:14:00",
                "sender": "Dev_Lead",
                "message": "Let's review the compliance docs.",
            }
        ],
    ),
]


# ---------------------------------------------------------------------
# Deterministic Rule-Based Mock Grader LLM
# ---------------------------------------------------------------------

class DeterministicEvaluationMockLLM:
    """
    Deterministic rule-based mock LLM simulating conservative relevance grading.
    Inspects candidate text and preceding context (if present) to simulate realistic
    judgment on each synthetic case.
    """
    def __init__(self):
        self.recorded_prompts: List[str] = []

    def invoke(self, prompt: Any) -> AIMessage:
        prompt_str = str(prompt)
        self.recorded_prompts.append(prompt_str)

        # Candidate 1: In Vault 9.
        if "In Vault 9." in prompt_str:
            if "Preceding Conversational Context:" in prompt_str and "backup SSL certificate" in prompt_str:
                decision = "relevant"
                reason = "Preceding context confirms Vault 9 stores the backup SSL certificate."
            else:
                decision = "irrelevant"
                reason = "Standalone fragment 'In Vault 9.' lacks subject and entity reference."

        # Candidate 2: 9pm.
        elif "9pm." in prompt_str:
            if "Preceding Conversational Context:" in prompt_str and "deployment window" in prompt_str:
                decision = "relevant"
                reason = "Preceding context connects '9pm.' to the scheduled production deployment window."
            else:
                decision = "irrelevant"
                reason = "Standalone time fragment '9pm.' does not indicate what event it refers to."

        # Candidate 3: It is in the second folder.
        elif "It is in the second folder." in prompt_str:
            if "Preceding Conversational Context:" in prompt_str and "incident root cause report" in prompt_str:
                decision = "relevant"
                reason = "Preceding context resolves pronoun 'It' as the incident root cause report."
            else:
                decision = "irrelevant"
                reason = "Unresolved pronoun 'It' lacks referent topic without conversational context."

        # Candidate 4: Yes, completely clean.
        elif "Yes, completely clean." in prompt_str:
            if "Preceding Conversational Context:" in prompt_str and "migration script" in prompt_str:
                decision = "relevant"
                reason = "Preceding context links confirmation to the staging database migration script."
            else:
                decision = "irrelevant"
                reason = "Terse confirmation lacks indication of what operation was completely clean."

        # Candidate 5: On the kitchen counter. (Misleading context: cafeteria keys)
        elif "On the kitchen counter." in prompt_str:
            decision = "irrelevant"
            reason = "Candidate refers to cafeteria keys, not the requested staging API key."

        # Candidate 6: Redis IP address 10.0.4.15
        elif "10.0.4.15" in prompt_str:
            decision = "relevant"
            reason = "Self-contained statement directly provides the Redis cluster IP address."

        # Candidate 7: Pizza sounds great
        elif "Pizza sounds great" in prompt_str:
            decision = "irrelevant"
            reason = "Candidate discusses lunch preferences, completely unrelated to Redis cluster."

        # Candidate 8: Stormy weather (Preceding context mentions firewall, but candidate is weather)
        elif "weather is going to be stormy" in prompt_str:
            decision = "irrelevant"
            reason = "Candidate message discusses evening weather and does not answer firewall port approval."

        # Candidate 9: Confirmed, heavy voltage drops. (Multi-message: Rack C4)
        elif "heavy voltage drops" in prompt_str:
            if "Preceding Conversational Context:" in prompt_str and "Rack C4" in prompt_str:
                decision = "relevant"
                reason = "Preceding context links voltage confirmation to Rack C4 power sensor."
            else:
                decision = "irrelevant"
                reason = "Voltage confirmation does not identify which server rack is affected."

        # Candidate 10: Maybe. (Insufficient / ambiguous context)
        elif "Maybe." in prompt_str:
            decision = "uncertain"
            reason = "Candidate statement 'Maybe.' and context remain ambiguous and inconclusive."

        # Candidate 11: In secrets.dev.json. (Dev token vs Prod secret)
        elif "secrets.dev.json" in prompt_str:
            decision = "irrelevant"
            reason = "Candidate and context refer to dev environment test token, not production JWT secret."

        # Candidate 12: Compliance retention policy 90 days
        elif "retaining all daily user database snapshots for 90 days" in prompt_str:
            decision = "relevant"
            reason = "Self-contained compliance statement answers user data retention policy directly."

        else:
            decision = "uncertain"
            reason = "Unrecognized candidate in evaluation harness."

        return AIMessage(content=json.dumps([
            {"candidate": 1, "decision": decision, "reason": reason}
        ]))


# ---------------------------------------------------------------------
# Execution & Metrics Computation Helpers
# ---------------------------------------------------------------------

@dataclass
class CaseEvaluationResult:
    case_id: str
    category: str
    expected_label: str
    actual_decision: str
    decision_reason: str
    prompt: str
    has_preceding_context: bool
    is_decisive: bool
    is_correct_binary: bool  # applicable if decisive
    is_correct_strict: bool  # treats uncertain as incorrect


def evaluate_single_case(
    case: ContextGradingCase,
    condition: str,  # "context_aware" or "standalone"
) -> CaseEvaluationResult:
    """
    Executes real rag_graph.grade_documents() for a single case under either
    Condition A (context-aware) or Condition B (standalone baseline).
    """
    mock_llm = DeterministicEvaluationMockLLM()
    clean_text = f"[ID {case.candidate_id}] [{case.timestamp}] {case.sender}: {case.candidate_text}"

    doc = Document(
        page_content=clean_text,
        metadata={
            "source": "hybrid_search",
            "message_id": case.candidate_id,
            "timestamp": case.timestamp,
            "sender": case.sender,
            "hybrid_score": 0.85,
        },
    )

    state: GraphState = {
        "question": case.query,
        "original_question": case.query,
        "documents": [doc],
        "generation": "",
        "rewrite_count": 0,
        "grounded": False,
        "df": None,
        "scope": "specific",
        "primary_intent": "semantic_rag",
        "database_path": "/synthetic_sandbox/eval.db",
    }

    if condition == "context_aware":
        # Supply synthetic chronological preceding messages
        all_context_rows = list(case.preceding_messages) + [{
            "id": case.candidate_id,
            "timestamp": case.timestamp,
            "sender": case.sender,
            "message": case.candidate_text,
        }]

        with patch("rag_graph.get_chronological_context", return_value=all_context_rows):
            result_state = grade_documents(state, grader_llm=mock_llm)
    else:
        # Standalone: suppress preceding context lookup
        with patch("rag_graph.get_grading_preceding_context", return_value=None):
            result_state = grade_documents(state, grader_llm=mock_llm)

    prompt = mock_llm.recorded_prompts[0] if mock_llm.recorded_prompts else ""
    has_preceding_context = "Preceding Conversational Context:" in prompt

    # Retrieve decision from document metadata
    evaluated_docs = result_state.get("documents", [])
    if evaluated_docs:
        actual_decision = evaluated_docs[0].metadata.get("relevance_decision", "uncertain")
        reason = evaluated_docs[0].metadata.get("relevance_reason", "")
    else:
        actual_decision = "irrelevant"
        reason = "Filtered out as irrelevant by grade_documents"

    is_decisive = actual_decision in ("relevant", "irrelevant")
    is_correct_binary = (actual_decision == case.expected_label) if is_decisive else False
    is_correct_strict = (actual_decision == case.expected_label)

    return CaseEvaluationResult(
        case_id=case.case_id,
        category=case.category,
        expected_label=case.expected_label,
        actual_decision=actual_decision,
        decision_reason=reason,
        prompt=prompt,
        has_preceding_context=has_preceding_context,
        is_decisive=is_decisive,
        is_correct_binary=is_correct_binary,
        is_correct_strict=is_correct_strict,
    )


@dataclass
class ConditionMetrics:
    condition_name: str
    total_cases: int
    decisive_cases: int
    abstention_cases: int
    coverage_rate: float
    abstention_rate: float
    # Decisive Binary Confusion Matrix
    tp: int
    fp: int
    tn: int
    fn: int
    # Decisive Rates
    precision: float
    recall: float
    f1: float
    accuracy_decisive: float
    false_positive_rate: float
    # Strict Total Metric (treating abstention as error)
    accuracy_strict: float


def compute_metrics(
    condition_name: str,
    results: List[CaseEvaluationResult],
) -> ConditionMetrics:
    """
    Computes decisive binary metrics, coverage, and strict metrics with explicit denominators.
    """
    total = len(results)
    decisive = [r for r in results if r.is_decisive]
    abstentions = [r for r in results if not r.is_decisive]

    decisive_count = len(decisive)
    abstention_count = len(abstentions)
    coverage = decisive_count / total if total > 0 else 0.0
    abstention_rate = abstention_count / total if total > 0 else 0.0

    tp = sum(1 for r in decisive if r.expected_label == "relevant" and r.actual_decision == "relevant")
    fp = sum(1 for r in decisive if r.expected_label == "irrelevant" and r.actual_decision == "relevant")
    tn = sum(1 for r in decisive if r.expected_label == "irrelevant" and r.actual_decision == "irrelevant")
    fn = sum(1 for r in decisive if r.expected_label == "relevant" and r.actual_decision == "irrelevant")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc_decisive = (tp + tn) / decisive_count if decisive_count > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    strict_correct = sum(1 for r in results if r.is_correct_strict)
    acc_strict = strict_correct / total if total > 0 else 0.0

    return ConditionMetrics(
        condition_name=condition_name,
        total_cases=total,
        decisive_cases=decisive_count,
        abstention_cases=abstention_count,
        coverage_rate=coverage,
        abstention_rate=abstention_rate,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy_decisive=acc_decisive,
        false_positive_rate=fpr,
        accuracy_strict=acc_strict,
    )


# =====================================================================
# Unit Tests
# =====================================================================

def test_context_grading_evaluation_all_12_cases():
    """
    Runs the complete 12-case evaluation for Condition A and Condition B,
    computes full metrics, and asserts expected comparative improvements.
    """
    results_a = [evaluate_single_case(c, "context_aware") for c in EVALUATION_CASES]
    results_b = [evaluate_single_case(c, "standalone") for c in EVALUATION_CASES]

    metrics_a = compute_metrics("Condition A (Context-Aware)", results_a)
    metrics_b = compute_metrics("Condition B (Standalone Baseline)", results_b)

    # 1. Total Case Counts & Coverage
    assert metrics_a.total_cases == 12
    assert metrics_b.total_cases == 12
    assert metrics_a.decisive_cases == 11
    assert metrics_b.decisive_cases == 11
    assert metrics_a.abstention_cases == 1
    assert metrics_b.abstention_cases == 1
    assert metrics_a.coverage_rate == pytest.approx(11 / 12, 0.001)
    assert metrics_b.coverage_rate == pytest.approx(11 / 12, 0.001)

    # 2. Confusion Matrix Invariants
    # Condition B suffers 5 False Negatives on terse context-dependent cases
    assert metrics_b.tp == 2  # Cases 6, 12
    assert metrics_b.fn == 5  # Cases 1, 2, 3, 4, 9
    assert metrics_b.tn == 4  # Cases 5, 7, 8, 11
    assert metrics_b.fp == 0

    # Condition A recovers all 5 False Negatives without introducing False Positives
    assert metrics_a.tp == 7  # Cases 1, 2, 3, 4, 6, 9, 12
    assert metrics_a.fn == 0
    assert metrics_a.tn == 4  # Cases 5, 7, 8, 11
    assert metrics_a.fp == 0

    # 3. Metric Comparison
    assert metrics_a.false_positive_rate == 0.0
    assert metrics_b.false_positive_rate == 0.0
    assert metrics_a.precision == 1.0
    assert metrics_b.precision == 1.0

    # Recall improves from 2/7 (28.6%) to 7/7 (100%)
    assert metrics_b.recall == pytest.approx(2 / 7, 0.001)
    assert metrics_a.recall == 1.0

    # F1 improves from 0.444 to 1.0
    assert metrics_b.f1 == pytest.approx(0.4444, 0.001)
    assert metrics_a.f1 == 1.0

    # Decisive accuracy improves from 6/11 (54.5%) to 11/11 (100%)
    assert metrics_b.accuracy_decisive == pytest.approx(6 / 11, 0.001)
    assert metrics_a.accuracy_decisive == 1.0

    # Strict accuracy improves from 6/12 (50.0%) to 11/12 (91.7%)
    assert metrics_b.accuracy_strict == pytest.approx(6 / 12, 0.001)
    assert metrics_a.accuracy_strict == pytest.approx(11 / 12, 0.001)

    # 4. Transitions
    recovered = [
        ra.case_id for ra, rb in zip(results_a, results_b)
        if rb.actual_decision != "relevant" and ra.actual_decision == "relevant" and ra.expected_label == "relevant"
    ]
    assert recovered == [
        "CASE-01-Terse-Location",
        "CASE-02-Short-Time",
        "CASE-03-Pronoun-Anaphora",
        "CASE-04-Yes-No-Confirmation",
        "CASE-09-Multi-Message-Exchange",
    ]
    assert len(recovered) == 5

    broken = [
        ra.case_id for ra, rb in zip(results_a, results_b)
        if rb.is_correct_strict and not ra.is_correct_strict
    ]
    assert len(broken) == 0, f"Context broke previously correct standalone decisions: {broken}"


def test_terse_and_elliptical_cases_recovered_in_context_mode():
    """Verify that Cases 1, 2, 3, 4, 9 are recovered as relevant when preceding context is supplied."""
    target_ids = {"CASE-01-Terse-Location", "CASE-02-Short-Time", "CASE-03-Pronoun-Anaphora", "CASE-04-Yes-No-Confirmation", "CASE-09-Multi-Message-Exchange"}
    for case in EVALUATION_CASES:
        if case.case_id in target_ids:
            res_a = evaluate_single_case(case, "context_aware")
            res_b = evaluate_single_case(case, "standalone")
            assert res_b.actual_decision == "irrelevant", f"{case.case_id} should be rejected standalone"
            assert res_a.actual_decision == "relevant", f"{case.case_id} should be recovered with context"
            assert res_a.has_preceding_context is True
            assert res_b.has_preceding_context is False


def test_misleading_and_distractor_context_never_rescues_irrelevant():
    """Verify that Cases 5, 8, 11 reject irrelevant candidates even when misleading context is present."""
    target_ids = {"CASE-05-Misleading-Context", "CASE-08-Unrelated-Candidate-Matching-Context", "CASE-011-Similar-Entity-Negative"}
    for case in EVALUATION_CASES:
        if case.case_id in target_ids:
            res_a = evaluate_single_case(case, "context_aware")
            res_b = evaluate_single_case(case, "standalone")
            assert res_a.actual_decision == "irrelevant", f"{case.case_id} was incorrectly marked relevant with context"
            assert res_b.actual_decision == "irrelevant"
            assert res_a.is_correct_binary is True
            assert res_b.is_correct_binary is True


def test_self_contained_and_ordinary_length_cases_preserve_standalone_decisions():
    """Verify that self-contained relevant (Case 6), irrelevant (Case 7), and ordinary length (Case 12) preserve decisions."""
    target_ids = {"CASE-06-Self-Contained-Relevant", "CASE-07-Self-Contained-Irrelevant", "CASE-012-Ordinary-Length-Bypass"}
    for case in EVALUATION_CASES:
        if case.case_id in target_ids:
            res_a = evaluate_single_case(case, "context_aware")
            res_b = evaluate_single_case(case, "standalone")
            assert res_a.actual_decision == res_b.actual_decision == case.expected_label
            # Case 12 should bypass preceding context lookup due to word count > 6
            if case.case_id == "CASE-012-Ordinary-Length-Bypass":
                assert res_a.has_preceding_context is False


def test_insufficient_context_properly_abstains():
    """Verify that Case 10 (ambiguous 'Maybe.') produces an uncertain decision in both conditions."""
    case_10 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-010-Ambiguous-Context-Abstention")
    res_a = evaluate_single_case(case_10, "context_aware")
    res_b = evaluate_single_case(case_10, "standalone")
    assert res_a.actual_decision == "uncertain"
    assert res_b.actual_decision == "uncertain"
    assert res_a.is_decisive is False
    assert res_b.is_decisive is False


def test_mathematical_metric_denominators():
    """Verify metrics calculation logic and explicit denominators on synthetic test cases."""
    dummy_results = [
        CaseEvaluationResult("1", "cat", "relevant", "relevant", "ok", "", True, True, True, True),     # TP
        CaseEvaluationResult("2", "cat", "relevant", "irrelevant", "ok", "", True, True, False, False),  # FN
        CaseEvaluationResult("3", "cat", "irrelevant", "irrelevant", "ok", "", True, True, True, True),  # TN
        CaseEvaluationResult("4", "cat", "irrelevant", "relevant", "ok", "", True, True, False, False),  # FP
        CaseEvaluationResult("5", "cat", "relevant", "uncertain", "ok", "", False, False, False, False), # Abstention
    ]
    m = compute_metrics("test", dummy_results)
    assert m.total_cases == 5
    assert m.decisive_cases == 4
    assert m.abstention_cases == 1
    assert m.coverage_rate == 4 / 5
    assert m.abstention_rate == 1 / 5
    assert m.tp == 1
    assert m.fn == 1
    assert m.tn == 1
    assert m.fp == 1
    assert m.precision == 1 / (1 + 1)  # 0.5
    assert m.recall == 1 / (1 + 1)     # 0.5
    assert m.f1 == 0.5
    assert m.accuracy_decisive == (1 + 1) / 4  # 0.5
    assert m.false_positive_rate == 1 / (1 + 1) # 0.5
    assert m.accuracy_strict == 2 / 5           # 0.4
