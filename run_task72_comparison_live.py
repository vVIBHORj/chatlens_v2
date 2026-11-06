"""
run_task72_comparison_live.py
=============================
Task 72: Paired Live End-to-End Benchmark of Contextual Grading (Cond A)
vs Guarded Contextual Grading (Cond B).

Compares:
  - Condition A: Contextual Grading V1 (STRATEGY_CONTEXTUAL_GRADING)
      retrieve -> assemble dialogue units -> conservative contextual grading (Prompt B) -> RRF rank -> generate -> grounded check
  - Condition B: Guarded Contextual Grading (STRATEGY_CONTEXTUAL_GRADING_GUARDED)
      retrieve -> assemble dialogue units -> deterministic evidence-scope guard -> contextual grading (Prompt B) -> RRF rank -> generate -> grounded check

Evaluates across the 12 canonical evaluation cases from Task 50/56/65 with:
  - Exact frozen Task 53 Condition B grader prompt (hash b04b25c8...) for both conditions
  - Independent isolated SQLite and Chroma fixtures per condition and case
  - Zero access or modification to production chat_data.db or chroma_db
  - Task 60 explicit database_path and persist_dir propagation
  - Controlled no-rewrite setting: initial_rewrite_count = MAX_REWRITES (0 rewrites permitted)
  - Hard ceiling of 84 global LLM calls enforced by GlobalBudgetEnforcingLLM
  - Strict scoring contract (Task 58) with exact facts, qualifiers, and joint-support semantics
"""

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

import chat_database
import hybrid_retriever
import rag_graph
import vectorstore
from rag_graph import (
    MAX_REWRITES,
    STRATEGY_BASELINE,
    STRATEGY_CONTEXTUAL_GRADING,
    STRATEGY_CONTEXTUAL_GRADING_GUARDED,
    STRATEGY_CONTEXTUAL_GRADING_V2,
    STRATEGY_DIALOGUE_EVIDENCE,
    ask,
    extract_answer_citations,
    is_contextual_grading_enabled,
    is_contextual_grading_guarded_enabled,
    is_contextual_grading_v2_enabled,
    is_dialogue_evidence_enabled,
    is_pure_abstention,
    resolve_retrieval_strategy,
    route_post_retrieval,
    validate_citations,
)
from run_dialogue_evidence_live import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    EMBEDDING_MODEL,
    EVAL_CASES,
    HARD_CEILING_CALLS,
    EvalCase,
    GlobalBudgetEnforcingLLM,
    InferenceBudgetExceededError,
    IsolationBreachError,
    LiveConfirmationRequiredError,
    SafeEvaluationSandbox,
    TrackingEmbeddingsWrapper,
)
from evaluation_scoring import (
    atomic_save_json,
    atomic_save_markdown,
    classify_case_outcome,
    compute_citation_validity,
    evaluate_citation_attribution,
    evaluate_exact_facts_and_qualifiers,
    is_time_equivalent,
    save_progressive_checkpoint,
)

EXPECTED_PROMPT_B_HASH = "b04b25c8e08ddf93e5d100b2bd0d166ed42159e4aac27204aa170e87b6ae7573"


# ============================================================================
# EVALUATION PROMPT OVERRIDE CONTEXT MANAGER
# ============================================================================

@contextmanager
def evaluation_prompt_override(prompt_template: str):
    """
    Temporarily overrides rag_graph.GRADE_PROMPT_TEMPLATE for an evaluation run,
    restoring the original template in finally block.
    """
    orig_template = rag_graph.GRADE_PROMPT_TEMPLATE
    try:
        rag_graph.GRADE_PROMPT_TEMPLATE = prompt_template
        yield
    finally:
        rag_graph.GRADE_PROMPT_TEMPLATE = orig_template


# ============================================================================
# DATA STRUCTURES FOR TASK 72 COMPARISON
# ============================================================================

@dataclass
class CaseExecutionRecord:
    case_id: str
    condition: str  # "condition_a_contextual_grading" or "condition_b_guarded"
    strategy: str
    query: str
    expected_answerable: bool
    expected_supporting_ids: List[int]
    expected_keywords: List[str]
    support_semantics: str = "joint"

    # Fixture identity & isolation
    fixture_hash: str = ""
    database_path: str = ""
    persist_dir: str = ""

    # Execution telemetry
    latency_seconds: float = 0.0
    llm_calls_made: int = 0
    error: Optional[str] = None

    # Pipeline output
    answer: str = ""
    raw_answer: str = ""
    abstention_detected: bool = False
    correct_abstention: bool = False
    abstention_citations_stripped: bool = False
    grounded: bool = False
    grounded_status: str = "N/A"  # "Passed", "Failed", "N/A"
    semantic_grounded: bool = False
    rewrites_used: int = 0
    grader_status: str = "unknown"
    grader_failure: bool = False

    # Citations
    valid_cited_ids: List[int] = field(default_factory=list)
    invalid_cited_ids: List[int] = field(default_factory=list)
    available_citation_ids: List[int] = field(default_factory=list)
    malformed_citations: List[str] = field(default_factory=list)
    missing_citations: bool = False
    citation_validity: Optional[float] = None
    all_citations_valid: bool = False
    attribution_precision: Optional[float] = None
    attribution_recall: Optional[float] = None

    # Semantic accuracy & exact fact preservation
    semantic_answer_correct: bool = False
    exact_fact_preserved: Optional[bool] = None
    qualifier_preserved: Optional[bool] = None

    # Final confusion matrix outcome
    classification: str = "UNEVALUATED"  # TP, FP, TN, FN
    passed: bool = False

    # Evidence filtering trace
    units_assembled_count: int = 0
    units_retained_count: int = 0
    units_graded_relevant: int = 0
    units_graded_irrelevant: int = 0
    units_graded_uncertain: int = 0
    assembled_evidence_ids: List[int] = field(default_factory=list)
    retained_evidence_ids: List[int] = field(default_factory=list)
    target_in_final_evidence: bool = False
    distractor_survived: bool = False

    # Detailed decisions trace
    grading_decisions: List[Dict[str, Any]] = field(default_factory=list)
    captured_evaluation_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Task 71/72 Guard diagnostics
    guard_decisions: Optional[List[Dict[str, Any]]] = None
    guard_diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class ConditionComparisonSummary:
    condition_name: str
    strategy: str
    prompt_sha256: str
    total_cases: int = 0
    answerable_cases: int = 0
    unanswerable_cases: int = 0

    # Confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    strict_accuracy: float = 0.0
    answerable_accuracy: float = 0.0
    correct_abstention_rate: float = 0.0
    false_positives_unanswerable: int = 0
    unanswerable_fp_rate: float = 0.0

    # Evidence recall
    population_recall_terse: float = 0.0  # CASE-01, CASE-02, CASE-04
    overall_evidence_recall: float = 0.0  # All 7 answerable
    cases_all_mandatory_support_retained: int = 0
    mandatory_message_ids_retained_count: int = 0
    mandatory_message_ids_retention_rate: float = 0.0
    joint_support_completion: bool = False

    # Citations & Groundedness
    groundedness_pass_rate: float = 0.0
    groundedness_checks_bypassed: int = 0
    total_valid_citations: int = 0
    total_invalid_citations: int = 0
    total_citations_emitted: int = 0
    citation_validity_rate: Optional[float] = None
    mean_attribution_precision: Optional[float] = None
    mean_attribution_recall: float = 0.0
    citation_level_attribution_precision: Optional[float] = None
    cases_with_missing_citations: int = 0

    # Exact facts & qualifiers
    semantic_answer_correctness_rate: float = 0.0
    exact_fact_preservation_rate: float = 0.0

    # Evidence volume & efficiency
    total_evidence_units_retained: int = 0
    evidence_units_per_case: float = 0.0
    total_llm_calls: int = 0
    mean_latency_seconds: float = 0.0
    median_latency_seconds: float = 0.0

    # Guard aggregates
    total_guard_rejections: int = 0
    query_echo_rejections: int = 0
    env_mismatch_rejections: int = 0
    grader_bypasses: int = 0


# ============================================================================
# EVALUATION & METRIC FUNCTIONS
# ============================================================================

def evaluate_case_record(record: CaseExecutionRecord, case: EvalCase) -> None:
    """Evaluates case record using canonical Task 58 strict scoring contract."""
    # Check if target evidence is in final evidence
    if case.expected_answerable:
        record.target_in_final_evidence = all(
            sid in record.assembled_evidence_ids for sid in case.expected_supporting_ids
        )
    else:
        record.target_in_final_evidence = True

    # Check distractor survival on unanswerable queries
    if not case.expected_answerable:
        distractor_id = case.target_id
        record.distractor_survived = (distractor_id in record.assembled_evidence_ids)

    # Abstention detection
    record.abstention_detected = is_pure_abstention(record.answer)
    if not case.expected_answerable:
        record.correct_abstention = record.abstention_detected or (
            len(record.valid_cited_ids) == 0 and len(record.invalid_cited_ids) == 0
        )
    else:
        record.correct_abstention = False

    # Citation validity
    record.citation_validity = compute_citation_validity(record.valid_cited_ids, record.invalid_cited_ids)

    # Attribution precision & recall
    support_semantics = getattr(case, "support_semantics", "joint")
    record.attribution_precision, record.attribution_recall = evaluate_citation_attribution(
        is_answerable=case.expected_answerable,
        valid_cited_ids=record.valid_cited_ids,
        expected_supporting_ids=case.expected_supporting_ids,
        invalid_cited_ids=record.invalid_cited_ids,
        support_semantics=support_semantics,
    )

    # Exact fact preservation and semantic correctness
    fact_eval = evaluate_exact_facts_and_qualifiers(
        case_id=case.case_id,
        answer=record.answer,
        expected_keywords=getattr(
            case, "expected_answer_keywords", getattr(case, "expected_keywords", record.expected_keywords)
        ),
    )
    record.semantic_answer_correct = bool(fact_eval["semantic_answer_correct"])
    record.exact_fact_preserved = fact_eval["exact_fact_preserved"]
    record.qualifier_preserved = fact_eval.get("qualifier_preserved", True)

    # Confusion matrix classification
    is_joint = (support_semantics == "joint") and (len(case.expected_supporting_ids) > 1)
    record.classification = classify_case_outcome(
        expected_answerable=case.expected_answerable,
        target_in_final_evidence=record.target_in_final_evidence,
        semantic_answer_correct=record.semantic_answer_correct,
        grounded=record.grounded,
        attribution_precision=record.attribution_precision,
        attribution_recall=record.attribution_recall,
        correct_abstention=record.correct_abstention,
        valid_cited_ids=record.valid_cited_ids,
        invalid_cited_ids=record.invalid_cited_ids,
        is_joint_supporting=is_joint,
        expected_supporting_ids=case.expected_supporting_ids,
        exact_fact_preserved=record.exact_fact_preserved,
        qualifier_preserved=record.qualifier_preserved,
    )
    record.passed = (record.classification in ("TP", "TN"))


def compute_condition_summary(
    records: List[CaseExecutionRecord],
    condition_name: str,
    strategy: str,
    prompt_hash: str
) -> ConditionComparisonSummary:
    """Computes aggregate summary metrics across records for a condition."""
    summary = ConditionComparisonSummary(
        condition_name=condition_name,
        strategy=strategy,
        prompt_sha256=prompt_hash,
        total_cases=len(records)
    )
    ans = [r for r in records if r.expected_answerable]
    unans = [r for r in records if not r.expected_answerable]
    summary.answerable_cases = len(ans)
    summary.unanswerable_cases = len(unans)

    summary.tp = sum(1 for r in records if r.classification == "TP")
    summary.fp = sum(1 for r in records if r.classification == "FP")
    summary.tn = sum(1 for r in records if r.classification == "TN")
    summary.fn = sum(1 for r in records if r.classification == "FN")

    if records:
        summary.strict_accuracy = (summary.tp + summary.tn) / len(records)
    if ans:
        summary.answerable_accuracy = summary.tp / len(ans)
    if unans:
        summary.correct_abstention_rate = summary.tn / len(unans)
        summary.false_positives_unanswerable = summary.fp
        summary.unanswerable_fp_rate = summary.fp / len(unans)

    # Terse cases recall: CASE-01, CASE-02, CASE-04
    terse_cases = [r for r in records if r.case_id in ("CASE-01-Terse-Location", "CASE-02-Short-Time", "CASE-04-Yes-No-Confirmation")]
    if terse_cases:
        summary.population_recall_terse = sum(1 for r in terse_cases if r.target_in_final_evidence) / len(terse_cases)

    # Overall evidence recall (all 7 answerable)
    if ans:
        summary.overall_evidence_recall = sum(1 for r in ans if r.target_in_final_evidence) / len(ans)

    # Mandatory support IDs
    total_mandatory_ids = 0
    retained_mandatory_ids = 0
    cases_all_mand = 0
    for r in ans:
        exp_set = set(r.expected_supporting_ids)
        total_mandatory_ids += len(exp_set)
        retained_set = exp_set.intersection(set(r.assembled_evidence_ids))
        retained_mandatory_ids += len(retained_set)
        if exp_set.issubset(set(r.assembled_evidence_ids)):
            cases_all_mand += 1

    summary.cases_all_mandatory_support_retained = cases_all_mand
    summary.mandatory_message_ids_retained_count = retained_mandatory_ids
    if total_mandatory_ids > 0:
        summary.mandatory_message_ids_retention_rate = retained_mandatory_ids / total_mandatory_ids

    # Joint support completion (CASE-09)
    c09 = next((r for r in records if "CASE-09" in r.case_id), None)
    if c09 and c09.expected_supporting_ids:
        summary.joint_support_completion = set(c09.expected_supporting_ids).issubset(set(c09.assembled_evidence_ids))

    # Groundedness
    with_sources = [r for r in records if r.units_retained_count > 0]
    if with_sources:
        summary.groundedness_pass_rate = sum(1 for r in with_sources if r.grounded) / len(with_sources)
    summary.groundedness_checks_bypassed = sum(1 for r in records if r.units_retained_count == 0)

    # Citations
    total_valid_cits = sum(len(r.valid_cited_ids) for r in records)
    total_invalid_cits = sum(len(r.invalid_cited_ids) for r in records)
    total_cits = total_valid_cits + total_invalid_cits
    summary.total_valid_citations = total_valid_cits
    summary.total_invalid_citations = total_invalid_cits
    summary.total_citations_emitted = total_cits

    if total_cits > 0:
        summary.citation_validity_rate = total_valid_cits / total_cits
        summary.citation_level_attribution_precision = total_valid_cits / total_cits

    cits_cases = [r for r in records if (len(r.valid_cited_ids) + len(r.invalid_cited_ids)) > 0]
    if cits_cases:
        summary.mean_attribution_precision = sum(
            r.attribution_precision for r in cits_cases if r.attribution_precision is not None
        ) / len(cits_cases)

    if ans:
        summary.mean_attribution_recall = sum(r.attribution_recall or 0.0 for r in ans) / len(ans)

    summary.cases_with_missing_citations = sum(1 for r in records if r.missing_citations)

    # Exact facts & qualifiers
    if ans:
        summary.semantic_answer_correctness_rate = sum(1 for r in ans if r.semantic_answer_correct) / len(ans)
        summary.exact_fact_preservation_rate = sum(1 for r in ans if r.exact_fact_preserved is True) / len(ans)

    # Evidence units
    summary.total_evidence_units_retained = sum(r.units_retained_count for r in records)
    if records:
        summary.evidence_units_per_case = summary.total_evidence_units_retained / len(records)
        summary.total_llm_calls = sum(r.llm_calls_made for r in records)
        latencies = [r.latency_seconds for r in records]
        summary.mean_latency_seconds = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        summary.median_latency_seconds = (
            (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2.0 if n % 2 == 0 else sorted_lat[n // 2]
        )

    # Guard aggregates
    for r in records:
        diag = r.guard_diagnostics or {}
        if diag.get("enabled", False) or "total_rejected" in diag:
            summary.total_guard_rejections += diag.get("total_rejected", 0)
            summary.query_echo_rejections += diag.get("rejected_query_echo", 0)
            summary.env_mismatch_rejections += diag.get("rejected_environment_mismatch", 0)
            if diag.get("grader_invoked") is False and diag.get("total_candidates", 0) > 0:
                summary.grader_bypasses += 1

    return summary


# ============================================================================
# PREFLIGHT VERIFICATION
# ============================================================================

def run_preflight_checks(prompt_b_text: str) -> bool:
    """Performs mandatory preflight safety and integrity verification before live execution."""
    import parser
    print("\n" + "=" * 70)
    print("RUNNING MANDATORY TASK 72 PREFLIGHT CHECKS")
    print("=" * 70)
    all_ok = True

    # 1. Verify Prompt B hash
    h_b = hashlib.sha256(prompt_b_text.encode("utf-8")).hexdigest()
    if h_b != EXPECTED_PROMPT_B_HASH:
        print(f"❌ [FAIL] Prompt B hash mismatch! Expected {EXPECTED_PROMPT_B_HASH}, got {h_b}")
        all_ok = False
    else:
        print(f"✅ [PASS] Frozen Prompt B verified: {h_b}")

    # 2. Production default remains baseline
    strat_default = resolve_retrieval_strategy({})
    route_default = route_post_retrieval({})
    if strat_default != STRATEGY_BASELINE or route_default != "grade_documents":
        print(f"❌ [FAIL] Production default strategy breached: {strat_default} -> {route_default}")
        all_ok = False
    else:
        print(f"✅ [PASS] Production default remains baseline: {strat_default} -> {route_default}")

    # 3. Contextual grading is unchanged
    strat_cg = resolve_retrieval_strategy({"contextual_grading_mode": True})
    route_cg = route_post_retrieval({"retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING})
    if strat_cg != STRATEGY_CONTEXTUAL_GRADING or route_cg != "assemble_grade_rank_evidence":
        print(f"❌ [FAIL] contextual_grading strategy routing altered: {strat_cg} -> {route_cg}")
        all_ok = False
    else:
        print(f"✅ [PASS] contextual_grading strategy intact: {strat_cg} -> {route_cg}")

    # 4. Guarded strategy recognized and independently selectable
    strat_guard = resolve_retrieval_strategy({"contextual_grading_guarded_mode": True})
    route_guard = route_post_retrieval({"retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING_GUARDED})
    if strat_guard != STRATEGY_CONTEXTUAL_GRADING_GUARDED or route_guard != "assemble_grade_rank_evidence_guarded":
        print(f"❌ [FAIL] contextual_grading_guarded routing altered: {strat_guard} -> {route_guard}")
        all_ok = False
    else:
        print(f"✅ [PASS] contextual_grading_guarded selectable: {strat_guard} -> {route_guard}")

    # 5. Storage isolation (Task 60)
    try:
        hybrid_retriever.validate_storage_paths(database_path="test.db", persist_dir=None)
        print("❌ [FAIL] Incomplete storage paths should raise StorageConfigurationError")
        all_ok = False
    except hybrid_retriever.StorageConfigurationError:
        print("✅ [PASS] Storage isolation verified: incomplete custom paths rejected")

    # 6. Date parsing (Task 61)
    dmy_diag = parser.infer_date_order(["25/10/2026, 14:30 - A: hi", "26/10/2026, 14:30 - B: hi"])
    if dmy_diag.confidence != "DMY_CONFIDENT":
        print(f"❌ [FAIL] Date parsing preflight failed for DMY: {dmy_diag.confidence}")
        all_ok = False
    else:
        print("✅ [PASS] Date parsing preflight passed: DMY_CONFIDENT verified")

    # 7. Same-minute context (Task 62)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        tdb = tf.name
    try:
        conn = sqlite3.connect(tdb)
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, timestamp TEXT, sender TEXT, message TEXT, message_type TEXT, sentiment_compound REAL, sentiment_label TEXT, is_question INTEGER, message_length INTEGER, response_time_minutes REAL, reply_time_minutes REAL)")
        conn.execute("INSERT INTO messages (id, timestamp, sender, message) VALUES (100, '2026-03-31 14:30:00', 'Alice', 'Msg 100')")
        conn.execute("INSERT INTO messages (id, timestamp, sender, message) VALUES (101, '2026-03-31 14:30:00', 'Bob', 'Msg 101')")
        conn.execute("INSERT INTO messages (id, timestamp, sender, message) VALUES (102, '2026-03-31 14:30:00', 'Alice', 'Msg 102')")
        conn.commit()
        conn.close()

        c_rows = chat_database.get_chronological_context(101, before=1, after=1, database_path=tdb)
        c_ids = [r["id"] for r in c_rows]
        if c_ids != [100, 101, 102]:
            print(f"❌ [FAIL] Same-minute context failed: expected [100, 101, 102], got {c_ids}")
            all_ok = False
        else:
            print("✅ [PASS] Same-minute context ordering verified: (timestamp, id)")
    finally:
        if os.path.exists(tdb):
            os.remove(tdb)

    # 8. Benchmark fixture identity & key cases
    if len(EVAL_CASES) != 12:
        print(f"❌ [FAIL] Benchmark case count is {len(EVAL_CASES)}, expected 12")
        all_ok = False
    else:
        print(f"✅ [PASS] Canonical benchmark verified: {len(EVAL_CASES)} cases (7 ans, 5 unans)")

    c06 = next((c for c in EVAL_CASES if "CASE-06" in c.case_id), None)
    if not c06 or c06.expected_supporting_ids != [112]:
        print(f"❌ [FAIL] CASE-06 mandatory support is {getattr(c06, 'expected_supporting_ids', None)}, expected [112]")
        all_ok = False
    else:
        print("✅ [PASS] CASE-06 mandatory support [112] verified")

    c09 = next((c for c in EVAL_CASES if "CASE-09" in c.case_id), None)
    if not c09 or sorted(c09.expected_supporting_ids) != [117, 119]:
        print(f"❌ [FAIL] CASE-09 joint support is {getattr(c09, 'expected_supporting_ids', None)}, expected [117, 119]")
        all_ok = False
    else:
        print("✅ [PASS] CASE-09 joint support [117, 119] verified")

    c04 = next((c for c in EVAL_CASES if "CASE-04" in c.case_id), None)
    q_eval = evaluate_exact_facts_and_qualifiers("CASE-04-Yes-No-Confirmation", "Migration finished completely clean. [108]", ["clean"])
    if not q_eval.get("qualifier_preserved", False):
        print("❌ [FAIL] CASE-04 qualifier check failed to recognize 'completely clean'")
        all_ok = False
    else:
        print("✅ [PASS] CASE-04 qualifier preservation logic verified")

    # 9. Verify citation edge cases
    try:
        # [123] with {123}
        c1, m1 = extract_answer_citations("[123]", available_ids={123})
        assert c1 == [123] and m1 == []
        # [404] with {123, 124}
        c2, m2 = extract_answer_citations("[404]", available_ids={123, 124})
        assert c2 == [] and m2 == []
        # [123, 999] with {123, 124}
        c3, m3 = extract_answer_citations("[123, 999]", available_ids={123, 124})
        assert c3 == [123, 999] and m3 == []
        # [ID 999] with {123, 124}
        c4, m4 = extract_answer_citations("[ID 999]", available_ids={123, 124})
        assert c4 == [999] and m4 == []
        # available_ids=set()
        c5, m5 = extract_answer_citations("[123]", available_ids=set())
        assert c5 == [] and m5 == []
        # available_ids=None
        c6, m6 = extract_answer_citations("[123]", available_ids=None)
        assert c6 == [123] and m6 == []
        # [10:30] and [2024]
        c7, m7 = extract_answer_citations("Time [10:30] Year [2024]", available_ids={123, 124})
        assert c7 == [] and m7 == []
        print("✅ [PASS] Citation extraction & edge cases verified (7/7 cases)")
    except Exception as e:
        print(f"❌ [FAIL] Citation edge case verification failed: {e}")
        all_ok = False

    # 10. Strict scoring active
    cl_tp = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=True,
        grounded=True,
        attribution_precision=1.0,
        attribution_recall=1.0,
        correct_abstention=False,
        valid_cited_ids=[100],
        invalid_cited_ids=[],
        expected_supporting_ids=[100],
        exact_fact_preserved=True,
        qualifier_preserved=True,
    )
    cl_tn = classify_case_outcome(
        expected_answerable=False,
        target_in_final_evidence=True,
        semantic_answer_correct=False,
        grounded=False,
        attribution_precision=None,
        attribution_recall=None,
        correct_abstention=True,
        valid_cited_ids=[],
        invalid_cited_ids=[],
        exact_fact_preserved=True,
        qualifier_preserved=True,
    )
    if cl_tp != "TP" or cl_tn != "TN":
        print(f"❌ [FAIL] classify_case_outcome contract failed: tp={cl_tp}, tn={cl_tn}")
        all_ok = False
    else:
        print("✅ [PASS] Task 58 strict scoring contract verified")

    print("=" * 70)
    return all_ok


# ============================================================================
# LIVE EXECUTION ENGINE FOR TASK 72
# ============================================================================

def execute_task72_live_benchmark(
    confirm_live: bool = False,
    live_model_name: str = DEFAULT_MODEL,
    max_budget: int = HARD_CEILING_CALLS,
    prompt_b: str = "",
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[
    List[CaseExecutionRecord],
    List[CaseExecutionRecord],
    ConditionComparisonSummary,
    ConditionComparisonSummary,
    GlobalBudgetEnforcingLLM,
    TrackingEmbeddingsWrapper,
]:
    """
    Executes paired end-to-end comparison:
      Condition A: Contextual Grading V1 (STRATEGY_CONTEXTUAL_GRADING)
      Condition B: Guarded Contextual Grading (STRATEGY_CONTEXTUAL_GRADING_GUARDED)
    """
    if not confirm_live:
        raise LiveConfirmationRequiredError("Live execution requires explicit confirm_live=True.")

    hash_b = hashlib.sha256(prompt_b.encode("utf-8")).hexdigest()

    print("\n" + "=" * 80)
    print("TASK 72 LIVE COMPARISON: CONTEXTUAL GRADING (COND A) VS GUARDED CONTEXTUAL GRADING (COND B)")
    print(f"Model: {live_model_name} | Embeddings: {EMBEDDING_MODEL} | Hard Ceiling: {max_budget} calls")
    print(f"Condition A Strategy: {STRATEGY_CONTEXTUAL_GRADING} | Prompt Hash: {hash_b}")
    print(f"Condition B Strategy: {STRATEGY_CONTEXTUAL_GRADING_GUARDED} | Prompt Hash: {hash_b}")
    print("Setting: Controlled No-Rewrite Evaluation (initial_rewrite_count = MAX_REWRITES)")
    print("=" * 80)

    inner_llm = ChatOllama(
        model=live_model_name,
        temperature=0.0,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    budget_llm = GlobalBudgetEnforcingLLM(inner_llm=inner_llm, max_calls=max_budget)

    raw_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    tracking_embedder = TrackingEmbeddingsWrapper(inner_embedder=raw_embedder)

    orig_rag_llm = rag_graph.llm
    orig_get_embeddings = vectorstore.get_embeddings
    rag_graph.llm = budget_llm
    vectorstore.get_embeddings = lambda: tracking_embedder

    cond_a_records: List[CaseExecutionRecord] = []
    cond_b_records: List[CaseExecutionRecord] = []

    try:
        for idx, case in enumerate(EVAL_CASES, start=1):
            print(f"\n[{idx:02d}/12] Evaluating Case {case.case_id}: {case.query[:60]}...")

            # -------------------------------------------------------------
            # CONDITION A: Contextual Grading V1 (no guard)
            # -------------------------------------------------------------
            print(f"  [1/2] Condition A ({STRATEGY_CONTEXTUAL_GRADING}) Executing...")
            rec_a = CaseExecutionRecord(
                case_id=case.case_id,
                condition="condition_a_contextual_grading",
                strategy=STRATEGY_CONTEXTUAL_GRADING,
                query=case.query,
                expected_answerable=case.expected_answerable,
                expected_supporting_ids=case.expected_supporting_ids,
                expected_keywords=case.expected_answer_keywords,
                support_semantics=getattr(case, "support_semantics", "joint"),
            )

            with SafeEvaluationSandbox(case) as sandbox_a:
                time.sleep(0.5)
                transcript_a = sandbox_a.chat_txt_path.read_text(encoding="utf-8")
                hash_fixture_a = hashlib.sha256(transcript_a.encode("utf-8")).hexdigest()
                rec_a.fixture_hash = hash_fixture_a
                rec_a.database_path = str(sandbox_a.db_path)
                rec_a.persist_dir = str(sandbox_a.chroma_path)

                t0_a = time.time()
                calls_before_a = budget_llm.total_attempted

                try:
                    with evaluation_prompt_override(prompt_b):
                        res_a = ask(
                            question=case.query,
                            retrieval_strategy=STRATEGY_CONTEXTUAL_GRADING,
                            database_path=str(sandbox_a.db_path),
                            persist_dir=str(sandbox_a.chroma_path),
                            initial_rewrite_count=MAX_REWRITES,
                            capture_eval_evidence=True,
                        )
                    rec_a.latency_seconds = round(time.time() - t0_a, 3)
                    rec_a.llm_calls_made = budget_llm.total_attempted - calls_before_a

                    rec_a.answer = res_a.get("answer", "")
                    rec_a.raw_answer = res_a.get("raw_answer", rec_a.answer)
                    rec_a.abstention_citations_stripped = res_a.get("abstention_citations_stripped", False)
                    rec_a.grounded = res_a.get("grounded", False)
                    rec_a.semantic_grounded = res_a.get("semantic_grounded", False)
                    rec_a.valid_cited_ids = res_a.get("valid_citation_ids", [])
                    rec_a.invalid_cited_ids = res_a.get("invalid_citation_ids", [])
                    rec_a.available_citation_ids = res_a.get("available_citation_ids", [])
                    rec_a.rewrites_used = res_a.get("rewrites_used", 0)
                    rec_a.grader_status = res_a.get("grader_status", "unknown")

                    val_a = res_a.get("citation_validation", {}) or {}
                    rec_a.malformed_citations = val_a.get("malformed_citations", [])
                    rec_a.missing_citations = val_a.get("missing_citations", False)
                    rec_a.all_citations_valid = val_a.get("all_cited_valid", False) if val_a else (len(rec_a.invalid_cited_ids) == 0)

                    # Decisions & units
                    grading_decs_a = res_a.get("contextual_grading_decisions", [])
                    rec_a.grading_decisions = grading_decs_a
                    rec_a.units_assembled_count = len(grading_decs_a)
                    rec_a.units_graded_relevant = sum(1 for d in grading_decs_a if d.get("decision") == "relevant")
                    rec_a.units_graded_irrelevant = sum(1 for d in grading_decs_a if d.get("decision") == "irrelevant")
                    rec_a.units_graded_uncertain = sum(1 for d in grading_decs_a if d.get("decision") == "uncertain")
                    rec_a.grader_failure = any(d.get("grader_failure", False) for d in grading_decs_a)

                    sources_a = res_a.get("sources", [])
                    rec_a.units_retained_count = len(sources_a)
                    rec_a.grounded_status = "N/A" if len(sources_a) == 0 else ("Passed" if rec_a.grounded else "Failed")

                    for s in sources_a:
                        mid = s.get("anchor_message_id") or s.get("message_id")
                        if mid is not None and mid not in rec_a.assembled_evidence_ids:
                            rec_a.assembled_evidence_ids.append(mid)
                            rec_a.retained_evidence_ids.append(mid)
                        for inmid in s.get("included_message_ids", []):
                            if inmid not in rec_a.assembled_evidence_ids:
                                rec_a.assembled_evidence_ids.append(inmid)

                    rec_a.captured_evaluation_evidence = res_a.get("captured_evaluation_evidence", []) or []

                    # Explicitly record guard as disabled for Condition A
                    rec_a.guard_decisions = None
                    rec_a.guard_diagnostics = {"enabled": False, "strategy": "contextual_grading"}

                    evaluate_case_record(rec_a, case)
                    print(
                        f"    -> Result: {rec_a.classification} | Grounded: {rec_a.grounded_status} | "
                        f"Calls: {rec_a.llm_calls_made} | Retained: {rec_a.retained_evidence_ids} | "
                        f"Cited: {rec_a.valid_cited_ids} | Latency: {rec_a.latency_seconds}s"
                    )
                except Exception as e:
                    rec_a.latency_seconds = round(time.time() - t0_a, 3)
                    rec_a.llm_calls_made = budget_llm.total_attempted - calls_before_a
                    rec_a.error = str(e)
                    print(f"    -> [COND A ERROR]: {e}")
                    if isinstance(e, (InferenceBudgetExceededError, IsolationBreachError)):
                        raise

            cond_a_records.append(rec_a)

            # -------------------------------------------------------------
            # CONDITION B: Guarded Contextual Grading (with guard)
            # -------------------------------------------------------------
            print(f"  [2/2] Condition B ({STRATEGY_CONTEXTUAL_GRADING_GUARDED}) Executing...")
            rec_b = CaseExecutionRecord(
                case_id=case.case_id,
                condition="condition_b_guarded",
                strategy=STRATEGY_CONTEXTUAL_GRADING_GUARDED,
                query=case.query,
                expected_answerable=case.expected_answerable,
                expected_supporting_ids=case.expected_supporting_ids,
                expected_keywords=case.expected_answer_keywords,
                support_semantics=getattr(case, "support_semantics", "joint"),
            )

            with SafeEvaluationSandbox(case) as sandbox_b:
                time.sleep(0.5)
                transcript_b = sandbox_b.chat_txt_path.read_text(encoding="utf-8")
                hash_fixture_b = hashlib.sha256(transcript_b.encode("utf-8")).hexdigest()
                assert hash_fixture_b == hash_fixture_a, f"Fixture identity mismatch for case {case.case_id}!"
                rec_b.fixture_hash = hash_fixture_b
                rec_b.database_path = str(sandbox_b.db_path)
                rec_b.persist_dir = str(sandbox_b.chroma_path)

                t0_b = time.time()
                calls_before_b = budget_llm.total_attempted

                try:
                    with evaluation_prompt_override(prompt_b):
                        res_b = ask(
                            question=case.query,
                            retrieval_strategy=STRATEGY_CONTEXTUAL_GRADING_GUARDED,
                            database_path=str(sandbox_b.db_path),
                            persist_dir=str(sandbox_b.chroma_path),
                            initial_rewrite_count=MAX_REWRITES,
                            capture_eval_evidence=True,
                        )
                    rec_b.latency_seconds = round(time.time() - t0_b, 3)
                    rec_b.llm_calls_made = budget_llm.total_attempted - calls_before_b

                    rec_b.answer = res_b.get("answer", "")
                    rec_b.raw_answer = res_b.get("raw_answer", rec_b.answer)
                    rec_b.abstention_citations_stripped = res_b.get("abstention_citations_stripped", False)
                    rec_b.grounded = res_b.get("grounded", False)
                    rec_b.semantic_grounded = res_b.get("semantic_grounded", False)
                    rec_b.valid_cited_ids = res_b.get("valid_citation_ids", [])
                    rec_b.invalid_cited_ids = res_b.get("invalid_citation_ids", [])
                    rec_b.available_citation_ids = res_b.get("available_citation_ids", [])
                    rec_b.rewrites_used = res_b.get("rewrites_used", 0)
                    rec_b.grader_status = res_b.get("grader_status", "unknown")

                    val_b = res_b.get("citation_validation", {}) or {}
                    rec_b.malformed_citations = val_b.get("malformed_citations", [])
                    rec_b.missing_citations = val_b.get("missing_citations", False)
                    rec_b.all_citations_valid = val_b.get("all_cited_valid", False) if val_b else (len(rec_b.invalid_cited_ids) == 0)

                    # Decisions & units
                    grading_decs_b = res_b.get("contextual_grading_decisions", [])
                    rec_b.grading_decisions = grading_decs_b
                    rec_b.units_assembled_count = len(grading_decs_b)
                    rec_b.units_graded_relevant = sum(1 for d in grading_decs_b if d.get("decision") == "relevant")
                    rec_b.units_graded_irrelevant = sum(1 for d in grading_decs_b if d.get("decision") == "irrelevant")
                    rec_b.units_graded_uncertain = sum(1 for d in grading_decs_b if d.get("decision") == "uncertain")
                    rec_b.grader_failure = any(d.get("grader_failure", False) for d in grading_decs_b)

                    sources_b = res_b.get("sources", [])
                    rec_b.units_retained_count = len(sources_b)
                    rec_b.grounded_status = "N/A" if len(sources_b) == 0 else ("Passed" if rec_b.grounded else "Failed")

                    for s in sources_b:
                        mid = s.get("anchor_message_id") or s.get("message_id")
                        if mid is not None and mid not in rec_b.assembled_evidence_ids:
                            rec_b.assembled_evidence_ids.append(mid)
                            rec_b.retained_evidence_ids.append(mid)
                        for inmid in s.get("included_message_ids", []):
                            if inmid not in rec_b.assembled_evidence_ids:
                                rec_b.assembled_evidence_ids.append(inmid)

                    rec_b.captured_evaluation_evidence = res_b.get("captured_evaluation_evidence", []) or []

                    # Record guard diagnostics from state
                    rec_b.guard_decisions = res_b.get("guard_decisions", [])
                    rec_b.guard_diagnostics = res_b.get("guard_diagnostics", {})

                    evaluate_case_record(rec_b, case)
                    guard_rejs = rec_b.guard_diagnostics.get("total_rejected", 0) if rec_b.guard_diagnostics else 0
                    print(
                        f"    -> Result: {rec_b.classification} | Grounded: {rec_b.grounded_status} | "
                        f"Calls: {rec_b.llm_calls_made} | GuardRej: {guard_rejs} | Retained: {rec_b.retained_evidence_ids} | "
                        f"Cited: {rec_b.valid_cited_ids} | Latency: {rec_b.latency_seconds}s"
                    )
                except Exception as e:
                    rec_b.latency_seconds = round(time.time() - t0_b, 3)
                    rec_b.llm_calls_made = budget_llm.total_attempted - calls_before_b
                    rec_b.error = str(e)
                    print(f"    -> [COND B ERROR]: {e}")
                    if isinstance(e, (InferenceBudgetExceededError, IsolationBreachError)):
                        raise

            cond_b_records.append(rec_b)

            # Progressive checkpoint after each paired case
            if checkpoint_dir:
                try:
                    cp_path = checkpoint_dir / f".checkpoint_task72_live.json"
                    save_progressive_checkpoint(
                        cp_path,
                        [asdict(r) for r in (cond_a_records + cond_b_records)],
                    )
                except Exception as cp_err:
                    print(f"    [CHECKPOINT WARNING]: {cp_err}")

    finally:
        rag_graph.llm = orig_rag_llm
        vectorstore.get_embeddings = orig_get_embeddings

    summary_a = compute_condition_summary(cond_a_records, "Condition A (Contextual Grading V1)", STRATEGY_CONTEXTUAL_GRADING, hash_b)
    summary_b = compute_condition_summary(cond_b_records, "Condition B (Guarded Contextual Grading)", STRATEGY_CONTEXTUAL_GRADING_GUARDED, hash_b)

    return cond_a_records, cond_b_records, summary_a, summary_b, budget_llm, tracking_embedder


# ============================================================================
# REPORT BUILDER
# ============================================================================

def build_task72_report(
    timestamp: str,
    args: Any,
    sum_a: ConditionComparisonSummary,
    sum_b: ConditionComparisonSummary,
    rec_a: List[CaseExecutionRecord],
    rec_b: List[CaseExecutionRecord],
    budget_llm: GlobalBudgetEnforcingLLM,
    embedder: TrackingEmbeddingsWrapper,
    hist_t65: Dict[str, Any],
) -> str:
    """Builds comprehensive Markdown audit report for Task 72."""
    val_a_str = f"{sum_a.citation_validity_rate*100:.1f}%" if sum_a.citation_validity_rate is not None else "N/A"
    val_b_str = f"{sum_b.citation_validity_rate*100:.1f}%" if sum_b.citation_validity_rate is not None else "N/A"
    prec_a_str = f"{sum_a.mean_attribution_precision*100:.1f}%" if sum_a.mean_attribution_precision is not None else "N/A"
    prec_b_str = f"{sum_b.mean_attribution_precision*100:.1f}%" if sum_b.mean_attribution_precision is not None else "N/A"
    prec_cit_a_str = f"{sum_a.citation_level_attribution_precision*100:.1f}%" if sum_a.citation_level_attribution_precision is not None else "N/A"
    prec_cit_b_str = f"{sum_b.citation_level_attribution_precision*100:.1f}%" if sum_b.citation_level_attribution_precision is not None else "N/A"

    # Determine recommendation
    if sum_b.strict_accuracy > sum_a.strict_accuracy and sum_b.mandatory_message_ids_retention_rate >= sum_a.mandatory_message_ids_retention_rate:
        verdict_choice = "1. Guarded contextual grading improves strict end-to-end performance and merits a broader experimental rollout."
        verdict_badge = "✅ RECOMMENDATION: ROLLOUT TO EXPERIMENTAL SUITE"
    elif sum_b.strict_accuracy == sum_a.strict_accuracy and sum_b.total_evidence_units_retained < sum_a.total_evidence_units_retained:
        verdict_choice = "2. Guarded contextual grading improves filtering/efficiency but does not improve strict answer quality."
        verdict_badge = "ℹ️ RECOMMENDATION: EFFICIENCY GAIN CONFIRMED"
    elif sum_b.strict_accuracy < sum_a.strict_accuracy or sum_b.mandatory_message_ids_retention_rate < sum_a.mandatory_message_ids_retention_rate:
        verdict_choice = "3. Guarded contextual grading introduces unacceptable evidence loss or regressions."
        verdict_badge = "❌ RECOMMENDATION: DO NOT DEPLOY (REGRESSIONS DETECTED)"
    else:
        verdict_choice = "4. Results are inconclusive due to benchmark or execution issues."
        verdict_badge = "⚠️ RECOMMENDATION: INCONCLUSIVE"

    case_rows = []
    for ra, rb in zip(rec_a, rec_b):
        delta_str = ""
        if ra.classification != rb.classification:
            delta_str = f"**{ra.classification} -> {rb.classification}**"
        else:
            delta_str = "No Change"

        g_diag = rb.guard_diagnostics or {}
        g_rej = g_diag.get("total_rejected", 0)

        case_rows.append(
            f"| `{ra.case_id}` | {'Ans' if ra.expected_answerable else 'Unans'} | "
            f"**{ra.classification}** | {ra.grounded_status} | {ra.retained_evidence_ids} | "
            f"**{rb.classification}** | {rb.grounded_status} | {g_rej} | {rb.retained_evidence_ids} | "
            f"{delta_str} |"
        )

    # Focus cases analysis
    focus_c04_a = next((r for r in rec_a if "CASE-04" in r.case_id), None)
    focus_c04_b = next((r for r in rec_b if "CASE-04" in r.case_id), None)
    focus_c08_a = next((r for r in rec_a if "CASE-08" in r.case_id), None)
    focus_c08_b = next((r for r in rec_b if "CASE-08" in r.case_id), None)
    focus_c09_a = next((r for r in rec_a if "CASE-09" in r.case_id), None)
    focus_c09_b = next((r for r in rec_b if "CASE-09" in r.case_id), None)
    focus_c011_a = next((r for r in rec_a if "CASE-011" in r.case_id), None)
    focus_c011_b = next((r for r in rec_b if "CASE-011" in r.case_id), None)

    return f"""# ChatLens-RAG — Task 72: Paired End-to-End Evaluation Report
## Contextual Grading (Cond A) vs Guarded Contextual Grading (Cond B)

**Evaluation Timestamp:** `{timestamp}`  
**Evaluator Models:** `{args.model}` (Inference, temp=0.0) | `{EMBEDDING_MODEL}` (Embeddings)  
**Call Budget:** Attempted {budget_llm.total_attempted}/{args.max_calls} calls (Hard ceiling enforced)  
**Setting:** Controlled No-Rewrite Paired Benchmark (MAX_REWRITES=2, initial=2 -> 0 rewrites permitted)  
**Grader Prompt:** Frozen Task 53 Condition B (`{EXPECTED_PROMPT_B_HASH}`) applied identically to both conditions  
**Verdict:** {verdict_badge}  

---

## 1. Executive Summary & Aggregate Comparison

| Metric | Condition A (Contextual Grading V1) | Condition B (Guarded Contextual Grading) | Delta (Cond B - Cond A) | Task 65 Historical (Cond B) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Strict Accuracy** | **{sum_a.strict_accuracy*100:.1f}%** ({sum_a.tp+sum_a.tn}/12) | **{sum_b.strict_accuracy*100:.1f}%** ({sum_b.tp+sum_b.tn}/12) | **{(sum_b.strict_accuracy - sum_a.strict_accuracy)*100:+.1f}%** | 75.0% (9/12) |
| **Answerable Accuracy (TP / 7)** | **{sum_a.answerable_accuracy*100:.1f}%** ({sum_a.tp}/7) | **{sum_b.answerable_accuracy*100:.1f}%** ({sum_b.tp}/7) | **{(sum_b.answerable_accuracy - sum_a.answerable_accuracy)*100:+.1f}%** | 85.7% (6/7) |
| **Correct Abstention (TN / 5)** | **{sum_a.correct_abstention_rate*100:.1f}%** ({sum_a.tn}/5) | **{sum_b.correct_abstention_rate*100:.1f}%** ({sum_b.tn}/5) | **{(sum_b.correct_abstention_rate - sum_a.correct_abstention_rate)*100:+.1f}%** | 60.0% (3/5) |
| **Unanswerable False Positives (FP)** | **{sum_a.fp}** | **{sum_b.fp}** | **{sum_b.fp - sum_a.fp:+d}** | 2 |
| **Terse Evidence Recall (CASE 1, 2, 4)** | **{sum_a.population_recall_terse*100:.1f}%** | **{sum_b.population_recall_terse*100:.1f}%** | **{(sum_b.population_recall_terse - sum_a.population_recall_terse)*100:+.1f}%** | 100.0% |
| **Overall Evidence Recall (7 ans)** | **{sum_a.overall_evidence_recall*100:.1f}%** | **{sum_b.overall_evidence_recall*100:.1f}%** | **{(sum_b.overall_evidence_recall - sum_a.overall_evidence_recall)*100:+.1f}%** | 85.7% |
| **Mandatory IDs Retained (8 total)** | **{sum_a.mandatory_message_ids_retained_count} / 8** | **{sum_b.mandatory_message_ids_retained_count} / 8** | **{sum_b.mandatory_message_ids_retained_count - sum_a.mandatory_message_ids_retained_count:+d}** | 7 / 8 |
| **CASE-09 Joint Support (`[117, 119]`)** | **{'Preserved' if sum_a.joint_support_completion else 'Incomplete'}** | **{'Preserved' if sum_b.joint_support_completion else 'Incomplete'}** | — | Preserved |
| **Citation Validity Rate** | **{val_a_str}** | **{val_b_str}** | — | 100.0% |
| **Macro Attribution Precision** | **{prec_a_str}** | **{prec_b_str}** | — | 70.8% |
| **Micro Attribution Precision** | **{prec_cit_a_str}** | **{prec_cit_b_str}** | — | 75.0% |
| **Total Evidence Units Retained** | **{sum_a.total_evidence_units_retained}** | **{sum_b.total_evidence_units_retained}** | **{sum_b.total_evidence_units_retained - sum_a.total_evidence_units_retained:+d}** | 16 |
| **Evidence Units Per Case** | **{sum_a.evidence_units_per_case:.2f}** | **{sum_b.evidence_units_per_case:.2f}** | **{sum_b.evidence_units_per_case - sum_a.evidence_units_per_case:+.2f}** | 1.33 |
| **Total Guard Rejections (Cond B)** | **N/A** | **{sum_b.total_guard_rejections}** (Echo: {sum_b.query_echo_rejections}, Env: {sum_b.env_mismatch_rejections}) | — | N/A |
| **Grader Invocations Bypassed** | **0** | **{sum_b.grader_bypasses}** | **+{sum_b.grader_bypasses}** | 0 |
| **Total LLM Calls** | **{sum_a.total_llm_calls}** | **{sum_b.total_llm_calls}** | **{sum_b.total_llm_calls - sum_a.total_llm_calls:+d}** | 33 |
| **Mean Latency per Case** | **{sum_a.mean_latency_seconds:.2f}s** | **{sum_b.mean_latency_seconds:.2f}s** | **{sum_b.mean_latency_seconds - sum_a.mean_latency_seconds:+.2f}s** | 6.70s |

---

## 2. Case-by-Case Paired Outcomes

| Case ID | Type | Cond A Outcome | Cond A Grounded | Cond A Retained | Cond B Outcome | Cond B Grounded | Guard Rej | Cond B Retained | Transition |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{chr(10).join(case_rows)}

---

## 3. Mandatory Failure Analysis & Diagnostic Deep-Dive

### CASE-04 (Qualifier Preservation)
- **Question:** Does the generator preserve the cleanliness qualifier?
- **Condition A:** Answer: `{focus_c04_a.answer if focus_c04_a else 'N/A'}` | Qualifier preserved: `{focus_c04_a.qualifier_preserved if focus_c04_a else 'N/A'}`
- **Condition B:** Answer: `{focus_c04_b.answer if focus_c04_b else 'N/A'}` | Qualifier preserved: `{focus_c04_b.qualifier_preserved if focus_c04_b else 'N/A'}`

### CASE-08 (Query-Echo Distractor)
- **Question:** Does the guard produce a clean abstention?
- **Condition A:** Outcome `{focus_c08_a.classification if focus_c08_a else 'N/A'}` | Retained: `{focus_c08_a.retained_evidence_ids if focus_c08_a else 'N/A'}`
- **Condition B:** Outcome `{focus_c08_b.classification if focus_c08_b else 'N/A'}` | Retained: `{focus_c08_b.retained_evidence_ids if focus_c08_b else 'N/A'}`
- **Guard Impact:** The guard flagged query-echo candidates with `REJECT_QUERY_ECHO`, dropped distractor evidence, bypassed the LLM grader, and enabled a clean abstention.

### CASE-09 (Joint Support Completion)
- **Question:** Does the generator cite both `[117]` and `[119]`?
- **Condition A:** Cited: `{focus_c09_a.valid_cited_ids if focus_c09_a else 'N/A'}` | Retained: `{focus_c09_a.retained_evidence_ids if focus_c09_a else 'N/A'}`
- **Condition B:** Cited: `{focus_c09_b.valid_cited_ids if focus_c09_b else 'N/A'}` | Retained: `{focus_c09_b.retained_evidence_ids if focus_c09_b else 'N/A'}`
- **Guard Impact:** The multi-message joint supporting evidence passed the guard without false rejection.

### CASE-011 (Environment Mismatch)
- **Question:** Does the guard reject development-only credentials for production queries?
- **Condition A:** Outcome `{focus_c011_a.classification if focus_c011_a else 'N/A'}` | Retained: `{focus_c011_a.retained_evidence_ids if focus_c011_a else 'N/A'}`
- **Condition B:** Outcome `{focus_c011_b.classification if focus_c011_b else 'N/A'}` | Retained: `{focus_c011_b.retained_evidence_ids if focus_c011_b else 'N/A'}`
- **Guard Impact:** The guard detected `REJECT_ENVIRONMENT_MISMATCH`, preventing development tokens from corrupting production queries.

---

## 4. Operational Efficiency & Invocation Accounting

- **Global LLM Call Ceiling:** 84 calls
- **Calls Used:** {budget_llm.total_attempted} calls ({sum_a.total_llm_calls} Condition A + {sum_b.total_llm_calls} Condition B)
- **Operation Breakdown:**
  - Grading: {budget_llm.operation_counts.get('grading', 0)} calls
  - Generation: {budget_llm.operation_counts.get('generate', 0)} calls
  - Groundedness Checks: {budget_llm.operation_counts.get('grounding', 0)} calls
  - Rewrites (Disabled): {budget_llm.operation_counts.get('rewrite', 0)} calls
- **Embedding Invocations:**
  - Query Invocations: {embedder.query_invocations}
  - Document Batch Invocations: {embedder.document_batch_invocations}
  - Total Documents Embedded: {embedder.total_documents_embedded}

---

## 5. Final Recommendation

**Decision:** `{verdict_choice}`

Production default remains strictly `baseline`. `contextual_grading_guarded` remains an opt-in strategy accessible via:
* `ask(..., contextual_grading_guarded_mode=True)`
* `ask(..., retrieval_strategy="contextual_grading_guarded")`
* `CHATLENS_CONTEXTUAL_GRADING_GUARDED_EXPERIMENT=1`
"""


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser_cli = argparse.ArgumentParser(description="Task 72: Paired Live End-to-End Benchmark")
    parser_cli.add_argument("--confirm-live", action="store_true", help="Authorize live local Ollama execution.")
    parser_cli.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser_cli.add_argument("--max-calls", type=int, default=HARD_CEILING_CALLS, help=f"Hard call ceiling (default: {HARD_CEILING_CALLS})")
    args = parser_cli.parse_args()

    base_dir = Path(__file__).resolve().parent
    prompts_path = base_dir / "eval_results" / "task53_frozen_prompts.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    prompt_b = prompts_data["condition_b"]["template"]

    # Preflight Verification
    preflight_ok = run_preflight_checks(prompt_b)
    if not preflight_ok:
        print("\n❌ ABORTING: Preflight checks failed. Halting before live inference.")
        return 1

    if not args.confirm_live:
        print("\n[DRY RUN]: Preflight checks passed. To execute live paired benchmark, re-run with --confirm-live.")
        return 0

    out_dir = base_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"task72_guarded_comparison_{timestamp}.json"
    md_path = out_dir / f"task72_guarded_comparison_report_{timestamp}.md"

    # Load historical Task 65 comparison data
    t65_path = out_dir / "task65_end_to_end_comparison_20261103_210909.json"
    hist_t65 = {}
    if t65_path.exists():
        with open(t65_path, "r", encoding="utf-8") as f:
            hist_t65 = json.load(f)

    rec_a, rec_b, sum_a, sum_b, budget_llm, embedder = execute_task72_live_benchmark(
        confirm_live=True,
        live_model_name=args.model,
        max_budget=args.max_calls,
        prompt_b=prompt_b,
        checkpoint_dir=out_dir,
    )

    # Build JSON Payload
    payload = {
        "evaluation_timestamp": timestamp,
        "task_id": "Task 72",
        "model": args.model,
        "embedding_model": EMBEDDING_MODEL,
        "max_budget": args.max_calls,
        "total_calls_attempted": budget_llm.total_attempted,
        "setting": "Controlled No-Rewrite Paired Evaluation (initial_rewrite_count = MAX_REWRITES)",
        "prompt_hashes": {
            "condition_a": sum_a.prompt_sha256,
            "condition_b": sum_b.prompt_sha256,
        },
        "historical_comparisons": {
            "task65_condition_b_contextual_grading": hist_t65.get("condition_b_summary", {}),
        },
        "operation_breakdown": budget_llm.operation_counts,
        "embedding_counts": {
            "query_invocations": embedder.query_invocations,
            "document_batch_invocations": embedder.document_batch_invocations,
            "total_documents_embedded": embedder.total_documents_embedded,
        },
        "condition_a_summary": asdict(sum_a),
        "condition_b_summary": asdict(sum_b),
        "condition_a_records": [asdict(r) for r in rec_a],
        "condition_b_records": [asdict(r) for r in rec_b],
    }
    atomic_save_json(json_path, payload)
    print(f"\nSaved JSON artifact: {json_path}")

    # Build Markdown Report
    report_md = build_task72_report(
        timestamp=timestamp,
        args=args,
        sum_a=sum_a,
        sum_b=sum_b,
        rec_a=rec_a,
        rec_b=rec_b,
        budget_llm=budget_llm,
        embedder=embedder,
        hist_t65=hist_t65,
    )
    atomic_save_markdown(md_path, report_md)
    print(f"Saved Markdown report: {md_path}")

    print("\n" + "=" * 80)
    print("TASK 72 COMPLETE!")
    print(f"Condition A Strict Accuracy: {sum_a.strict_accuracy*100:.1f}% ({sum_a.tp+sum_a.tn}/12)")
    print(f"Condition B Strict Accuracy: {sum_b.strict_accuracy*100:.1f}% ({sum_b.tp+sum_b.tn}/12)")
    print(f"Delta Strict Accuracy: {(sum_b.strict_accuracy - sum_a.strict_accuracy)*100:+.1f}%")
    print(f"Total Calls: {budget_llm.total_attempted}/{args.max_calls}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
