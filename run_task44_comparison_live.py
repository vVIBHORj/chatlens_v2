"""
run_task44_comparison_live.py
=============================
Task 44: Live Comparison of Dialogue Evidence vs. Contextual Grading.

Evaluates two opt-in retrieval strategies across the 12 canonical evaluation cases:
  - Mode A (Dialogue Evidence):
      retrieve -> assemble and rank dialogue evidence -> generate -> check groundedness
  - Mode B (Contextual Grading):
      retrieve -> assemble dialogue evidence -> grade assembled units -> rank retained units -> generate -> check groundedness

Safety & Budget Constraints:
  - Global hard ceiling: 84 LLM invocations across both modes and all 12 cases.
  - Fail-closed call budget counter intercepting all LLM operations.
  - Strict storage isolation: zero access/modification to production chat_data.db or chroma_db.
  - Fail-closed isolation interceptor raising IsolationBreachError on production access.
  - Requires explicit --confirm-live flag to run live inference (defaults to dry-run).
  - Saves full audit trail to eval_results/ (timestamped JSON and Markdown summary).
"""

import argparse
import copy
import inspect
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

import chat_database
import hybrid_retriever
import rag_graph
import vectorstore
from rag_graph import (
    STRATEGY_BASELINE,
    STRATEGY_CONTEXTUAL_GRADING,
    STRATEGY_DIALOGUE_EVIDENCE,
    TOP_K_EVIDENCE_UNITS,
    ask,
    is_pure_abstention,
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


# ============================================================================
# DATA STRUCTURES FOR TASK 44 COMPARISON
# ============================================================================

@dataclass
class CaseExecutionRecord:
    case_id: str
    mode: str  # "dialogue_evidence" or "contextual_grading"
    query: str
    expected_answerable: bool
    expected_supporting_ids: List[int]
    expected_keywords: List[str]

    # Retrieval & Assembly
    assembled_evidence_ids: List[int] = field(default_factory=list)
    target_in_final_evidence: bool = False

    # Mode B Grading specifics
    units_assembled_count: int = 0
    units_graded_relevant: int = 0
    units_graded_irrelevant: int = 0
    units_graded_uncertain: int = 0
    units_retained_count: int = 0
    grading_decisions: List[Dict[str, Any]] = field(default_factory=list)
    grader_status: str = "unknown"
    grader_failure: bool = False
    distractor_survived: bool = False

    # Generation & Citations
    raw_answer: str = ""
    answer: str = ""
    abstention_citations_stripped: bool = False
    valid_cited_ids: List[int] = field(default_factory=list)
    invalid_cited_ids: List[int] = field(default_factory=list)
    all_citations_valid: bool = False
    abstention_detected: bool = False
    correct_abstention: bool = False

    # Groundedness & Attribution
    grounded: bool = False
    semantic_grounded: bool = False
    attribution_precision: float = 0.0
    attribution_recall: float = 0.0

    # Classification & Exact Fact
    classification: str = "UNKNOWN"  # TP, FP, TN, FN
    passed: bool = False
    exact_fact_preserved: Optional[bool] = None

    # Performance
    latency_seconds: float = 0.0
    llm_calls_made: int = 0
    rewrites_used: int = 0
    error: Optional[str] = None


@dataclass
class ModeComparisonSummary:
    mode_name: str
    total_cases: int = 0
    answerable_cases: int = 0
    unanswerable_cases: int = 0

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    strict_accuracy: float = 0.0
    decisive_accuracy: float = 0.0
    coverage_rate: float = 0.0

    population_recall_terse: float = 0.0  # CASE-01, 02, 04
    overall_evidence_recall: float = 0.0
    citation_validity_rate: float = 0.0
    mean_attribution_precision: float = 0.0
    groundedness_pass_rate: float = 0.0
    correct_abstention_rate: float = 0.0
    false_positives_unanswerable: int = 0
    case12_status: str = "NOT EVALUABLE (case is answerable)"

    exact_fact_preservation_rate: float = 0.0
    total_llm_calls: int = 0
    mean_latency_seconds: float = 0.0
    median_latency_seconds: float = 0.0


# ============================================================================
# SCORING & METRICS HELPERS
# ============================================================================

def evaluate_case_record(record: CaseExecutionRecord, case: EvalCase):
    """Evaluate deterministic evaluation metrics on one case run."""
    ans = record.answer.strip()
    ans_lower = ans.lower()

    # 1. Abstention detection using canonical rules
    record.abstention_detected = is_pure_abstention(ans) or not record.valid_cited_ids

    # 2. Check exact-fact preservation for critical cases
    if case.case_id == "CASE-02-Short-Time":
        # Must preserve 9pm / 9:00 / 21:00
        record.exact_fact_preserved = any(kw.lower() in ans_lower for kw in ["9pm", "9:00", "21:00"])
    elif case.case_id == "CASE-04-Yes-No-Confirmation":
        # Must preserve clean / completely clean qualifier
        record.exact_fact_preserved = "clean" in ans_lower
    elif case.expected_supporting_ids:
        has_all_kw = any(kw.lower() in ans_lower for kw in case.expected_answer_keywords)
        record.exact_fact_preserved = has_all_kw

    # 3. Target evidence retention
    record.target_in_final_evidence = (case.target_id in record.assembled_evidence_ids)

    # 4. Attribution & Classification
    if case.expected_supporting_ids:
        # Answerable Case
        correct_ids = [cid for cid in record.valid_cited_ids if cid in case.expected_supporting_ids]
        record.attribution_recall = len(correct_ids) / len(case.expected_supporting_ids)
        record.attribution_precision = (
            len(correct_ids) / len(record.valid_cited_ids) if record.valid_cited_ids else 0.0
        )
        record.correct_abstention = False

        has_kw = any(kw.lower() in ans_lower for kw in case.expected_answer_keywords)
        if record.grounded and has_kw and record.attribution_precision == 1.0 and record.attribution_recall > 0:
            record.classification = "TP"
            record.passed = True
        else:
            record.classification = "FN"
            record.passed = False
    else:
        # Unanswerable / Distractor Case
        record.correct_abstention = record.abstention_detected and len(record.valid_cited_ids) == 0
        if record.correct_abstention:
            record.classification = "TN"
            record.passed = True
            record.attribution_precision = 1.0
            record.attribution_recall = 1.0
        else:
            record.classification = "FP"
            record.passed = False
            record.attribution_precision = 0.0
            record.attribution_recall = 0.0

    # 5. Distractor tracking for Mode B
    if case.case_id in ("CASE-05-Misleading-Context", "CASE-07-Self-Contained-Irrelevant", "CASE-08-Unrelated-Candidate-Matching-Context", "CASE-011-Similar-Entity-Negative"):
        # Did the distractor message survive in final evidence?
        record.distractor_survived = case.target_id in record.assembled_evidence_ids


def compute_summary(records: List[CaseExecutionRecord], mode_name: str) -> ModeComparisonSummary:
    summary = ModeComparisonSummary(mode_name=mode_name, total_cases=len(records))
    if not records:
        return summary

    summary.answerable_cases = sum(1 for r in records if r.expected_answerable)
    summary.unanswerable_cases = sum(1 for r in records if not r.expected_answerable)

    summary.tp = sum(1 for r in records if r.classification == "TP")
    summary.fp = sum(1 for r in records if r.classification == "FP")
    summary.tn = sum(1 for r in records if r.classification == "TN")
    summary.fn = sum(1 for r in records if r.classification == "FN")

    summary.strict_accuracy = (summary.tp + summary.tn) / len(records)
    decisive = [r for r in records if r.classification in ("TP", "FP", "TN", "FN")]
    summary.decisive_accuracy = (summary.tp + summary.tn) / len(decisive) if decisive else 0.0
    summary.coverage_rate = len(decisive) / len(records)

    # Terse cases: CASE-01, CASE-02, CASE-04
    terse = [r for r in records if r.case_id in ("CASE-01-Terse-Location", "CASE-02-Short-Time", "CASE-04-Yes-No-Confirmation")]
    if terse:
        summary.population_recall_terse = sum(1 for r in terse if r.target_in_final_evidence) / len(terse)

    # Answerable cases evidence recall & grounding
    ans = [r for r in records if r.expected_answerable]
    if ans:
        summary.overall_evidence_recall = sum(1 for r in ans if r.target_in_final_evidence) / len(ans)
        summary.groundedness_pass_rate = sum(1 for r in ans if r.grounded) / len(ans)

    # Unanswerable cases abstention & FP
    unans = [r for r in records if not r.expected_answerable]
    if unans:
        summary.correct_abstention_rate = sum(1 for r in unans if r.correct_abstention) / len(unans)
        summary.false_positives_unanswerable = sum(1 for r in unans if r.classification == "FP")

    # Citations
    cited_runs = [r for r in records if r.valid_cited_ids or r.invalid_cited_ids]
    if cited_runs:
        summary.citation_validity_rate = sum(1 for r in cited_runs if r.all_citations_valid) / len(cited_runs)
    else:
        summary.citation_validity_rate = 1.0

    summary.mean_attribution_precision = sum(r.attribution_precision for r in records) / len(records)

    # Exact facts
    fact_runs = [r for r in records if r.exact_fact_preserved is not None]
    if fact_runs:
        summary.exact_fact_preservation_rate = sum(1 for r in fact_runs if r.exact_fact_preserved) / len(fact_runs)

    summary.total_llm_calls = sum(r.llm_calls_made for r in records)
    latencies = [r.latency_seconds for r in records]
    summary.mean_latency_seconds = sum(latencies) / len(latencies) if latencies else 0.0
    latencies_sorted = sorted(latencies)
    mid = len(latencies_sorted) // 2
    if latencies_sorted:
        summary.median_latency_seconds = (
            (latencies_sorted[mid - 1] + latencies_sorted[mid]) / 2.0
            if len(latencies_sorted) % 2 == 0
            else latencies_sorted[mid]
        )

    return summary


# ============================================================================
# PAIRED COMPARISON RUNNER
# ============================================================================

def execute_task44_comparison(
    live_model_name: str = DEFAULT_MODEL,
    max_budget: int = HARD_CEILING_CALLS,
    confirm_live: bool = False,
) -> Tuple[
    List[CaseExecutionRecord],
    List[CaseExecutionRecord],
    ModeComparisonSummary,
    ModeComparisonSummary,
    GlobalBudgetEnforcingLLM,
    TrackingEmbeddingsWrapper,
]:
    """
    Executes the paired comparison of:
      Mode A: Dialogue Evidence (rank-only)
      Mode B: Contextual Grading (assemble -> grade -> rank)
    across all 12 canonical cases in an isolated sandbox.
    """
    if not confirm_live:
        raise LiveConfirmationRequiredError("Live execution requires explicit confirm_live=True.")

    # 1. Initialize local ChatOllama and global budget counter
    inner_llm = ChatOllama(
        model=live_model_name,
        temperature=0.0,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    budget_llm = GlobalBudgetEnforcingLLM(inner_llm=inner_llm, max_calls=max_budget)

    orig_rag_llm = rag_graph.llm
    rag_graph.llm = budget_llm

    orig_get_embeddings = vectorstore.get_embeddings
    real_embedder = orig_get_embeddings()
    tracking_embedder = TrackingEmbeddingsWrapper(real_embedder)
    vectorstore.get_embeddings = lambda: tracking_embedder

    mode_a_records: List[CaseExecutionRecord] = []
    mode_b_records: List[CaseExecutionRecord] = []

    print("\n" + "=" * 80)
    print("TASK 44 LIVE COMPARISON: DIALOGUE EVIDENCE (MODE A) VS CONTEXTUAL GRADING (MODE B)")
    print(f"Model: {live_model_name} | Embeddings: {EMBEDDING_MODEL} | Ceiling: {max_budget} calls")
    print("=" * 80)

    try:
        for idx, case in enumerate(EVAL_CASES, start=1):
            print(f"\n" + "-" * 75)
            print(f"CASE {idx:02d}/12: [{case.case_id}] ({'ANSWERABLE' if case.expected_answerable else 'UNANSWERABLE'})")
            print(f"Query: \"{case.query}\"")
            print(f"Target ID: {case.target_id} | Expected Supporting: {case.expected_supporting_ids}")
            print("-" * 75)

            with SafeEvaluationSandbox(case) as env:
                # -------------------------------------------------------------
                # 1. MODE A: Dialogue Evidence Pipeline (Rank-Only)
                # -------------------------------------------------------------
                print(f"  [Mode A: Dialogue Evidence] Executing...")
                t0_a = time.time()
                calls_before_a = budget_llm.total_attempted
                rec_a = CaseExecutionRecord(
                    case_id=case.case_id,
                    mode="dialogue_evidence",
                    query=case.query,
                    expected_answerable=case.expected_answerable,
                    expected_supporting_ids=case.expected_supporting_ids,
                    expected_keywords=case.expected_answer_keywords,
                )

                try:
                    res_a = rag_graph.ask(
                        question=case.query,
                        retrieval_strategy=STRATEGY_DIALOGUE_EVIDENCE,
                        database_path=str(env.db_path),
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
                    rec_a.rewrites_used = res_a.get("rewrites_used", 0)
                    val_a = res_a.get("citation_validation")
                    rec_a.all_citations_valid = (
                        val_a.get("all_cited_valid", False) if val_a else (len(rec_a.invalid_cited_ids) == 0)
                    )

                    # Extract assembled unit IDs
                    for s in res_a.get("sources", []):
                        mid = s.get("anchor_message_id") or s.get("message_id")
                        if mid is not None and mid not in rec_a.assembled_evidence_ids:
                            rec_a.assembled_evidence_ids.append(mid)
                        for inmid in s.get("included_message_ids", []):
                            if inmid not in rec_a.assembled_evidence_ids:
                                rec_a.assembled_evidence_ids.append(inmid)

                    evaluate_case_record(rec_a, case)
                    print(
                        f"    -> Result: {rec_a.classification} | Grounded: {rec_a.grounded} | "
                        f"Calls: {rec_a.llm_calls_made} | Cited: {rec_a.valid_cited_ids} | Latency: {rec_a.latency_seconds}s"
                    )

                except Exception as e:
                    rec_a.latency_seconds = round(time.time() - t0_a, 3)
                    rec_a.llm_calls_made = budget_llm.total_attempted - calls_before_a
                    rec_a.error = str(e)
                    print(f"    -> [MODE A ERROR]: {e}")
                    if isinstance(e, (InferenceBudgetExceededError, IsolationBreachError)):
                        raise

                mode_a_records.append(rec_a)

                # -------------------------------------------------------------
                # 2. MODE B: Contextual Grading (Assemble -> Grade -> Rank)
                # -------------------------------------------------------------
                print(f"  [Mode B: Contextual Grading] Executing...")
                t0_b = time.time()
                calls_before_b = budget_llm.total_attempted
                rec_b = CaseExecutionRecord(
                    case_id=case.case_id,
                    mode="contextual_grading",
                    query=case.query,
                    expected_answerable=case.expected_answerable,
                    expected_supporting_ids=case.expected_supporting_ids,
                    expected_keywords=case.expected_answer_keywords,
                )

                try:
                    res_b = rag_graph.ask(
                        question=case.query,
                        retrieval_strategy=STRATEGY_CONTEXTUAL_GRADING,
                        database_path=str(env.db_path),
                    )
                    rec_b.latency_seconds = round(time.time() - t0_b, 3)
                    rec_b.llm_calls_made = budget_llm.total_attempted - calls_before_b
                    rec_b.answer = res_b.get("answer", "")
                    rec_b.raw_answer = res_b.get("raw_answer", rec_b.answer)
                    rec_b.abstention_citations_stripped = res_b.get("abstention_citations_stripped", False)
                    rec_b.grounded = res_b.get("grounded", False)
                    rec_b.semantic_grounded = res_b.get("semantic_grounded", False)
                    rec_b.valid_cited_ids = res_b.get("valid_citation_ids", [])
                    rec_b.invalid_citation_ids = res_b.get("invalid_citation_ids", [])
                    rec_b.rewrites_used = res_b.get("rewrites_used", 0)
                    rec_b.grader_status = res_b.get("grader_status", "unknown")
                    val_b = res_b.get("citation_validation")
                    rec_b.all_citations_valid = (
                        val_b.get("all_cited_valid", False) if val_b else (len(rec_b.invalid_cited_ids) == 0)
                    )

                    # Extract contextual grading decisions & assembled unit IDs
                    grading_decs = res_b.get("contextual_grading_decisions", [])
                    rec_b.grading_decisions = grading_decs
                    rec_b.units_assembled_count = len(grading_decs)
                    rec_b.units_graded_relevant = sum(1 for d in grading_decs if d.get("decision") == "relevant")
                    rec_b.units_graded_irrelevant = sum(1 for d in grading_decs if d.get("decision") == "irrelevant")
                    rec_b.units_graded_uncertain = sum(1 for d in grading_decs if d.get("decision") == "uncertain")
                    rec_b.grader_failure = any(d.get("grader_failure", False) for d in grading_decs)

                    for s in res_b.get("sources", []):
                        mid = s.get("anchor_message_id") or s.get("message_id")
                        if mid is not None and mid not in rec_b.assembled_evidence_ids:
                            rec_b.assembled_evidence_ids.append(mid)
                        for inmid in s.get("included_message_ids", []):
                            if inmid not in rec_b.assembled_evidence_ids:
                                rec_b.assembled_evidence_ids.append(inmid)

                    rec_b.units_retained_count = len(res_b.get("sources", []))

                    evaluate_case_record(rec_b, case)
                    print(
                        f"    -> Result: {rec_b.classification} | Grounded: {rec_b.grounded} | "
                        f"Calls: {rec_b.llm_calls_made} | Graded: {rec_b.units_assembled_count} (rel={rec_b.units_graded_relevant}, irrel={rec_b.units_graded_irrelevant}, unc={rec_b.units_graded_uncertain}) | "
                        f"Cited: {rec_b.valid_cited_ids} | Latency: {rec_b.latency_seconds}s"
                    )

                except Exception as e:
                    rec_b.latency_seconds = round(time.time() - t0_b, 3)
                    rec_b.llm_calls_made = budget_llm.total_attempted - calls_before_b
                    rec_b.error = str(e)
                    print(f"    -> [MODE B ERROR]: {e}")
                    if isinstance(e, (InferenceBudgetExceededError, IsolationBreachError)):
                        raise

                mode_b_records.append(rec_b)

    finally:
        rag_graph.llm = orig_rag_llm
        vectorstore.get_embeddings = orig_get_embeddings

    summary_a = compute_summary(mode_a_records, "Mode A (Dialogue Evidence)")
    summary_b = compute_summary(mode_b_records, "Mode B (Contextual Grading)")

    return mode_a_records, mode_b_records, summary_a, summary_b, budget_llm, tracking_embedder


# ============================================================================
# CLI ENTRY POINT & ARTIFACT GENERATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Task 44: Live Comparison of Dialogue Evidence vs Contextual Grading")
    parser.add_argument("--confirm-live", action="store_true", help="Authorize local Ollama inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (default: qwen2.5:3b)")
    parser.add_argument("--max-calls", type=int, default=HARD_CEILING_CALLS, help="Hard call ceiling (default: 84)")
    args = parser.parse_args()

    if not args.confirm_live:
        print("\n" + "=" * 80)
        print("[DRY RUN DEFAULT] Live inference was NOT authorized with --confirm-live.")
        print("Preflight Case Manifest (12 Canonical Cases):")
        print("=" * 80)
        for idx, c in enumerate(EVAL_CASES, 1):
            status_str = "ANSWERABLE" if c.expected_answerable else "UNANSWERABLE"
            print(f"  {idx:02d}. [{c.case_id}] {c.category} ({status_str})")
            print(f"      Query: \"{c.query}\"")
            print(f"      Target ID: {c.target_id} | Expected Supporting IDs: {c.expected_supporting_ids}")

        print("\nStrategy Selection Verification:")
        print("  - Mode A: retrieval_strategy='dialogue_evidence'")
        print("  - Mode B: retrieval_strategy='contextual_grading'")
        print(f"Planning Call Budget: ~60-72 calls | Hard Ceiling: {args.max_calls}")
        print("\nTo execute live inference safely in the isolated sandbox, supply --confirm-live.")
        return 0

    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"task44_comparison_{ts}.json"
    out_md = out_dir / f"task44_comparison_report_{ts}.md"

    recs_a, recs_b, sum_a, sum_b, budget_tracker, embed_tracker = execute_task44_comparison(
        live_model_name=args.model,
        max_budget=args.max_calls,
        confirm_live=True,
    )

    # Compile JSON report
    report_dict = {
        "evaluation_timestamp": ts,
        "model": args.model,
        "embedding_model": EMBEDDING_MODEL,
        "max_budget": args.max_calls,
        "total_calls_attempted": budget_tracker.total_attempted,
        "operation_breakdown": budget_tracker.operation_counts,
        "embedding_counts": {
            "query_invocations": embed_tracker.query_invocations,
            "document_batch_invocations": embed_tracker.document_batch_invocations,
            "total_documents_embedded": embed_tracker.total_documents_embedded,
        },
        "mode_a_summary": asdict(sum_a),
        "mode_b_summary": asdict(sum_b),
        "mode_a_records": [asdict(r) for r in recs_a],
        "mode_b_records": [asdict(r) for r in recs_b],
    }

    out_json.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"\n✓ Saved machine-readable audit trail to: {out_json}")

    # Build readable Markdown summary
    md_lines = [
        f"# Task 44 Live Evaluation Report: Dialogue Evidence vs. Contextual Grading",
        f"**Timestamp:** {ts} | **Model:** `{args.model}` | **Embedding Model:** `{EMBEDDING_MODEL}`",
        f"**Global Call Budget:** {budget_tracker.total_attempted}/{args.max_calls} calls (Headroom remaining: {args.max_calls - budget_tracker.total_attempted})",
        "",
        "## 1. Summary Metrics Comparison",
        "",
        "| Metric | Mode A (Dialogue Evidence) | Mode B (Contextual Grading) | Delta / Assessment |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Overall Strict Accuracy** | {sum_a.strict_accuracy:.1%} ({sum_a.tp + sum_a.tn}/12) | {sum_b.strict_accuracy:.1%} ({sum_b.tp + sum_b.tn}/12) | {'+' if sum_b.strict_accuracy >= sum_a.strict_accuracy else ''}{sum_b.strict_accuracy - sum_a.strict_accuracy:.1%} |",
        f"| **Decisive Accuracy** | {sum_a.decisive_accuracy:.1%} | {sum_b.decisive_accuracy:.1%} | {'+' if sum_b.decisive_accuracy >= sum_a.decisive_accuracy else ''}{sum_b.decisive_accuracy - sum_a.decisive_accuracy:.1%} |",
        f"| **Terse Evidence Recall (CASE 1,2,4)** | {sum_a.population_recall_terse:.1%} | {sum_b.population_recall_terse:.1%} | {'+' if sum_b.population_recall_terse >= sum_a.population_recall_terse else ''}{sum_b.population_recall_terse - sum_a.population_recall_terse:.1%} |",
        f"| **Overall Evidence Recall (7 answerable)** | {sum_a.overall_evidence_recall:.1%} | {sum_b.overall_evidence_recall:.1%} | {'+' if sum_b.overall_evidence_recall >= sum_a.overall_evidence_recall else ''}{sum_b.overall_evidence_recall - sum_a.overall_evidence_recall:.1%} |",
        f"| **False Positives (5 unanswerable)** | {sum_a.false_positives_unanswerable}/5 | {sum_b.false_positives_unanswerable}/5 | {'Resolved' if sum_b.false_positives_unanswerable < sum_a.false_positives_unanswerable else 'Unchanged'} |",
        f"| **Correct Abstention Rate** | {sum_a.correct_abstention_rate:.1%} | {sum_b.correct_abstention_rate:.1%} | {'+' if sum_b.correct_abstention_rate >= sum_a.correct_abstention_rate else ''}{sum_b.correct_abstention_rate - sum_a.correct_abstention_rate:.1%} |",
        f"| **Citation Validity Rate** | {sum_a.citation_validity_rate:.1%} | {sum_b.citation_validity_rate:.1%} | 100% Target |",
        f"| **Mean Attribution Precision** | {sum_a.mean_attribution_precision:.3f} | {sum_b.mean_attribution_precision:.3f} | |",
        f"| **Groundedness Pass Rate (Answerable)** | {sum_a.groundedness_pass_rate:.1%} | {sum_b.groundedness_pass_rate:.1%} | |",
        f"| **Exact-Fact Preservation Rate** | {sum_a.exact_fact_preservation_rate:.1%} | {sum_b.exact_fact_preservation_rate:.1%} | |",
        f"| **Total LLM Invocations** | {sum_a.total_llm_calls} calls | {sum_b.total_llm_calls} calls | +{sum_b.total_llm_calls - sum_a.total_llm_calls} calls for grading |",
        f"| **Mean Latency / Case** | {sum_a.mean_latency_seconds:.2f}s | {sum_b.mean_latency_seconds:.2f}s | |",
        f"| **Median Latency / Case** | {sum_a.median_latency_seconds:.2f}s | {sum_b.median_latency_seconds:.2f}s | |",
        "",
        "## 2. Operation Call Breakdown",
        f"- **Grading calls:** {budget_tracker.operation_counts.get('grading', 0)}",
        f"- **Generation calls:** {budget_tracker.operation_counts.get('generate', 0)}",
        f"- **Groundedness calls:** {budget_tracker.operation_counts.get('grounding', 0)}",
        f"- **Query rewrite calls:** {budget_tracker.operation_counts.get('rewrite', 0)}",
        f"- **Total LLM calls executed:** {budget_tracker.total_attempted} (Ceiling: {args.max_calls})",
        f"- **Embedding operations:** {embed_tracker.query_invocations} query embeds, {embed_tracker.document_batch_invocations} doc batch embeds ({embed_tracker.total_documents_embedded} docs)",
        "",
        "## 3. Case-by-Case Outcomes Table",
        "",
        "| Case ID | Expected | Mode A Class | Mode A Grounded | Mode B Class | Mode B Grounded | Mode B Graded (Rel/Irrel/Unc) | Mode B Distractor Filtered |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for ra, rb in zip(recs_a, recs_b):
        exp_str = "Answerable" if ra.expected_answerable else "Unanswerable"
        graded_breakdown = f"{rb.units_graded_relevant}/{rb.units_graded_irrelevant}/{rb.units_graded_uncertain}"
        distractor_filtered = "N/A" if ra.expected_answerable else ("YES" if not rb.distractor_survived else "NO (Survived)")
        md_lines.append(
            f"| `{ra.case_id}` | {exp_str} | **{ra.classification}** | {ra.grounded} | **{rb.classification}** | {rb.grounded} | {graded_breakdown} | {distractor_filtered} |"
        )

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"✓ Saved Markdown evaluation report to: {out_md}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

