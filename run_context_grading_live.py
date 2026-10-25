"""
run_context_grading_live.py
===========================
Multi-Case Live Relevance Grading Evaluation Runner for ChatLens-RAG.

Evaluates real-model relevance grading behavior across the 12 deterministic
synthetic evaluation cases from test_context_grading_evaluation.py under two
strictly controlled conditions:
  - Condition A (Context-Aware): Preceding conversational context supplied.
  - Condition B (Standalone Baseline): Preceding context lookup omitted.

Safety & Execution Guard:
  - Defaults to DRY-RUN mode (zero model calls, zero network/Ollama access).
  - Explicit --confirm-live flag is REQUIRED to execute live inference.
  - Planned inference-call budget enforcement (default limit: 24 calls).
  - Pre-execution budget validation: aborts before running if planned calls exceed limit.
  - Strictly isolated sandbox: rejects production chat_data.db and chroma_db.
  - Guarantees teardown and restoration of function defaults and storage paths.
"""

import argparse
import copy
import inspect
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import chat_database
import hybrid_retriever
import rag_graph
import vectorstore
from rag_graph import (
    format_candidate_for_grader,
    grade_candidate_batch,
    parse_grader_response,
)
from test_context_grading_evaluation import (
    EVALUATION_CASES,
    ContextGradingCase,
)


DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_MAX_CALLS = 24
DEFAULT_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------

class IsolationBreachError(Exception):
    """Raised when an operation attempts to access storage outside the sandbox."""
    pass


class InferenceBudgetExceededError(Exception):
    """Raised when planned inference calls exceed the authorized call budget."""
    pass


class LiveConfirmationRequiredError(Exception):
    """Raised when live execution is attempted without explicit user confirmation."""
    pass


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class CaseExecutionResult:
    case_id: str
    category: str
    query: str
    candidate_id: int
    candidate_text: str
    expected_label: str  # "relevant" or "irrelevant"
    condition: str  # "context_aware" or "standalone"
    prompt: str
    has_preceding_context: bool
    raw_output: str
    parsed_decision: str  # "relevant", "irrelevant", or "uncertain"
    reasoning: str
    status: str  # "success" or "failure"
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class ConditionSummaryMetrics:
    condition_name: str
    total_cases: int
    decisive_cases: int
    abstention_cases: int
    coverage_rate: float
    abstention_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    decisive_accuracy: float
    false_positive_rate: float
    strict_accuracy: float
    infra_failures: int


@dataclass
class MultiCaseEvaluationReport:
    model_name: str
    cases_evaluated: int
    condition_a_results: List[CaseExecutionResult]
    condition_b_results: List[CaseExecutionResult]
    metrics_a: ConditionSummaryMetrics
    metrics_b: ConditionSummaryMetrics
    recovered_cases: List[str]
    broken_decisions: List[str]
    invariant_cases: List[str]
    infra_failures_count: int
    attempted_calls: int = 0
    max_budget: int = DEFAULT_MAX_CALLS
    is_complete: bool = True
    termination_reason: Optional[str] = None


# ---------------------------------------------------------------------
# Budget-Enforcing & Prompt-Capturing Wrapper
# ---------------------------------------------------------------------

class BudgetEnforcingLLMWrapper:
    """
    Enforces a strict runtime call budget on a grader LLM and captures prompts.
    Immediately checks the budget before each invocation and increments attempted
    calls regardless of whether the call succeeds or raises an exception.
    """
    def __init__(self, inner_llm: Any, max_calls: int = DEFAULT_MAX_CALLS):
        self.inner_llm = inner_llm
        self.max_calls = max_calls
        self.attempted_calls: int = 0
        self.captured_prompts: List[str] = []
        self.exhausted: bool = False

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        # 1. Check whether call budget is exhausted
        if self.attempted_calls >= self.max_calls:
            self.exhausted = True
            raise InferenceBudgetExceededError(
                f"Inference budget exhausted: attempted {self.attempted_calls}/{self.max_calls} calls. "
                "Call rejected before invoking model."
            )
        # 2. Increment attempted-call count BEFORE invoking model
        self.attempted_calls += 1
        self.captured_prompts.append(str(prompt))

        # 3. Invoke underlying model exactly once
        return self.inner_llm.invoke(prompt, *args, **kwargs)


class PromptCapturingLLMWrapper(BudgetEnforcingLLMWrapper):
    """Backwards-compatible alias for BudgetEnforcingLLMWrapper."""
    pass


# ---------------------------------------------------------------------
# Fail-Closed Isolated Sandbox Environment
# ---------------------------------------------------------------------

class SafeContextGradingEnvironment:
    """
    Manages an isolated temporary SQLite database populated with the 12
    synthetic evaluation cases. Installs fail-closed guards blocking access
    to production chat_data.db and chroma_db.
    """
    def __init__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="chatlens_context_eval_safe_")
        self.tmp_path = Path(self.tmp_dir).resolve()
        self.db_path = self.tmp_path / "eval_sandbox.db"
        self.chroma_path = self.tmp_path / "eval_chroma"

        # Production path references for guard assertion
        self.prod_db_path = Path("./chat_data.db").resolve()
        self.prod_chroma_path = Path("./chroma_db").resolve()

        # Original bindings to restore
        self.orig_db_path = chat_database.DATABASE_PATH
        self.orig_chroma_persist_dir = vectorstore.PERSIST_DIR
        self.orig_get_connection = chat_database.get_connection
        self.orig_load_msg = hybrid_retriever.load_message_vectorstore
        self.orig_load_chunk = hybrid_retriever.load_chunk_vectorstore
        self.orig_vec_load_msg = vectorstore.load_message_vectorstore
        self.orig_vec_load_chunk = vectorstore.load_chunk_vectorstore
        self.orig_defaults: Dict[Any, Tuple[Any, Any]] = {}

        self.is_active = False

    def setup(self):
        """Populates the isolated SQLite database with synthetic test cases."""
        try:
            self.activate()
            # Initialize schema in the temporary database
            chat_database.create_database(str(self.db_path))

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            for case in EVALUATION_CASES:
                # Insert preceding messages for context expansion
                for m in case.preceding_messages:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO messages (
                            id, timestamp, sender, message, message_type, is_question, message_length
                        ) VALUES (?, ?, ?, ?, 'chat', 0, ?)
                        """,
                        (m["id"], m["timestamp"], m["sender"], m["message"], len(m["message"].split())),
                    )

                # Insert the candidate message itself
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO messages (
                        id, timestamp, sender, message, message_type, is_question, message_length
                    ) VALUES (?, ?, ?, ?, 'chat', 0, ?)
                    """,
                    (
                        case.candidate_id,
                        case.timestamp,
                        case.sender,
                        case.candidate_text,
                        len(case.candidate_text.split()),
                    ),
                )

            conn.commit()
            conn.close()
        except Exception:
            self.teardown()
            raise

    def activate(self):
        """Installs fail-closed interceptors for database and vectorstore access."""
        target_db = str(self.db_path)
        target_chroma = str(self.chroma_path)
        resolved_tmp = self.tmp_path

        chat_database.DATABASE_PATH = target_db
        vectorstore.PERSIST_DIR = target_chroma

        # Dynamically update defaults on chat_database functions
        for name, fn in inspect.getmembers(chat_database, inspect.isfunction):
            if fn.__module__ == "chat_database" and fn.__defaults__:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                default_offset = len(params) - len(fn.__defaults__)
                new_defaults = []
                has_change = False
                for idx, def_val in enumerate(fn.__defaults__):
                    p = params[default_offset + idx]
                    if p.name == "database_path" and (
                        def_val == self.orig_db_path or def_val == "./chat_data.db"
                    ):
                        new_defaults.append(target_db)
                        has_change = True
                    else:
                        new_defaults.append(def_val)
                if has_change and fn not in self.orig_defaults:
                    installed_tuple = tuple(new_defaults)
                    self.orig_defaults[fn] = (fn.__defaults__, installed_tuple)
                    fn.__defaults__ = installed_tuple

        # Fail-closed SQLite interceptor
        def fail_closed_get_connection(database_path=None):
            if database_path is None:
                chosen_str = target_db
            else:
                p_str = str(database_path).replace("\\", "/")
                resolved = Path(str(database_path)).resolve()
                if (
                    resolved == self.prod_db_path
                    or p_str in ("./chat_data.db", "chat_data.db")
                    or resolved.name == "chat_data.db"
                ):
                    raise IsolationBreachError(
                        f"FAIL-CLOSED BREACH: Explicit access attempted to production database: {database_path}"
                    )
                chosen_str = str(database_path)

            resolved_chosen = Path(chosen_str).resolve()
            if not resolved_chosen.is_relative_to(resolved_tmp):
                raise IsolationBreachError(
                    f"FAIL-CLOSED BREACH: Access attempted outside isolated sandbox: {resolved_chosen}"
                )
            return self.orig_get_connection(str(resolved_chosen))

        chat_database.get_connection = fail_closed_get_connection
        globals()["get_connection"] = fail_closed_get_connection

        # Fail-closed Chroma path validator and interceptors
        def _validate_chroma_path(persist_dir: Optional[str] = None) -> Path:
            if persist_dir is None:
                resolved = Path(target_chroma).resolve()
            else:
                p_str = str(persist_dir).replace("\\", "/")
                resolved = Path(persist_dir).resolve()
                if (
                    resolved == self.prod_chroma_path
                    or p_str in ("./chroma_db", "chroma_db")
                    or resolved.name == "chroma_db"
                ):
                    raise IsolationBreachError(
                        f"FAIL-CLOSED BREACH: Explicit access attempted to production Chroma: {persist_dir}"
                    )
            if not resolved.is_relative_to(resolved_tmp):
                raise IsolationBreachError(
                    f"FAIL-CLOSED BREACH: Chroma access attempted outside sandbox: {resolved}"
                )
            return resolved

        def fail_closed_mock_store(persist_dir=None):
            _validate_chroma_path(persist_dir)
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = []
            mock_store.similarity_search_with_relevance_scores.return_value = []
            mock_store.as_retriever.return_value = MagicMock()
            return mock_store

        hybrid_retriever.load_message_vectorstore = fail_closed_mock_store
        hybrid_retriever.load_chunk_vectorstore = fail_closed_mock_store
        vectorstore.load_message_vectorstore = fail_closed_mock_store
        vectorstore.load_chunk_vectorstore = fail_closed_mock_store

        self.is_active = True
        self.assert_fail_closed_isolation()

    def assert_fail_closed_isolation(self):
        """Verifies active storage configuration resolves strictly inside sandbox."""
        active_db = Path(chat_database.DATABASE_PATH).resolve()
        active_chroma = Path(vectorstore.PERSIST_DIR).resolve()

        if not active_db.is_relative_to(self.tmp_path):
            raise IsolationBreachError(
                f"Isolation check failed: database {active_db} is outside {self.tmp_path}"
            )
        if not active_chroma.is_relative_to(self.tmp_path):
            raise IsolationBreachError(
                f"Isolation check failed: chroma {active_chroma} is outside {self.tmp_path}"
            )
        if active_db == self.prod_db_path:
            raise IsolationBreachError("Isolation check failed: points to production chat_data.db")
        if active_chroma == self.prod_chroma_path:
            raise IsolationBreachError("Isolation check failed: points to production chroma_db")

    def teardown(self):
        """Restores original module bindings and cleans up temporary files."""
        chat_database.DATABASE_PATH = self.orig_db_path
        vectorstore.PERSIST_DIR = self.orig_chroma_persist_dir
        chat_database.get_connection = self.orig_get_connection
        globals()["get_connection"] = self.orig_get_connection
        hybrid_retriever.load_message_vectorstore = self.orig_load_msg
        hybrid_retriever.load_chunk_vectorstore = self.orig_load_chunk
        vectorstore.load_message_vectorstore = self.orig_vec_load_msg
        vectorstore.load_chunk_vectorstore = self.orig_vec_load_chunk

        for fn, (orig, installed) in self.orig_defaults.items():
            try:
                if fn.__defaults__ == installed:
                    fn.__defaults__ = orig
            except Exception:
                pass
        self.orig_defaults.clear()

        self.is_active = False

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()


# ---------------------------------------------------------------------
# Candidate Document Construction
# ---------------------------------------------------------------------

def build_candidate_document(case: ContextGradingCase) -> Document:
    """
    Builds the standardized candidate document for a synthetic case.
    Guarantees identical representation between Condition A and Condition B.
    """
    clean_text = f"[ID {case.candidate_id}] [{case.timestamp}] {case.sender}: {case.candidate_text}"
    return Document(
        page_content=clean_text,
        metadata={
            "source": "hybrid_search",
            "message_id": case.candidate_id,
            "timestamp": case.timestamp,
            "sender": case.sender,
            "hybrid_score": 0.85,
        },
    )


# ---------------------------------------------------------------------
# Single Case Evaluation
# ---------------------------------------------------------------------

def evaluate_case_condition(
    case: ContextGradingCase,
    condition: str,  # "context_aware" or "standalone"
    grader_llm: Any,
    database_path: str,
) -> CaseExecutionResult:
    """
    Evaluates a single synthetic case under Condition A or Condition B using
    the real rag_graph.grade_candidate_batch() pipeline.
    """
    doc = build_candidate_document(case)

    # Use existing BudgetEnforcingLLMWrapper if provided, else create a local instance
    if isinstance(grader_llm, BudgetEnforcingLLMWrapper):
        wrapper = grader_llm
    else:
        wrapper = BudgetEnforcingLLMWrapper(grader_llm, max_calls=DEFAULT_MAX_CALLS)

    prompt_count_before = len(wrapper.captured_prompts)
    t0 = time.time()
    raw_output = ""
    error_msg = None

    if condition == "context_aware":
        graded_batch = grade_candidate_batch(
            question=case.query,
            batch_docs=[doc],
            grader_llm=wrapper,
            database_path=database_path,
        )
    elif condition == "standalone":
        # Suppress preceding context lookup strictly for Condition B
        with patch("rag_graph.get_grading_preceding_context", return_value=None):
            graded_batch = grade_candidate_batch(
                question=case.query,
                batch_docs=[doc],
                grader_llm=wrapper,
                database_path=database_path,
            )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    latency = time.time() - t0

    # Intercept budget exhaustion immediately so it is not masked as normal parsing fallback
    if getattr(wrapper, "exhausted", False):
        raise InferenceBudgetExceededError(
            f"Inference budget exhausted ({wrapper.attempted_calls}/{wrapper.max_calls} calls attempted)."
        )

    # Capture prompt passed during this evaluation
    if len(wrapper.captured_prompts) > prompt_count_before:
        prompt = wrapper.captured_prompts[prompt_count_before]
    else:
        prompt = ""
    has_preceding_context = "Preceding Conversational Context:" in prompt

    if graded_batch:
        _, decision, reason, status = graded_batch[0]
        if "InferenceBudgetExceededError" in str(reason) or "Inference budget exhausted" in str(reason):
            raise InferenceBudgetExceededError(f"Budget exhausted during grading: {reason}")
    else:
        decision = "uncertain"
        reason = "Empty batch result from grade_candidate_batch"
        status = "failure"

    return CaseExecutionResult(
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        candidate_id=case.candidate_id,
        candidate_text=case.candidate_text,
        expected_label=case.expected_label,
        condition=condition,
        prompt=prompt,
        has_preceding_context=has_preceding_context,
        raw_output=raw_output,
        parsed_decision=decision,
        reasoning=reason,
        status=status,
        latency_seconds=latency,
        error=error_msg,
    )


# ---------------------------------------------------------------------
# Metrics Calculation
# ---------------------------------------------------------------------

def compute_condition_metrics(
    condition_name: str,
    results: List[CaseExecutionResult],
) -> ConditionSummaryMetrics:
    """
    Computes binary metrics, abstention rates, and coverage with explicit denominators.
    """
    total = len(results)
    decisive_results = [r for r in results if r.parsed_decision in ("relevant", "irrelevant")]
    decisive_count = len(decisive_results)
    abstention_count = total - decisive_count

    coverage = decisive_count / total if total > 0 else 0.0
    abstention_rate = abstention_count / total if total > 0 else 0.0

    tp = sum(1 for r in decisive_results if r.parsed_decision == "relevant" and r.expected_label == "relevant")
    fp = sum(1 for r in decisive_results if r.parsed_decision == "relevant" and r.expected_label == "irrelevant")
    tn = sum(1 for r in decisive_results if r.parsed_decision == "irrelevant" and r.expected_label == "irrelevant")
    fn = sum(1 for r in decisive_results if r.parsed_decision == "irrelevant" and r.expected_label == "relevant")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    decisive_acc = (tp + tn) / decisive_count if decisive_count > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    strict_acc = (tp + tn) / total if total > 0 else 0.0
    infra_failures = sum(1 for r in results if r.status == "failure")

    return ConditionSummaryMetrics(
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
        decisive_accuracy=decisive_acc,
        false_positive_rate=fpr,
        strict_accuracy=strict_acc,
        infra_failures=infra_failures,
    )


# ---------------------------------------------------------------------
# Full Suite Execution (Live or Mocked)
# ---------------------------------------------------------------------

def run_context_grading_suite(
    grader_llm: Any,
    env: SafeContextGradingEnvironment,
    cases: Optional[List[ContextGradingCase]] = None,
    max_budget: Optional[int] = None,
) -> MultiCaseEvaluationReport:
    """
    Executes the 12-case A/B evaluation suite against the provided LLM instance.
    Enforces a single shared BudgetEnforcingLLMWrapper across all cases and conditions.
    """
    eval_cases = cases or EVALUATION_CASES
    results_a: List[CaseExecutionResult] = []
    results_b: List[CaseExecutionResult] = []

    # Single shared wrapper enforcing the overall run budget across all cases and conditions
    if isinstance(grader_llm, BudgetEnforcingLLMWrapper):
        enforcing_wrapper = grader_llm
    else:
        budget = max_budget if max_budget is not None else DEFAULT_MAX_CALLS
        enforcing_wrapper = BudgetEnforcingLLMWrapper(inner_llm=grader_llm, max_calls=budget)

    is_complete = True
    termination_reason = None

    try:
        for case in eval_cases:
            res_a = evaluate_case_condition(
                case=case,
                condition="context_aware",
                grader_llm=enforcing_wrapper,
                database_path=str(env.db_path),
            )
            results_a.append(res_a)

            res_b = evaluate_case_condition(
                case=case,
                condition="standalone",
                grader_llm=enforcing_wrapper,
                database_path=str(env.db_path),
            )
            results_b.append(res_b)
    except InferenceBudgetExceededError as exc:
        is_complete = False
        termination_reason = str(exc)

    metrics_a = compute_condition_metrics("Context-Aware (Condition A)", results_a)
    metrics_b = compute_condition_metrics("Standalone Baseline (Condition B)", results_b)

    recovered = []
    broken = []
    invariant = []

    for ra, rb in zip(results_a, results_b):
        # Case recovered: standalone was incorrect, context-aware is correct
        if rb.parsed_decision != rb.expected_label and ra.parsed_decision == ra.expected_label:
            recovered.append(ra.case_id)
        # Decision broken: standalone was correct, context-aware made an error
        elif rb.parsed_decision == rb.expected_label and ra.parsed_decision != ra.expected_label:
            broken.append(ra.case_id)
        else:
            invariant.append(ra.case_id)

    total_infra_failures = metrics_a.infra_failures + metrics_b.infra_failures
    inner_model = getattr(enforcing_wrapper.inner_llm, "model", str(type(enforcing_wrapper.inner_llm).__name__))

    return MultiCaseEvaluationReport(
        model_name=inner_model,
        cases_evaluated=len(results_a),
        condition_a_results=results_a,
        condition_b_results=results_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        recovered_cases=recovered,
        broken_decisions=broken,
        invariant_cases=invariant,
        infra_failures_count=total_infra_failures,
        attempted_calls=enforcing_wrapper.attempted_calls,
        max_budget=enforcing_wrapper.max_calls,
        is_complete=is_complete,
        termination_reason=termination_reason,
    )


# ---------------------------------------------------------------------
# Formatting & Reporting
# ---------------------------------------------------------------------

def format_evaluation_summary(report: MultiCaseEvaluationReport) -> str:
    """Formats markdown results summary table and metrics."""
    ma = report.metrics_a
    mb = report.metrics_b

    status_str = (
        f"COMPLETE (All {report.cases_evaluated} cases evaluated | {report.attempted_calls}/{report.max_budget} calls used)"
        if report.is_complete
        else f"INCOMPLETE ({report.attempted_calls}/{report.max_budget} calls used)"
    )

    lines = [
        "## ChatLens-RAG Multi-Case Relevance Grading Evaluation",
        f"**Model:** `{report.model_name}` | **Cases Evaluated:** {report.cases_evaluated} | **Status:** {status_str}",
        "",
    ]

    if not report.is_complete:
        lines.extend([
            "> [!WARNING]",
            f"> **Inference Budget Exhausted:** {report.termination_reason}",
            f"> Evaluation aborted mid-suite. The metrics below reflect only the {report.cases_evaluated} cases completed prior to budget exhaustion.",
            "",
        ])

    lines.extend([
        "### Case-by-Case Results",
        "| Case ID | Category | Expected | Standalone (B) | Context-Aware (A) | Preceding Context | Transition |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---|",
    ])

    for ra, rb in zip(report.condition_a_results, report.condition_b_results):
        if ra.case_id in report.recovered_cases:
            transition = "**Recovered (+1 TP)**"
        elif ra.case_id in report.broken_decisions:
            transition = "**BROKEN (Regression)**"
        else:
            transition = "Invariant"

        has_ctx = "Yes" if ra.has_preceding_context else "No"
        lines.append(
            f"| `{ra.case_id}` | {ra.category} | `{ra.expected_label}` | `{rb.parsed_decision}` | `{ra.parsed_decision}` | {has_ctx} | {transition} |"
        )

    lines.extend([
        "",
        "### Summary Metrics Comparison",
        "| Metric | Standalone Baseline (B) | Context-Aware (A) | Delta (A vs B) |",
        "|:---|:---:|:---:|:---:|",
        f"| **Decisive Coverage** | {mb.coverage_rate:.2%} ({mb.decisive_cases}/{mb.total_cases}) | {ma.coverage_rate:.2%} ({ma.decisive_cases}/{ma.total_cases}) | {ma.coverage_rate - mb.coverage_rate:+.2%} |",
        f"| **Abstention Rate** | {mb.abstention_rate:.2%} ({mb.abstention_cases}/{mb.total_cases}) | {ma.abstention_rate:.2%} ({ma.abstention_cases}/{ma.total_cases}) | {ma.abstention_rate - mb.abstention_rate:+.2%} |",
        f"| **True Positives (TP)** | {mb.tp} | {ma.tp} | {ma.tp - mb.tp:+d} |",
        f"| **False Negatives (FN)** | {mb.fn} | {ma.fn} | {ma.fn - mb.fn:+d} |",
        f"| **True Negatives (TN)** | {mb.tn} | {ma.tn} | {ma.tn - mb.tn:+d} |",
        f"| **False Positives (FP)** | {mb.fp} | {ma.fp} | {ma.fp - mb.fp:+d} |",
        f"| **Precision** | {mb.precision:.2%} ({mb.tp}/{mb.tp + mb.fp if mb.tp + mb.fp > 0 else 1}) | {ma.precision:.2%} ({ma.tp}/{ma.tp + ma.fp if ma.tp + ma.fp > 0 else 1}) | {ma.precision - mb.precision:+.2%} |",
        f"| **Recall** | {mb.recall:.2%} ({mb.tp}/{mb.tp + mb.fn if mb.tp + mb.fn > 0 else 1}) | {ma.recall:.2%} ({ma.tp}/{ma.tp + ma.fn if ma.tp + ma.fn > 0 else 1}) | {ma.recall - mb.recall:+.2%} |",
        f"| **F1 Score** | {mb.f1:.4f} | {ma.f1:.4f} | {ma.f1 - mb.f1:+.4f} |",
        f"| **Decisive Accuracy** | {mb.decisive_accuracy:.2%} ({mb.tp + mb.tn}/{mb.decisive_cases if mb.decisive_cases > 0 else 1}) | {ma.decisive_accuracy:.2%} ({ma.tp + ma.tn}/{ma.decisive_cases if ma.decisive_cases > 0 else 1}) | {ma.decisive_accuracy - mb.decisive_accuracy:+.2%} |",
        f"| **False Positive Rate** | {mb.false_positive_rate:.2%} ({mb.fp}/{mb.fp + mb.tn if mb.fp + mb.tn > 0 else 1}) | {ma.false_positive_rate:.2%} ({ma.fp}/{ma.fp + ma.tn if ma.fp + ma.tn > 0 else 1}) | {ma.false_positive_rate - mb.false_positive_rate:+.2%} |",
        f"| **Strict Accuracy (Abstentions as Errors)** | {mb.strict_accuracy:.2%} ({mb.tp + mb.tn}/{mb.total_cases}) | {ma.strict_accuracy:.2%} ({ma.tp + ma.tn}/{ma.total_cases}) | {ma.strict_accuracy - mb.strict_accuracy:+.2%} |",
        "",
        f"**Recovered Cases:** {len(report.recovered_cases)} | **Broken Decisions:** {len(report.broken_decisions)} | **Infrastructure Failures:** {report.infra_failures_count}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI Entry Point & Dry-Run Guard
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Case Live Relevance Grading Evaluation Runner (Dry-run default)."
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        default=False,
        help="Explicit authorization flag required to execute live model calls against Ollama.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicitly run in dry-run mode (default behavior).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Grader model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"Maximum inference-call budget limit (default: {DEFAULT_MAX_CALLS}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-call inference timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}s).",
    )

    args = parser.parse_args()

    # Pre-execution budget calculation
    num_cases = len(EVALUATION_CASES)
    planned_calls = num_cases * 2  # Condition A + Condition B

    print("=====================================================================")
    print("ChatLens-RAG: Multi-Case Relevance Grading Evaluation Runner")
    print("=====================================================================")
    print(f"Target Grader Model:      {args.model}")
    print(f"Evaluation Cases:         {num_cases}")
    print(f"Planned Inference Calls:  {planned_calls} (12 cases x 2 conditions)")
    print(f"Max Call Budget Limit:    {args.max_calls}")
    print("=====================================================================")

    # Enforce inference budget prior to any action
    if planned_calls > args.max_calls:
        err_msg = (
            f"INFERENCE BUDGET EXCEEDED: Planned inference calls ({planned_calls}) "
            f"exceeds authorized maximum budget ({args.max_calls}). Aborting."
        )
        print(f"\n[ERROR] {err_msg}", file=sys.stderr)
        raise InferenceBudgetExceededError(err_msg)

    # STRICT LIVE GUARD: Default to dry-run unless --confirm-live is supplied
    is_live_requested = args.confirm_live and not args.dry_run

    if not is_live_requested:
        print("\n[DRY-RUN MODE] Zero model calls will be executed.")
        print("Verifying isolated sandbox setup and dataset integrity...")

        with SafeContextGradingEnvironment() as env:
            env.assert_fail_closed_isolation()
            print(f"Temporary Sandbox Location: {env.tmp_path}")
            print(f"Isolated SQLite Store:      {env.db_path}")

            print("\nEvaluation Cases Registered:")
            for idx, c in enumerate(EVALUATION_CASES, 1):
                preceding_count = len(c.preceding_messages)
                print(f"  {idx:02d}. [{c.case_id}] {c.category} (preceding_messages={preceding_count}, expected={c.expected_label})")

        print("\n---------------------------------------------------------------------")
        print("DRY-RUN VERIFICATION SUCCESSFUL.")
        print("Zero live model calls or network endpoints were contacted.")
        print("To execute live inference in an authorized session, supply --confirm-live.")
        print("---------------------------------------------------------------------")
        return 0

    # Live Mode Execution (Requires explicit --confirm-live in authorized runs)
    print("\n[LIVE MODE AUTHORIZED] Initializing isolated evaluation environment...")
    with SafeContextGradingEnvironment() as env:
        env.assert_fail_closed_isolation()
        print(f"Temporary Sandbox Location: {env.tmp_path}")
        print(f"Isolated SQLite Store:      {env.db_path}")

        from langchain_ollama import ChatOllama
        live_llm = ChatOllama(
            model=args.model,
            temperature=0.0,
            timeout=args.timeout,
        )

        wrapped_live_llm = BudgetEnforcingLLMWrapper(
            inner_llm=live_llm,
            max_calls=args.max_calls,
        )

        print("\nExecuting live evaluation across 12 cases (Condition A and B)...")
        report = run_context_grading_suite(
            grader_llm=wrapped_live_llm,
            env=env,
            cases=EVALUATION_CASES,
            max_budget=args.max_calls,
        )

        print("\n" + format_evaluation_summary(report))
        if not report.is_complete:
            print(f"\n[ABORTED] {report.termination_reason}", file=sys.stderr)
            return 2

        print("\nLive grading evaluation completed safely within isolated sandbox.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (InferenceBudgetExceededError, LiveConfirmationRequiredError, IsolationBreachError) as exc:
        print(f"\n[ABORTED] {exc}", file=sys.stderr)
        sys.exit(2)
