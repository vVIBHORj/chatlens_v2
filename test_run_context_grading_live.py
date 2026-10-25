"""
test_run_context_grading_live.py
================================
Offline deterministic test suite for run_context_grading_live.py.

Verifies:
  1. Dry-run default makes zero model calls.
  2. Missing --confirm-live prevents inference.
  3. The inference budget blocks excess calls before execution.
  4. Both conditions receive identical candidates, candidate IDs, queries, and expected labels.
  5. The context difference is the only intended treatment variable.
  6. Production SQLite/Chroma paths are rejected with IsolationBreachError.
  7. The global production LLM is never invoked.
  8. Metrics, coverage, and abstention denominators are mathematically exact.
  9. Cleanup and restoration occur after simulated failures.

SAFETY: Strictly offline; zero Ollama, network, or production database access.
"""

import copy
import inspect
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import chat_database
import hybrid_retriever
import rag_graph
import vectorstore
from run_context_grading_live import (
    EVALUATION_CASES,
    BudgetEnforcingLLMWrapper,
    CaseExecutionResult,
    ConditionSummaryMetrics,
    InferenceBudgetExceededError,
    IsolationBreachError,
    LiveConfirmationRequiredError,
    MultiCaseEvaluationReport,
    PromptCapturingLLMWrapper,
    SafeContextGradingEnvironment,
    build_candidate_document,
    compute_condition_metrics,
    evaluate_case_condition,
    format_evaluation_summary,
    main,
    run_context_grading_suite,
)


class DeterministicMockGraderLLM:
    """Mock LLM for offline harness verification; records calls without network."""
    def __init__(self):
        self.call_count = 0
        self.captured_prompts = []

    def invoke(self, prompt, *args, **kwargs):
        self.call_count += 1
        prompt_str = str(prompt)
        self.captured_prompts.append(prompt_str)

        if "In Vault 9." in prompt_str:
            if "Preceding Conversational Context:" in prompt_str:
                decision = "relevant"
            else:
                decision = "irrelevant"
        elif "retaining all daily user database snapshots for 90 days" in prompt_str:
            decision = "relevant"
        elif "Pizza sounds great" in prompt_str:
            decision = "irrelevant"
        elif "Maybe." in prompt_str:
            decision = "uncertain"
        else:
            decision = "irrelevant"

        return AIMessage(content=json.dumps([
            {"candidate": 1, "decision": decision, "reason": "Deterministic mock evaluation"}
        ]))


# ---------------------------------------------------------------------
# Test 1 & 2: Dry-run default makes zero model calls & prevents inference
# ---------------------------------------------------------------------

def test_dry_run_default_makes_zero_model_calls(capsys):
    """Verifies that running without --confirm-live executes in dry-run mode with 0 model calls."""
    with patch.object(sys, "argv", ["run_context_grading_live.py"]):
        with patch("langchain_ollama.ChatOllama") as mock_ollama:
            exit_code = main()
            assert exit_code == 0
            # ChatOllama must NEVER be instantiated in dry-run
            mock_ollama.assert_not_called()

    captured = capsys.readouterr()
    assert "[DRY-RUN MODE] Zero model calls will be executed." in captured.out
    assert "DRY-RUN VERIFICATION SUCCESSFUL." in captured.out
    assert "Zero live model calls or network endpoints were contacted." in captured.out


def test_missing_confirm_live_flag_prevents_inference(capsys):
    """Verifies that passing --dry-run explicitly or omitting --confirm-live blocks inference."""
    with patch.object(sys, "argv", ["run_context_grading_live.py", "--dry-run"]):
        with patch("langchain_ollama.ChatOllama") as mock_ollama:
            exit_code = main()
            assert exit_code == 0
            mock_ollama.assert_not_called()

    captured = capsys.readouterr()
    assert "[DRY-RUN MODE]" in captured.out


# ---------------------------------------------------------------------
# Test 3: Inference budget blocks excess calls before execution
# ---------------------------------------------------------------------

def test_inference_budget_blocks_excess_calls_before_execution():
    """Verifies that setting a budget below planned calls aborts before any case is evaluated."""
    # 12 cases * 2 conditions = 24 planned calls. Setting max_calls=10 must raise InferenceBudgetExceededError.
    with patch.object(sys, "argv", ["run_context_grading_live.py", "--max-calls", "10"]):
        with pytest.raises(InferenceBudgetExceededError) as exc_info:
            main()
        assert "Planned inference calls (24) exceeds authorized maximum budget (10)" in str(exc_info.value)


# ---------------------------------------------------------------------
# Test 4: Both conditions receive identical candidates and labels
# ---------------------------------------------------------------------

def test_both_conditions_receive_identical_candidates_and_labels():
    """Verifies that candidate text, message IDs, queries, and ground truth match between conditions."""
    with SafeContextGradingEnvironment() as env:
        for case in EVALUATION_CASES:
            doc_a = build_candidate_document(case)
            doc_b = build_candidate_document(case)

            # Assert identical content
            assert doc_a.page_content == doc_b.page_content
            assert doc_a.metadata["message_id"] == doc_b.metadata["message_id"] == case.candidate_id
            assert doc_a.metadata["timestamp"] == doc_b.metadata["timestamp"] == case.timestamp
            assert doc_a.metadata["sender"] == doc_b.metadata["sender"] == case.sender

            # Assert expected labels are fixed and identical
            assert case.expected_label in ("relevant", "irrelevant")


# ---------------------------------------------------------------------
# Test 5: Context difference is the only intended treatment variable
# ---------------------------------------------------------------------

def test_context_difference_is_only_intended_treatment_variable():
    """
    Verifies that for short replies, Condition A prompt includes Preceding Conversational Context
    while Condition B prompt suppresses it, while candidate text is identical in both prompts.
    """
    mock_llm = DeterministicMockGraderLLM()
    # Case 1 is a terse reply: "In Vault 9."
    case_1 = EVALUATION_CASES[0]

    with SafeContextGradingEnvironment() as env:
        res_a = evaluate_case_condition(
            case=case_1,
            condition="context_aware",
            grader_llm=mock_llm,
            database_path=str(env.db_path),
        )
        res_b = evaluate_case_condition(
            case=case_1,
            condition="standalone",
            grader_llm=mock_llm,
            database_path=str(env.db_path),
        )

        assert res_a.has_preceding_context is True
        assert "Preceding Conversational Context:" in res_a.prompt
        assert "Where did we store the backup SSL certificate?" in res_a.prompt

        assert res_b.has_preceding_context is False
        assert "Preceding Conversational Context:" not in res_b.prompt

        # Both prompts MUST contain the identical candidate message text
        assert "In Vault 9." in res_a.prompt
        assert "In Vault 9." in res_b.prompt


# ---------------------------------------------------------------------
# Test 6: Production SQLite/Chroma paths are rejected
# ---------------------------------------------------------------------

def test_production_sqlite_and_chroma_paths_are_rejected():
    """Verifies fail-closed interceptors block any access to production paths."""
    with SafeContextGradingEnvironment() as env:
        # 1. Direct explicit access to production database path must raise IsolationBreachError
        with pytest.raises(IsolationBreachError):
            chat_database.get_connection("./chat_data.db")

        with pytest.raises(IsolationBreachError):
            chat_database.get_connection("chat_data.db")

        # 2. Access outside temporary sandbox directory must raise IsolationBreachError
        with pytest.raises(IsolationBreachError):
            chat_database.get_connection(r"C:\Windows\System32\fake.db")

        # 3. Explicit production chroma load must raise IsolationBreachError
        with pytest.raises(IsolationBreachError):
            hybrid_retriever.load_message_vectorstore("./chroma_db")

        with pytest.raises(IsolationBreachError):
            vectorstore.load_message_vectorstore("chroma_db")


# ---------------------------------------------------------------------
# Test 7: Global production LLM is never invoked
# ---------------------------------------------------------------------

def test_global_production_llm_is_never_invoked():
    """Verifies rag_graph.llm is never touched when running the evaluation suite."""
    poisoned_production_llm = MagicMock()
    poisoned_production_llm.invoke.side_effect = RuntimeError("BREACH: Global production LLM was invoked!")

    with patch.object(rag_graph, "llm", poisoned_production_llm):
        mock_injected_llm = DeterministicMockGraderLLM()
        with SafeContextGradingEnvironment() as env:
            # Run suite with injected LLM
            report = run_context_grading_suite(
                grader_llm=mock_injected_llm,
                env=env,
                cases=[EVALUATION_CASES[0], EVALUATION_CASES[1]],
            )
            assert report.cases_evaluated == 2
            # Global production LLM must have 0 calls
            poisoned_production_llm.invoke.assert_not_called()


# ---------------------------------------------------------------------
# Test 8: Metrics and abstention denominators are mathematically verified
# ---------------------------------------------------------------------

def test_metrics_and_abstention_denominators_are_mathematically_exact():
    """Verifies exact calculation of TP, FP, TN, FN, precision, recall, F1, FPR, coverage, and strict accuracy."""
    dummy_results = [
        # 2 TP
        CaseExecutionResult("C1", "Cat", "Q", 1, "T", "relevant", "cond", "P", True, "", "relevant", "R", "success", 0.1),
        CaseExecutionResult("C2", "Cat", "Q", 2, "T", "relevant", "cond", "P", True, "", "relevant", "R", "success", 0.1),
        # 1 FP
        CaseExecutionResult("C3", "Cat", "Q", 3, "T", "irrelevant", "cond", "P", True, "", "relevant", "R", "success", 0.1),
        # 2 TN
        CaseExecutionResult("C4", "Cat", "Q", 4, "T", "irrelevant", "cond", "P", True, "", "irrelevant", "R", "success", 0.1),
        CaseExecutionResult("C5", "Cat", "Q", 5, "T", "irrelevant", "cond", "P", True, "", "irrelevant", "R", "success", 0.1),
        # 1 FN
        CaseExecutionResult("C6", "Cat", "Q", 6, "T", "relevant", "cond", "P", True, "", "irrelevant", "R", "success", 0.1),
        # 2 Abstentions (uncertain)
        CaseExecutionResult("C7", "Cat", "Q", 7, "T", "relevant", "cond", "P", True, "", "uncertain", "R", "success", 0.1),
        CaseExecutionResult("C8", "Cat", "Q", 8, "T", "irrelevant", "cond", "P", True, "", "uncertain", "R", "success", 0.1),
    ]

    metrics = compute_condition_metrics("TestCondition", dummy_results)

    # Denominators:
    # Total: 8
    # Decisive: 6 (2 TP, 1 FP, 2 TN, 1 FN)
    # Abstentions: 2
    assert metrics.total_cases == 8
    assert metrics.decisive_cases == 6
    assert metrics.abstention_cases == 2
    assert metrics.coverage_rate == pytest.approx(6 / 8)  # 75.0%
    assert metrics.abstention_rate == pytest.approx(2 / 8)  # 25.0%

    assert metrics.tp == 2
    assert metrics.fp == 1
    assert metrics.tn == 2
    assert metrics.fn == 1

    # Precision: TP / (TP + FP) = 2 / 3
    assert metrics.precision == pytest.approx(2 / 3)
    # Recall: TP / (TP + FN) = 2 / 3
    assert metrics.recall == pytest.approx(2 / 3)
    # F1: 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
    assert metrics.f1 == pytest.approx(2 / 3)
    # Decisive Accuracy: (TP + TN) / Decisive = (2 + 2) / 6 = 4/6 = 2/3
    assert metrics.decisive_accuracy == pytest.approx(4 / 6)
    # False Positive Rate: FP / (FP + TN) = 1 / (1 + 2) = 1/3
    assert metrics.false_positive_rate == pytest.approx(1 / 3)
    # Strict Accuracy (treating abstentions as errors): (TP + TN) / Total = 4 / 8 = 0.50
    assert metrics.strict_accuracy == pytest.approx(4 / 8)


# ---------------------------------------------------------------------
# Test 9: Cleanup and restoration occur after simulated failures
# ---------------------------------------------------------------------

def test_cleanup_and_restoration_occur_after_simulated_failures():
    """Verifies that teardown removes sandbox and restores function defaults even on exception."""
    orig_db_path = chat_database.DATABASE_PATH
    orig_get_connection = chat_database.get_connection
    saved_tmp_path = None

    try:
        with SafeContextGradingEnvironment() as env:
            saved_tmp_path = env.tmp_path
            assert saved_tmp_path.exists()
            assert chat_database.DATABASE_PATH == str(env.db_path)
            # Simulate a runtime failure inside the active sandbox
            raise RuntimeError("Simulated mid-run failure!")
    except RuntimeError as e:
        assert "Simulated mid-run failure!" in str(e)

    # Assert teardown restored globals and deleted temporary directory
    assert chat_database.DATABASE_PATH == orig_db_path
    assert chat_database.get_connection == orig_get_connection
    assert saved_tmp_path is not None
    assert not saved_tmp_path.exists()


# ---------------------------------------------------------------------
# Tests 10-15: Runtime Budget Enforcement & Invocation Tracking
# ---------------------------------------------------------------------

def test_calls_up_to_limit_are_allowed():
    """Verifies that calls strictly up to max_calls succeed and increment attempted_calls."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="ok")

    wrapper = BudgetEnforcingLLMWrapper(mock_llm, max_calls=3)
    assert wrapper.attempted_calls == 0

    wrapper.invoke("prompt 1")
    wrapper.invoke("prompt 2")
    wrapper.invoke("prompt 3")

    assert wrapper.attempted_calls == 3
    assert mock_llm.invoke.call_count == 3


def test_call_exceeding_budget_rejected_before_underlying_invoke():
    """Verifies that exceeding the budget raises InferenceBudgetExceededError without calling inner model."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="ok")

    wrapper = BudgetEnforcingLLMWrapper(mock_llm, max_calls=2)
    wrapper.invoke("prompt 1")
    wrapper.invoke("prompt 2")
    assert mock_llm.invoke.call_count == 2

    # 3rd call must be rejected BEFORE invoking mock_llm
    with pytest.raises(InferenceBudgetExceededError) as exc_info:
        wrapper.invoke("prompt 3")

    assert "Inference budget exhausted" in str(exc_info.value)
    # mock_llm MUST NOT have been called a 3rd time
    assert mock_llm.invoke.call_count == 2


def test_failed_underlying_calls_still_consume_budget():
    """Verifies that exceptions raised by inner model still increment attempted_calls."""
    failing_mock = MagicMock()
    failing_mock.invoke.side_effect = RuntimeError("Ollama connection error")

    wrapper = BudgetEnforcingLLMWrapper(failing_mock, max_calls=2)

    with pytest.raises(RuntimeError):
        wrapper.invoke("prompt 1")
    assert wrapper.attempted_calls == 1

    with pytest.raises(RuntimeError):
        wrapper.invoke("prompt 2")
    assert wrapper.attempted_calls == 2

    # 3rd call hits budget limit and fails before calling failing_mock
    with pytest.raises(InferenceBudgetExceededError):
        wrapper.invoke("prompt 3")

    assert failing_mock.invoke.call_count == 2


def test_retries_consume_additional_budget():
    """Verifies that retrying an invocation consumes additional budget calls."""
    flaky_mock = MagicMock()
    flaky_mock.invoke.side_effect = [
        RuntimeError("Transient 503"),
        AIMessage(content="ok on retry"),
    ]

    wrapper = BudgetEnforcingLLMWrapper(flaky_mock, max_calls=2)

    # Attempt 1 fails
    with pytest.raises(RuntimeError):
        wrapper.invoke("prompt")
    assert wrapper.attempted_calls == 1

    # Attempt 2 (retry) succeeds but consumes 2nd budget call
    res = wrapper.invoke("prompt")
    assert res.content == "ok on retry"
    assert wrapper.attempted_calls == 2

    # Attempt 3 (another retry/call) is blocked
    with pytest.raises(InferenceBudgetExceededError):
        wrapper.invoke("prompt")


def test_counter_is_shared_across_multiple_cases_and_both_conditions():
    """Verifies that one shared wrapper enforces budget across cases and across Condition A and B."""
    mock_llm = DeterministicMockGraderLLM()
    wrapper = BudgetEnforcingLLMWrapper(mock_llm, max_calls=10)

    with SafeContextGradingEnvironment() as env:
        report = run_context_grading_suite(
            grader_llm=wrapper,
            env=env,
            cases=[EVALUATION_CASES[0], EVALUATION_CASES[1]],
        )

    # 2 cases evaluated under 2 conditions = 4 invocations total
    assert wrapper.attempted_calls == 4
    assert report.attempted_calls == 4
    assert report.is_complete is True


def test_suite_larger_than_budget_cannot_exceed_limit():
    """Verifies that a suite with more cases than budget cannot exceed the invocation limit."""
    mock_llm = DeterministicMockGraderLLM()
    # 3 cases would normally take 6 calls. Set budget to 3 calls.
    wrapper = BudgetEnforcingLLMWrapper(mock_llm, max_calls=3)

    with SafeContextGradingEnvironment() as env:
        report = run_context_grading_suite(
            grader_llm=wrapper,
            env=env,
            cases=[EVALUATION_CASES[0], EVALUATION_CASES[1], EVALUATION_CASES[2]],
            max_budget=3,
        )

    assert wrapper.attempted_calls == 3
    assert mock_llm.call_count == 3
    assert report.attempted_calls == 3
    assert report.is_complete is False


def test_budget_exhaustion_stops_suite_and_reports_incomplete():
    """Verifies that mid-suite budget exhaustion stops immediately and marks report incomplete."""
    mock_llm = DeterministicMockGraderLLM()
    wrapper = BudgetEnforcingLLMWrapper(mock_llm, max_calls=3)

    with SafeContextGradingEnvironment() as env:
        report = run_context_grading_suite(
            grader_llm=wrapper,
            env=env,
            cases=EVALUATION_CASES,
            max_budget=3,
        )

    assert report.is_complete is False
    assert "Inference budget exhausted" in report.termination_reason
    assert report.attempted_calls == 3
    assert report.max_budget == 3

    # Ensure format_evaluation_summary displays INCOMPLETE status and warning alert
    summary_text = format_evaluation_summary(report)
    assert "INCOMPLETE" in summary_text
    assert "Inference Budget Exhausted:" in summary_text
