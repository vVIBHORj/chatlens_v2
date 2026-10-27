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
from langchain_core.documents import Document
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
    EXPERIMENTAL_GRADE_PROMPT_TEMPLATE,
    EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
    format_candidate_structure_aware,
    format_candidate_dialogue_unit,
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
        elif "pizza" in prompt_str.lower():
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


# ---------------------------------------------------------------------
# Tests 16-19: Isolated Prompt Experiment & Metric Enhancements (Task 17)
# ---------------------------------------------------------------------

def test_experimental_prompt_removes_literal_examples_and_instructs_context():
    """
    Verifies that EXPERIMENTAL_GRADE_PROMPT_TEMPLATE:
      1. Removes literal shortcut tokens 'yes', '9pm', 'sure' from short-reply rules.
      2. Instructs the grader to interpret candidate alongside Preceding Conversational Context.
      3. Clarifies context is supporting evidence, not an independent candidate to grade.
    """
    # 1. Literal examples removed
    assert "'yes'" not in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE
    assert "'9pm'" not in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE
    assert "'sure'" not in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE

    # 2. Contextual interpretation instruction present
    assert "When Preceding Conversational Context is provided" in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE
    assert "interpret the candidate message in the light of that dialogue" in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE

    # 3. Context role clarification present
    assert "supporting evidence for interpreting the candidate message, NOT an independent candidate" in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE

    # 4. JSON contract and uncertain options preserved
    assert '"candidate"' in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE
    assert '"decision"' in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE
    assert '"reason"' in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE
    assert "uncertain" in EXPERIMENTAL_GRADE_PROMPT_TEMPLATE


def test_prompt_experiment_mode_isolates_only_prompt_template():
    """
    Verifies that in prompt_experiment mode:
      1. Both Control and Treatment receive identical candidate messages, IDs, and queries.
      2. Both Control and Treatment have Preceding Conversational Context enabled.
      3. The ONLY variable differing is the prompt template wording.
    """
    mock_llm = DeterministicMockGraderLLM()
    case_1 = EVALUATION_CASES[0]  # "In Vault 9."

    with SafeContextGradingEnvironment() as env:
        res_ctrl = evaluate_case_condition(
            case=case_1,
            condition="control_prompt",
            grader_llm=mock_llm,
            database_path=str(env.db_path),
            prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
        )
        res_treat = evaluate_case_condition(
            case=case_1,
            condition="treatment_prompt",
            grader_llm=mock_llm,
            database_path=str(env.db_path),
            prompt_template=EXPERIMENTAL_GRADE_PROMPT_TEMPLATE,
        )

        # Identical candidates, queries, and ground truth
        assert res_ctrl.candidate_id == res_treat.candidate_id == case_1.candidate_id
        assert res_ctrl.query == res_treat.query == case_1.query
        assert res_ctrl.expected_label == res_treat.expected_label == case_1.expected_label

        # Both have preceding context enabled
        assert res_ctrl.has_preceding_context is True
        assert res_treat.has_preceding_context is True
        assert "Preceding Conversational Context:" in res_ctrl.prompt
        assert "Preceding Conversational Context:" in res_treat.prompt

        # Prompt template difference: Control has '9pm', Treatment does not
        assert "'9pm'" in res_ctrl.prompt
        assert "'9pm'" not in res_treat.prompt
        assert "NOT an independent candidate to grade" in res_treat.prompt
        assert "NOT an independent candidate to grade" not in res_ctrl.prompt


def test_case_07_fixture_exceeds_threshold_and_bypasses_expansion():
    """
    Verifies that corrected CASE-07 has >6 words and >35 chars, bypassing context expansion.
    """
    case_7 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-07-Self-Contained-Irrelevant")
    words = case_7.candidate_text.split()
    assert len(words) > 6, f"Expected >6 words, got {len(words)}"
    assert len(case_7.candidate_text) > 35, f"Expected >35 chars, got {len(case_7.candidate_text)}"

    doc = build_candidate_document(case_7)
    from rag_graph import should_expand_semantic_anchor, get_grading_preceding_context
    assert should_expand_semantic_anchor(doc) is False
    with patch("rag_graph.get_chronological_context") as mock_chrono:
        ctx = get_grading_preceding_context(doc)
        assert ctx is None
        mock_chrono.assert_not_called()


def test_population_recall_and_abstention_metrics():
    """
    Verifies exact mathematical calculations for decisive recall vs population recall,
    strict accuracy, decisive accuracy, and explicit confusion breakdown.
    """
    # 8 total cases: 4 ground-truth relevant, 4 ground-truth irrelevant
    # Relevant cases:
    #   Case 1: relevant -> relevant (TP)
    #   Case 2: relevant -> relevant (TP)
    #   Case 3: relevant -> irrelevant (FN)
    #   Case 4: relevant -> uncertain (Abstention on relevant)
    # Irrelevant cases:
    #   Case 5: irrelevant -> irrelevant (TN)
    #   Case 6: irrelevant -> irrelevant (TN)
    #   Case 7: irrelevant -> relevant (FP)
    #   Case 8: irrelevant -> uncertain (Abstention on irrelevant)
    dummy_results = [
        CaseExecutionResult("c1", "cat", "q", 1, "t", "relevant", "cond", "p", True, "", "relevant", "r", "success", 0.1),
        CaseExecutionResult("c2", "cat", "q", 2, "t", "relevant", "cond", "p", True, "", "relevant", "r", "success", 0.1),
        CaseExecutionResult("c3", "cat", "q", 3, "t", "relevant", "cond", "p", True, "", "irrelevant", "r", "success", 0.1),
        CaseExecutionResult("c4", "cat", "q", 4, "t", "relevant", "cond", "p", True, "", "uncertain", "r", "success", 0.1),
        CaseExecutionResult("c5", "cat", "q", 5, "t", "irrelevant", "cond", "p", False, "", "irrelevant", "r", "success", 0.1),
        CaseExecutionResult("c6", "cat", "q", 6, "t", "irrelevant", "cond", "p", False, "", "irrelevant", "r", "success", 0.1),
        CaseExecutionResult("c7", "cat", "q", 7, "t", "irrelevant", "cond", "p", False, "", "relevant", "r", "success", 0.1),
        CaseExecutionResult("c8", "cat", "q", 8, "t", "irrelevant", "cond", "p", False, "", "uncertain", "r", "success", 0.1),
    ]

    metrics = compute_condition_metrics("Test Condition", dummy_results)

    assert metrics.total_cases == 8
    assert metrics.decisive_cases == 6
    assert metrics.abstention_cases == 2

    assert metrics.tp == 2
    assert metrics.fn == 1
    assert metrics.tn == 2
    assert metrics.fp == 1

    assert metrics.total_ground_truth_relevant == 4
    assert metrics.total_ground_truth_irrelevant == 4
    assert metrics.uncertain_relevant == 1
    assert metrics.uncertain_irrelevant == 1

    # Decisive Recall: TP / (TP + FN) = 2 / (2 + 1) = 2/3
    assert metrics.recall == pytest.approx(2 / 3)

    # Population Recall: TP / Total Ground Truth Relevant = 2 / 4 = 0.50
    assert metrics.population_recall == pytest.approx(2 / 4)

    # Precision: TP / (TP + FP) = 2 / (2 + 1) = 2/3
    assert metrics.precision == pytest.approx(2 / 3)

    # Decisive Accuracy: (TP + TN) / Decisive = (2 + 2) / 6 = 4/6 = 2/3
    assert metrics.decisive_accuracy == pytest.approx(4 / 6)

    # Strict Accuracy: (TP + TN) / Total = (2 + 2) / 8 = 4/8 = 0.50
    assert metrics.strict_accuracy == pytest.approx(4 / 8)


# ---------------------------------------------------------------------
# Test 20: Structure-Aware Formatter Transcript & Target Marking
# ---------------------------------------------------------------------

def test_format_candidate_structure_aware_chronological_transcript_and_marking():
    """
    Verifies that format_candidate_structure_aware presents preceding messages and
    the candidate as a single chronological transcript with turn numbers, normalized
    timestamps, preserved message content, and explicit target marking.
    """
    from langchain_core.documents import Document

    cand_doc = Document(
        page_content="[ID 102] [2026-10-14T09:11:00] Dev_Sam: In Vault 9.",
        metadata={
            "source": "hybrid_search",
            "message_id": 102,
            "timestamp": "2026-10-14T09:11:00",
            "sender": "Dev_Sam",
        },
    )
    preceding_ctx = (
        "[ID 101] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?"
    )

    formatted = format_candidate_structure_aware(
        idx=1,
        document=cand_doc,
        preceding_context=preceding_ctx,
    )

    expected_lines = [
        "[Candidate 1] (Chronological Conversation Transcript | Target Message ID: 102)",
        "[Turn 1] [ID 101] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?",
        "[Turn 2 (TARGET MESSAGE TO EVALUATE)] [ID 102] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9.",
    ]
    assert formatted == "\n".join(expected_lines)

    # Multi-turn test (2 preceding turns + 1 candidate = 3 turns)
    preceding_multi = (
        "[ID 100] [14/10/2026 09:08 AM] Dev_Lead: Check security docs.\n"
        "[ID 101] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?"
    )
    formatted_multi = format_candidate_structure_aware(
        idx=2,
        document=cand_doc,
        preceding_context=preceding_multi,
    )
    lines_multi = formatted_multi.splitlines()
    assert len(lines_multi) == 4
    assert lines_multi[0] == "[Candidate 2] (Chronological Conversation Transcript | Target Message ID: 102)"
    assert lines_multi[1].startswith("[Turn 1] [ID 100]")
    assert lines_multi[2].startswith("[Turn 2] [ID 101]")
    assert lines_multi[3] == "[Turn 3 (TARGET MESSAGE TO EVALUATE)] [ID 102] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9."


# ---------------------------------------------------------------------
# Test 21: Structure-Aware Formatter Edge Cases
# ---------------------------------------------------------------------

def test_format_candidate_structure_aware_missing_context_and_edge_cases():
    """
    Verifies handling of missing context, malformed timestamps, and absent speaker metadata.
    """
    from langchain_core.documents import Document

    # 1. Missing / None context -> formats as clean single target message without turns
    cand_doc = Document(
        page_content="[ID 102] [2026-10-14T09:11:00] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 102},
    )
    out_none = format_candidate_structure_aware(1, cand_doc, preceding_context=None)
    assert "[Turn" not in out_none
    assert out_none == (
        "[Candidate 1] (Single Message ID: 102 | TARGET MESSAGE TO EVALUATE)\n"
        "[ID 102] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9."
    )

    out_empty = format_candidate_structure_aware(1, cand_doc, preceding_context="   ")
    assert "[Turn" not in out_empty
    assert out_empty == out_none

    # 2. Malformed timestamp -> preserved untouched, never invented
    doc_bad_ts = Document(
        page_content="[ID 200] [invalid-raw-timestamp] Alice: Test message",
        metadata={"source": "hybrid_search", "message_id": 200},
    )
    out_bad_ts = format_candidate_structure_aware(1, doc_bad_ts, None)
    assert "[invalid-raw-timestamp]" in out_bad_ts
    assert "Alice: Test message" in out_bad_ts

    # 3. Absent speaker metadata / line without speaker colon
    doc_no_spk = Document(
        page_content="[ID 201] [2026-10-14T09:15:00] System broadcast message without speaker",
        metadata={"source": "hybrid_search", "message_id": 201},
    )
    out_no_spk = format_candidate_structure_aware(1, doc_no_spk, None)
    assert "[ID 201] [14/10/2026 09:15 AM] System broadcast message without speaker" in out_no_spk


# ---------------------------------------------------------------------
# Test 22: Structure Experiment Mode Single-Variable & Shared Context
# ---------------------------------------------------------------------

def test_structure_experiment_mode_isolates_only_formatting_and_shares_context():
    """
    Verifies that structure_experiment mode:
      1. Resolves preceding context strictly once per case (never repeated for Treatment).
      2. Uses the exact same production GRADE_PROMPT_TEMPLATE for both conditions.
      3. Uses identical candidate IDs, texts, queries, and expected labels.
      4. Shares the single 24-call budget wrapper.
      5. Isolates candidate formatting as the ONLY difference.
    """
    with SafeContextGradingEnvironment() as env:
        mock_llm = DeterministicMockGraderLLM()

        # Spy on get_grading_preceding_context to verify call count
        orig_fn = rag_graph.get_grading_preceding_context
        lookup_call_count = 0

        def spy_lookup(*args, **kwargs):
            nonlocal lookup_call_count
            lookup_call_count += 1
            return orig_fn(*args, **kwargs)

        with patch("rag_graph.get_grading_preceding_context", side_effect=spy_lookup):
            report = run_context_grading_suite(
                grader_llm=mock_llm,
                env=env,
                cases=EVALUATION_CASES,
                max_budget=24,
                mode="structure_experiment",
            )

        assert report.is_complete is True
        assert report.cases_evaluated == 12

        # CRITICAL: Context lookup is performed strictly ONCE per case (12 times total, NOT 24)
        assert lookup_call_count == 12, (
            f"Expected exactly 12 context lookups (1 per candidate), got {lookup_call_count}"
        )

        assert report.metrics_a.condition_name == "Control Formatter (Standard)"
        assert report.metrics_b.condition_name == "Treatment Formatter (Structure-Aware)"

        # Compare every case between Control and Treatment
        for res_ctrl, res_treat in zip(report.condition_a_results, report.condition_b_results):
            # Identical case metadata and ground truth
            assert res_ctrl.case_id == res_treat.case_id
            assert res_ctrl.candidate_id == res_treat.candidate_id
            assert res_ctrl.candidate_text == res_treat.candidate_text
            assert res_ctrl.query == res_treat.query
            assert res_ctrl.expected_label == res_treat.expected_label
            assert res_ctrl.has_preceding_context == res_treat.has_preceding_context

            # Both conditions use the EXACT same production GRADE_PROMPT_TEMPLATE guidelines
            assert "Do NOT reject a candidate solely because of zero lexical keyword overlap" in res_ctrl.prompt
            assert "Do NOT reject a candidate solely because of zero lexical keyword overlap" in res_treat.prompt
            assert "Evaluation Guidelines:" in res_ctrl.prompt
            assert "Evaluation Guidelines:" in res_treat.prompt

            # Neither condition uses EXPERIMENTAL_GRADE_PROMPT_TEMPLATE
            assert "NOT an independent candidate to grade" not in res_ctrl.prompt
            assert "NOT an independent candidate to grade" not in res_treat.prompt

            # If preceding context is present, Control has separate block, Treatment has unified transcript
            if res_ctrl.has_preceding_context:
                assert "Preceding Conversational Context:" in res_ctrl.prompt
                assert "Preceding Conversational Context:" not in res_treat.prompt
                assert "Chronological Conversation Transcript" in res_treat.prompt
                assert "[Turn 1]" in res_treat.prompt
                assert "(TARGET MESSAGE TO EVALUATE)" in res_treat.prompt

        # Shared call budget: exactly 24 calls attempted (12 cases x 2 conditions)
        assert report.attempted_calls == 24
        assert mock_llm.call_count == 24


# ---------------------------------------------------------------------
# Test 23: CLI Dry-Run and Mode Dispatch for structure_experiment
# ---------------------------------------------------------------------

def test_structure_experiment_cli_dry_run_and_mode_dispatch(capsys):
    """
    Verifies CLI --mode structure_experiment executes safely in dry-run mode
    with 0 model calls and complete sandbox isolation.
    """
    with patch.object(sys, "argv", ["run_context_grading_live.py", "--mode", "structure_experiment"]):
        with patch("langchain_ollama.ChatOllama") as mock_ollama:
            exit_code = main()
            assert exit_code == 0
            mock_ollama.assert_not_called()

    captured = capsys.readouterr().out
    assert "Evaluation Mode:          structure_experiment" in captured
    assert "DRY-RUN VERIFICATION SUCCESSFUL" in captured


# ---------------------------------------------------------------------
# Test 24: Multiline Message Fidelity and Boundary Preservation
# ---------------------------------------------------------------------

def test_format_candidate_structure_aware_multiline_fidelity_and_edge_cases():
    """
    Verifies that format_candidate_structure_aware preserves multiline message bodies
    without fragmenting them into artificial turns:
      1. One context message with a multiline body remains exactly Turn 1.
      2. Multiple context messages where one contains embedded newlines, blank lines,
         colons, URLs, and brackets preserve all content within their respective turns.
      3. Candidate body with embedded newlines remains a single target turn.
      4. Chronological ordering is preserved and exactly one target candidate is marked.
      5. Malformed context lines without standard headers fail safely.
    """
    from langchain_core.documents import Document

    # Scenario 1: One multiline context message + single-line candidate
    multiline_ctx = (
        "[ID 101] [14/10/2026 09:10 AM] Dev_Alex: Question on deployment:\n"
        "Step 1: Check environment\n"
        "Step 2: Run migration"
    )
    cand_doc = Document(
        page_content="[ID 102] [2026-10-14T09:11:00] Dev_Sam: In Vault 9.",
        metadata={"source": "hybrid_search", "message_id": 102},
    )
    formatted_1 = format_candidate_structure_aware(1, cand_doc, preceding_context=multiline_ctx)

    # Must have exactly 2 turns (Turn 1 = preceding, Turn 2 = target candidate)
    assert "[Turn 1]" in formatted_1
    assert "[Turn 2 (TARGET MESSAGE TO EVALUATE)]" in formatted_1
    assert "[Turn 3" not in formatted_1

    # Multiline body preserved within Turn 1
    assert "Step 1: Check environment\nStep 2: Run migration" in formatted_1

    # Scenario 2: Multiple context messages with blank lines, colons, URLs, brackets
    complex_ctx = (
        "[ID 100] [14/10/2026 09:05 AM] Dev_Ops: Alert triggered: high CPU [alert-id: 99]\n"
        "[ID 101] [14/10/2026 09:10 AM] Dev_Alex: See internal dashboard:\n"
        "\n"
        "URL: https://metrics.internal.net/cpu?range=1h\n"
        "Config: [cluster=prod-us-east]\n"
        "[ID 102] [14/10/2026 09:12 AM] Dev_Lead: Under investigation."
    )
    multiline_cand = Document(
        page_content=(
            "[ID 103] [2026-10-14T09:15:00] Dev_Sam: Resolution plan:\n"
            "- Step 1: restart daemon\n"
            "- Step 2: verify logs: https://logs.internal.net/app"
        ),
        metadata={"source": "hybrid_search", "message_id": 103},
    )
    formatted_2 = format_candidate_structure_aware(2, multiline_cand, preceding_context=complex_ctx)

    # Exactly 4 turns in total (3 preceding messages + 1 candidate)
    assert "[Turn 1] [ID 100]" in formatted_2
    assert "[Turn 2] [ID 101]" in formatted_2
    assert "[Turn 3] [ID 102]" in formatted_2
    assert "[Turn 4 (TARGET MESSAGE TO EVALUATE)] [ID 103]" in formatted_2
    assert "[Turn 5" not in formatted_2

    # Verify complex content in Turn 2 (blank lines, colons, URLs, brackets)
    assert "URL: https://metrics.internal.net/cpu?range=1h" in formatted_2
    assert "Config: [cluster=prod-us-east]" in formatted_2
    assert "\n\nURL:" in formatted_2

    # Verify multiline content in candidate Turn 4
    assert "- Step 1: restart daemon\n- Step 2: verify logs: https://logs.internal.net/app" in formatted_2

    # Scenario 3: Candidate body with embedded newlines and no context
    formatted_3 = format_candidate_structure_aware(3, multiline_cand, preceding_context=None)
    assert "[Turn" not in formatted_3
    assert "[Candidate 3] (Single Message ID: 103 | TARGET MESSAGE TO EVALUATE)" in formatted_3
    assert "- Step 1: restart daemon\n- Step 2: verify logs: https://logs.internal.net/app" in formatted_3

    # Scenario 4: Malformed context lines and missing metadata
    doc_no_meta = Document(
        page_content="[ID 200] [invalid-ts] Anonymous: Message body",
        metadata={},
    )
    malformed_ctx = "Headerless line 1\nHeaderless line 2"
    formatted_4 = format_candidate_structure_aware(1, doc_no_meta, preceding_context=malformed_ctx)
    assert "[Candidate 1] (Chronological Conversation Transcript)" in formatted_4
    assert "[Turn 1] Headerless line 1\nHeaderless line 2" in formatted_4
    assert "[Turn 2 (TARGET MESSAGE TO EVALUATE)] [ID 200] [invalid-ts] Anonymous: Message body" in formatted_4
    assert "[Turn 3" not in formatted_4

    # Scenario 5: Exact preservation of leading indentation, internal blank lines, and trailing blank lines
    indent_trailing_ctx = (
        "[ID 105] [14/10/2026 09:20 AM] Dev_Alex: Code snippet:\n"
        "    def calculate_total():\n"
        "        return 42\n"
        "\n"
    )
    cand_doc_indent = Document(
        page_content=(
            "[ID 106] [2026-10-14T09:22:00] Dev_Sam:    Indented candidate response\n"
            "        with extra indent\n"
            "\n"
        ),
        metadata={"source": "hybrid_search", "message_id": 106},
    )
    formatted_5 = format_candidate_structure_aware(5, cand_doc_indent, preceding_context=indent_trailing_ctx)
    # Turn 1 must preserve leading indentation and trailing blank line
    assert (
        "[Turn 1] [ID 105] [14/10/2026 09:20 AM] Dev_Alex: Code snippet:\n"
        "    def calculate_total():\n"
        "        return 42\n"
        "\n"
    ) in formatted_5

    # Turn 2 must preserve candidate indentation and trailing blank line
    assert (
        "[Turn 2 (TARGET MESSAGE TO EVALUATE)] [ID 106] [14/10/2026 09:22 AM] Dev_Sam:    Indented candidate response\n"
        "        with extra indent\n"
        "\n"
    ) in formatted_5
    assert "[Turn 3" not in formatted_5


# ---------------------------------------------------------------------
# Task 28: Offline Prompt/Formatter Compatibility Study
# ---------------------------------------------------------------------

def test_case_09_deterministic_exact_reconstruction_and_differences():
    """
    Task 28 (Item 1 & 6): Deterministic offline reconstruction of CASE-09.
    Reconstructs both standard and structure-aware candidate strings and verifies
    exact differences in headers, turn numbering, timestamps, and target markings.
    """
    case_9 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-09-Multi-Message-Exchange")
    doc_9 = build_candidate_document(case_9)

    with SafeContextGradingEnvironment() as env:
        resolved_context = rag_graph.get_grading_preceding_context(
            doc_9, database_path=str(env.db_path)
        )

    assert resolved_context is not None
    # Verify resolved context contains turn 1 and turn 2
    assert "[ID 117]" in resolved_context
    assert "[ID 118]" in resolved_context

    # Reconstruct Control (Standard) candidate string
    ctrl_str = rag_graph.format_candidate_for_grader(1, doc_9, preceding_context=resolved_context)

    # Reconstruct Treatment (Structure-Aware) candidate string
    treat_str = format_candidate_structure_aware(1, doc_9, preceding_context=resolved_context)

    # Verify exact structure of Control
    expected_ctrl = (
        "[Candidate 1] (Single Message ID: 119)\n"
        "Preceding Conversational Context:\n"
        "[ID 117] [14/10/2026 01:40 PM] Dev_Monitor: Rack C4 power sensor is throwing warnings.\n"
        "[ID 118] [14/10/2026 01:41 PM] Dev_Ops: Can anyone confirm on site?\n\n"
        "Candidate Message to Evaluate:\n"
        "[ID 119] [2026-10-14T13:42:00] Dev_Facilities: Confirmed, heavy voltage drops."
    )
    assert ctrl_str == expected_ctrl

    # Verify exact structure of Treatment
    expected_treat = (
        "[Candidate 1] (Chronological Conversation Transcript | Target Message ID: 119)\n"
        "[Turn 1] [ID 117] [14/10/2026 01:40 PM] Dev_Monitor: Rack C4 power sensor is throwing warnings.\n"
        "[Turn 2] [ID 118] [14/10/2026 01:41 PM] Dev_Ops: Can anyone confirm on site?\n"
        "[Turn 3 (TARGET MESSAGE TO EVALUATE)] [ID 119] [14/10/2026 01:42 PM] Dev_Facilities: Confirmed, heavy voltage drops."
    )
    assert treat_str == expected_treat

    # Differential Invariant Assertions:
    # 1. Header label
    assert "(Single Message ID: 119)" in ctrl_str
    assert "(Chronological Conversation Transcript | Target Message ID: 119)" in treat_str

    # 2. Section markers vs Turn numbering
    assert "Preceding Conversational Context:" in ctrl_str
    assert "Candidate Message to Evaluate:" in ctrl_str
    assert "Preceding Conversational Context:" not in treat_str
    assert "Candidate Message to Evaluate:" not in treat_str

    assert "[Turn 1]" in treat_str
    assert "[Turn 2]" in treat_str
    assert "[Turn 3 (TARGET MESSAGE TO EVALUATE)]" in treat_str
    assert "[Turn" not in ctrl_str

    # 3. Target identification
    assert "(TARGET MESSAGE TO EVALUATE)" in treat_str
    assert "(TARGET MESSAGE TO EVALUATE)" not in ctrl_str

    # 4. Chronological order: Turn 1 (01:40 PM) < Turn 2 (01:41 PM) < Turn 3 (01:42 PM)
    t1_pos = treat_str.find("[Turn 1]")
    t2_pos = treat_str.find("[Turn 2]")
    t3_pos = treat_str.find("[Turn 3 (TARGET MESSAGE TO EVALUATE)]")
    assert 0 <= t1_pos < t2_pos < t3_pos

    # 5. Message IDs and speakers preserved in both
    for msg_id in ("117", "118", "119"):
        assert f"[ID {msg_id}]" in ctrl_str
        assert f"[ID {msg_id}]" in treat_str
    for spk in ("Dev_Monitor", "Dev_Ops", "Dev_Facilities"):
        assert spk in ctrl_str
        assert spk in treat_str

    # 6. Candidate message body is unchanged
    assert "Confirmed, heavy voltage drops." in ctrl_str
    assert "Confirmed, heavy voltage drops." in treat_str

    # 7. Entity isolation: "Rack C4" is exclusively in Turn 1, NOT in Turn 3
    assert "Rack C4" in treat_str.split("\n")[1]
    assert "Rack C4" not in treat_str.split("\n")[3]


def test_production_prompt_against_structured_representation_with_mock_grader():
    """
    Task 28 (Item 2): Test production prompt against structured representation using mocked grader.
    Verifies that the prompt is unchanged and that the target message and preceding turns are passed correctly.
    """
    case_9 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-09-Multi-Message-Exchange")
    doc_9 = build_candidate_document(case_9)

    with SafeContextGradingEnvironment() as env:
        resolved_context = rag_graph.get_grading_preceding_context(
            doc_9, database_path=str(env.db_path)
        )

    # Use a prompt-capturing mock LLM
    captured = []
    class PromptCaptorLLM:
        def invoke(self, prompt, *args, **kwargs):
            captured.append(str(prompt))
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "uncertain", "reason": "Target lacks explicit entity under Guideline 1"}
            ]))

    mock_captor = PromptCaptorLLM()

    # Grade with production prompt and structured formatter
    graded = rag_graph.grade_candidate_batch(
        question=case_9.query,
        batch_docs=[doc_9],
        grader_llm=mock_captor,
        prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
        candidate_formatter=format_candidate_structure_aware,
        preceding_contexts=[resolved_context],
    )

    assert len(graded) == 1
    assert len(captured) == 1
    prompt_text = captured[0]

    # Verify production prompt guidelines are completely unchanged
    assert "Evaluation Guidelines:\n1. Evaluate each candidate individually." in prompt_text
    assert "2. Candidates may be single messages (which may include preceding conversational context for short replies) or conversation transcript chunks." in prompt_text
    assert "3. Conversational chats are informal: short replies (e.g., 'yes', '9pm', 'sure')" in prompt_text
    assert "7. If you are uncertain or the context is ambiguous, mark 'uncertain'" in prompt_text

    # Verify query preservation
    assert f"User Question:\n{case_9.query}" in prompt_text

    # Verify target message and preceding turns are passed correctly into the prompt
    assert "[Turn 1] [ID 117] [14/10/2026 01:40 PM] Dev_Monitor: Rack C4 power sensor is throwing warnings." in prompt_text
    assert "[Turn 2] [ID 118] [14/10/2026 01:41 PM] Dev_Ops: Can anyone confirm on site?" in prompt_text
    assert "[Turn 3 (TARGET MESSAGE TO EVALUATE)] [ID 119] [14/10/2026 01:42 PM] Dev_Facilities: Confirmed, heavy voltage drops." in prompt_text

    # Notice: Production prompt lacks guidelines explaining how to evaluate [Turn N (TARGET MESSAGE TO EVALUATE)]
    # in relation to preceding turns. Under Guideline 1 ("Evaluate each candidate individually"), an isolated
    # Turn 3 lacks "Rack C4", explaining why qwen2.5:3b defaults to "uncertain" under Guideline 7.
    assert "When a candidate is formatted as a Chronological Conversation Transcript" not in prompt_text


def test_experimental_structure_prompt_variant_instructs_turn_reference_resolution():
    """
    Task 28 (Item 3): Verifies that EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE explicitly
    instructs the grader to use preceding turns to resolve references when assessing
    the marked target message.
    """
    # 1. Verifies prompt has correct formatting placeholders
    assert "{question}" in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE
    assert "{candidates_text}" in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE

    # 2. Verifies explicit guidance on chronological conversation transcripts and marked target message
    assert "When a candidate is formatted as a Chronological Conversation Transcript with a marked target message" in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE
    assert "Use the preceding turns to resolve references, antecedents, pronouns, entities, and confirmations in the marked target message." in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE

    # 3. Verifies target message relevance rule
    assert "The target message is relevant if the preceding dialogue establishes its meaning and the target message helps answer or resolve the user question." in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE

    # 4. Verifies supporting evidence rule (preventing distractors from promoting off-topic candidates)
    assert "Preceding turns are supporting context for interpreting the marked target message, NOT independent candidates to grade." in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE
    assert "An unrelated or off-topic target message does not become relevant merely because preceding turns discuss the user question." in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE

    # 5. Verifies reference resolution rule in relevance marking guideline
    assert "If a candidate is helpful or directly answers the question (either on its own or through reference resolution with preceding turns), mark 'relevant'." in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE

    # 6. Preserves JSON schema contract
    assert '- "candidate": integer candidate number' in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE
    assert '- "decision": exactly "relevant", "irrelevant", or "uncertain"' in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE
    assert '- "reason": brief explanation' in EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE


def test_experimental_structure_prompt_mock_evaluation_and_decision_parsing():
    """
    Task 28 (Item 4): Test experimental structure prompt with mocked responses to ensure
    parsing and decision handling remain unchanged.
    Note: Mocked responses verify deterministic software contracts and plumbing;
    they do not demonstrate or claim real-model grading quality.
    """
    case_9 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-09-Multi-Message-Exchange")
    doc_9 = build_candidate_document(case_9)

    with SafeContextGradingEnvironment() as env:
        resolved_context = rag_graph.get_grading_preceding_context(
            doc_9, database_path=str(env.db_path)
        )

    # 1. Test Mocked "relevant" decision with resolution reasoning
    class MockRelevantLLM:
        def invoke(self, prompt, *args, **kwargs):
            return AIMessage(content=json.dumps([
                {
                    "candidate": 1,
                    "decision": "relevant",
                    "reason": "Turn 1 identifies Rack C4, Turn 2 asks for confirmation, and Turn 3 confirms heavy voltage drops answering the query."
                }
            ]))

    results_rel = rag_graph.grade_candidate_batch(
        question=case_9.query,
        batch_docs=[doc_9],
        grader_llm=MockRelevantLLM(),
        prompt_template=EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
        candidate_formatter=format_candidate_structure_aware,
        preceding_contexts=[resolved_context],
    )
    doc_res, dec, reason, status = results_rel[0]
    assert doc_res == doc_9
    assert dec == "relevant"
    assert status == "success"
    assert "Turn 1 identifies Rack C4" in reason

    # 2. Test Mocked "irrelevant" decision
    class MockIrrelevantLLM:
        def invoke(self, prompt, *args, **kwargs):
            return AIMessage(content=json.dumps([
                {
                    "candidate": 1,
                    "decision": "irrelevant",
                    "reason": "Target message is about unrelated facility maintenance."
                }
            ]))

    results_irrel = rag_graph.grade_candidate_batch(
        question=case_9.query,
        batch_docs=[doc_9],
        grader_llm=MockIrrelevantLLM(),
        prompt_template=EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
        candidate_formatter=format_candidate_structure_aware,
        preceding_contexts=[resolved_context],
    )
    _, dec_irr, reason_irr, status_irr = results_irrel[0]
    assert dec_irr == "irrelevant"
    assert status_irr == "success"

    # 3. Test Mocked "uncertain" decision
    class MockUncertainLLM:
        def invoke(self, prompt, *args, **kwargs):
            return AIMessage(content=json.dumps([
                {
                    "candidate": 1,
                    "decision": "uncertain",
                    "reason": "Ambiguous voltage report."
                }
            ]))

    results_unc = rag_graph.grade_candidate_batch(
        question=case_9.query,
        batch_docs=[doc_9],
        grader_llm=MockUncertainLLM(),
        prompt_template=EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
        candidate_formatter=format_candidate_structure_aware,
        preceding_contexts=[resolved_context],
    )
    _, dec_unc, reason_unc, status_unc = results_unc[0]
    assert dec_unc == "uncertain"
    assert status_unc == "success"

    # 4. Test Malformed JSON fails open safely to uncertain
    class MockMalformedLLM:
        def invoke(self, prompt, *args, **kwargs):
            return AIMessage(content="[NOT_VALID_JSON]")

    results_malformed = rag_graph.grade_candidate_batch(
        question=case_9.query,
        batch_docs=[doc_9],
        grader_llm=MockMalformedLLM(),
        prompt_template=EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
        candidate_formatter=format_candidate_structure_aware,
        preceding_contexts=[resolved_context],
    )
    _, dec_fallback, reason_fallback, status_fallback = results_malformed[0]
    assert dec_fallback == "uncertain"
    assert reason_fallback == "parsing_fallback"
    assert status_fallback == "failure"


def test_case_12_negative_control_has_no_preceding_context_and_no_transcript():
    """
    Task 28 (Item 5): CASE-12 Negative Control verification.
    CASE-12 has ordinary length (>6 words / >35 chars) and bypasses preceding context lookup.
    Verifies that:
      - should_expand_semantic_anchor is False
      - get_grading_preceding_context returns None
      - Both Control and Treatment produce standalone message representations with no turns
      - Any score difference in real-model evaluation is token-level variance on standalone text,
        NOT evidence of transcript-structure reasoning.
    """
    case_12 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-012-Ordinary-Length-Bypass")
    doc_12 = build_candidate_document(case_12)

    # 1. Word and character threshold verification
    word_count = len(case_12.candidate_text.split())
    char_count = len(case_12.candidate_text)
    assert word_count == 18, f"Expected 18 words, got {word_count}"
    assert char_count == 120, f"Expected 120 characters, got {char_count}"
    assert word_count > 6
    assert char_count > 35
    assert rag_graph.should_expand_semantic_anchor(doc_12) is False

    # 2. Context lookup returns None
    with SafeContextGradingEnvironment() as env:
        resolved_context = rag_graph.get_grading_preceding_context(
            doc_12, database_path=str(env.db_path)
        )
    assert resolved_context is None

    # 3. Format with Control formatter
    ctrl_str = rag_graph.format_candidate_for_grader(1, doc_12, preceding_context=None)
    # 4. Format with Treatment formatter
    treat_str = format_candidate_structure_aware(1, doc_12, preceding_context=None)

    # Invariants for Negative Control:
    # A. Neither has a conversation transcript header
    assert "Chronological Conversation Transcript" not in ctrl_str
    assert "Chronological Conversation Transcript" not in treat_str

    # B. Neither has turn labels
    assert "[Turn" not in ctrl_str
    assert "[Turn" not in treat_str

    # C. Neither contains preceding messages (e.g. ID 124)
    assert "124" not in ctrl_str
    assert "124" not in treat_str

    # D. Exact candidate message body is identical in both
    body_text = "Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage."
    assert body_text in ctrl_str
    assert body_text in treat_str

    # E. Exact string structures
    assert ctrl_str == (
        "[Candidate 1] (Single Message ID: 125)\n"
        "[ID 125] [2026-10-14T10:15:00] Dev_Compliance: Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage."
    )
    assert treat_str == (
        "[Candidate 1] (Single Message ID: 125 | TARGET MESSAGE TO EVALUATE)\n"
        "[ID 125] [14/10/2026 10:15 AM] Dev_Compliance: Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage."
    )

    # F. Mock grader prompt capture: verify both production and experimental prompts receive standalone candidate
    captured_prompts = []
    class NegativeControlMockLLM:
        def invoke(self, prompt, *args, **kwargs):
            captured_prompts.append(str(prompt))
            return AIMessage(content=json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Standalone policy"}]))

    mock_llm = NegativeControlMockLLM()

    # With production prompt
    rag_graph.grade_candidate_batch(
        question=case_12.query,
        batch_docs=[doc_12],
        grader_llm=mock_llm,
        prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
        candidate_formatter=format_candidate_structure_aware,
        preceding_contexts=[None],
    )
    # With experimental structure prompt
    rag_graph.grade_candidate_batch(
        question=case_12.query,
        batch_docs=[doc_12],
        grader_llm=mock_llm,
        prompt_template=EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
        candidate_formatter=format_candidate_structure_aware,
        preceding_contexts=[None],
    )

    assert len(captured_prompts) == 2
    for p in captured_prompts:
        assert "[Turn 1]" not in p
        assert "[Turn 2]" not in p
        assert "[Candidate 1] (Single Message ID: 125 | TARGET MESSAGE TO EVALUATE)" in p
        assert f"User Question:\n{case_12.query}" in p
        assert body_text in p


def test_offline_study_invariants_across_cases():
    """
    Task 28 (Item 6): Explicit invariant checks across CASE-09 and CASE-12:
      1. Target identification: marked uniquely on the target candidate
      2. Chronological order: turns are strictly in ascending time order
      3. Query preservation: exact question preserved without alteration
      4. Unchanged candidate text: verbatim candidate body preserved
      5. No duplicate context resolution: lookup executed strictly once per candidate
    """
    case_9 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-09-Multi-Message-Exchange")
    case_12 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-012-Ordinary-Length-Bypass")

    doc_9 = build_candidate_document(case_9)
    doc_12 = build_candidate_document(case_12)

    with SafeContextGradingEnvironment() as env:
        # Check duplicate resolution tracking
        lookup_calls = []
        orig_fn = rag_graph.get_grading_preceding_context

        def tracked_lookup(doc, database_path=None):
            lookup_calls.append(doc.metadata.get("message_id"))
            return orig_fn(doc, database_path=database_path)

        with patch("rag_graph.get_grading_preceding_context", side_effect=tracked_lookup):
            ctx_9 = rag_graph.get_grading_preceding_context(doc_9, database_path=str(env.db_path))
            ctx_12 = rag_graph.get_grading_preceding_context(doc_12, database_path=str(env.db_path))

        # Invariant 5: No duplicate context resolution (called exactly once per candidate)
        assert lookup_calls == [119, 125]

        # Invariant 1: Target identification
        cand_9_fmt = format_candidate_structure_aware(1, doc_9, preceding_context=ctx_9)
        cand_12_fmt = format_candidate_structure_aware(1, doc_12, preceding_context=ctx_12)

        # CASE-09 target identification
        assert cand_9_fmt.count("TARGET MESSAGE TO EVALUATE") == 1
        assert "[Turn 3 (TARGET MESSAGE TO EVALUATE)]" in cand_9_fmt
        assert "[Turn 1 (TARGET MESSAGE TO EVALUATE)]" not in cand_9_fmt
        assert "[Turn 2 (TARGET MESSAGE TO EVALUATE)]" not in cand_9_fmt

        # CASE-12 target identification
        assert cand_12_fmt.count("TARGET MESSAGE TO EVALUATE") == 1
        assert "(Single Message ID: 125 | TARGET MESSAGE TO EVALUATE)" in cand_12_fmt

        # Invariant 2: Chronological order in CASE-09
        lines_9 = [line for line in cand_9_fmt.splitlines() if line.startswith("[Turn")]
        assert len(lines_9) == 3
        assert lines_9[0].startswith("[Turn 1] [ID 117] [14/10/2026 01:40 PM]")
        assert lines_9[1].startswith("[Turn 2] [ID 118] [14/10/2026 01:41 PM]")
        assert lines_9[2].startswith("[Turn 3 (TARGET MESSAGE TO EVALUATE)] [ID 119] [14/10/2026 01:42 PM]")

        # Invariant 3: Query preservation in assembled prompts
        for case, doc, ctx in [(case_9, doc_9, ctx_9), (case_12, doc_12, ctx_12)]:
            formatted_cand = format_candidate_structure_aware(1, doc, preceding_context=ctx)
            for template in (rag_graph.GRADE_PROMPT_TEMPLATE, EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE):
                prompt = template.format(question=case.query, candidates_text=formatted_cand)
                assert f"User Question:\n{case.query}\n" in prompt
                # Question is not trimmed, altered, or lowercased
                assert case.query in prompt

        # Invariant 4: Unchanged candidate text
        assert case_9.candidate_text == "Confirmed, heavy voltage drops."
        assert case_9.candidate_text in cand_9_fmt
        assert case_12.candidate_text == "Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage."
        assert case_12.candidate_text in cand_12_fmt


# ---------------------------------------------------------------------
# Task 29: Isolated Structure-Prompt Experiment Tests
# ---------------------------------------------------------------------

def test_structure_prompt_experiment_mode_isolates_only_prompt_template_and_reuses_context():
    """
    Task 29 (Items 1, 2, 3, 4, 7): Verifies that structure_prompt_experiment mode:
      1. Uses format_candidate_structure_aware for both conditions.
      2. The ONLY experimental difference is the prompt template (Control vs Treatment).
      3. Context lookup occurs strictly ONCE per case (12 total lookups).
      4. Query, candidate text, IDs, chronology, and target markers are identical.
      5. Shared budget enforces exactly 24 attempted calls (12 cases x 2 conditions).
    """
    mock_llm = DeterministicMockGraderLLM()
    lookup_count = 0
    orig_lookup = rag_graph.get_grading_preceding_context

    def spy_lookup(*args, **kwargs):
        nonlocal lookup_count
        lookup_count += 1
        return orig_lookup(*args, **kwargs)

    with SafeContextGradingEnvironment() as env:
        with patch("rag_graph.get_grading_preceding_context", side_effect=spy_lookup):
            report = run_context_grading_suite(
                grader_llm=mock_llm,
                env=env,
                cases=EVALUATION_CASES,
                max_budget=24,
                mode="structure_prompt_experiment",
            )

    assert report.is_complete is True
    assert report.cases_evaluated == 12

    # CRITICAL: Context lookup is performed strictly ONCE per candidate (12 times total, NOT 24)
    assert lookup_count == 12, f"Expected exactly 12 context lookups, got {lookup_count}"

    assert report.metrics_a.condition_name == "Control (Production Prompt + Structured Transcript)"
    assert report.metrics_b.condition_name == "Treatment (Structure-Aware Prompt + Structured Transcript)"

    # Shared call budget: exactly 24 calls attempted (12 cases x 2 conditions)
    assert report.attempted_calls == 24
    assert mock_llm.call_count == 24

    # Verify per-case isolation and exact formatting identity across all 12 cases
    for res_ctrl, res_treat in zip(report.condition_a_results, report.condition_b_results):
        assert res_ctrl.case_id == res_treat.case_id
        assert res_ctrl.candidate_id == res_treat.candidate_id
        assert res_ctrl.candidate_text == res_treat.candidate_text
        assert res_ctrl.query == res_treat.query
        assert res_ctrl.expected_label == res_treat.expected_label
        assert res_ctrl.has_preceding_context == res_treat.has_preceding_context

        # Extract the candidates_text block from both prompts
        # Candidates block starts after "Retrieved Candidates:\n" and ends before "\n\nEvaluation Guidelines:"
        ctrl_cand_part = res_ctrl.prompt.split("Retrieved Candidates:\n")[1].split("\n\nEvaluation Guidelines:")[0]
        treat_cand_part = res_treat.prompt.split("Retrieved Candidates:\n")[1].split("\n\nEvaluation Guidelines:")[0]

        # Invariant: Formatted candidate text MUST BE 100% BYTE-FOR-BYTE IDENTICAL
        assert ctrl_cand_part == treat_cand_part, (
            f"Case {res_ctrl.case_id}: Formatted candidate text differs between Control and Treatment!\n"
            f"Control:\n{ctrl_cand_part}\nTreatment:\n{treat_cand_part}"
        )

        # Invariant: The ONLY difference is the prompt template guidelines
        # Control has production guidelines
        assert "1. Evaluate each candidate individually." in res_ctrl.prompt
        assert "3. Conversational chats are informal: short replies (e.g., 'yes', '9pm', 'sure')" in res_ctrl.prompt
        assert "When a candidate is formatted as a Chronological Conversation Transcript" not in res_ctrl.prompt

        # Treatment has structure-aware experimental prompt guidelines
        assert "When a candidate is formatted as a Chronological Conversation Transcript with a marked target message" in res_treat.prompt
        assert "Use the preceding turns to resolve references, antecedents, pronouns, entities, and confirmations in the marked target message." in res_treat.prompt
        assert "Preceding turns are supporting context for interpreting the marked target message, NOT independent candidates to grade." in res_treat.prompt


def test_structure_prompt_experiment_case_12_negative_control_exact_identity():
    """
    Task 29 (Item 5): Negative Control Verification for CASE-12 under structure_prompt_experiment.
    Verifies that:
      - CASE-12 has identical candidate formatting in both conditions.
      - CASE-12 has zero transcript turns and no conversation transcript header.
      - Fails explicitly if candidate formatting differs between Control and Treatment.
    """
    case_12 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-012-Ordinary-Length-Bypass")

    with SafeContextGradingEnvironment() as env:
        report = run_context_grading_suite(
            grader_llm=DeterministicMockGraderLLM(),
            env=env,
            cases=[case_12],
            max_budget=2,
            mode="structure_prompt_experiment",
        )

    res_ctrl = report.condition_a_results[0]
    res_treat = report.condition_b_results[0]

    # Extract candidates_text
    ctrl_cand = res_ctrl.prompt.split("Retrieved Candidates:\n")[1].split("\n\nEvaluation Guidelines:")[0]
    treat_cand = res_treat.prompt.split("Retrieved Candidates:\n")[1].split("\n\nEvaluation Guidelines:")[0]

    # Verify no transcript turns in candidate representation
    assert "[Turn" not in ctrl_cand
    assert "[Turn" not in treat_cand
    assert "Chronological Conversation Transcript" not in ctrl_cand
    assert "Chronological Conversation Transcript" not in treat_cand

    # Must be 100% identical byte-for-byte
    assert ctrl_cand == treat_cand

    expected_single_msg = (
        "[Candidate 1] (Single Message ID: 125 | TARGET MESSAGE TO EVALUATE)\n"
        "[ID 125] [14/10/2026 10:15 AM] Dev_Compliance: Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage."
    )
    assert ctrl_cand == expected_single_msg


def test_structure_prompt_experiment_cli_dry_run_and_mode_dispatch(capsys):
    """
    Task 29 (Item 7): Verifies CLI --mode structure_prompt_experiment executes safely in dry-run mode
    with 0 model calls and complete sandbox isolation.
    """
    with patch.object(sys, "argv", ["run_context_grading_live.py", "--mode", "structure_prompt_experiment"]):
        with patch("langchain_ollama.ChatOllama") as mock_ollama:
            exit_code = main()
            assert exit_code == 0
            mock_ollama.assert_not_called()

    captured = capsys.readouterr().out
    assert "Evaluation Mode:          structure_prompt_experiment" in captured
    assert "Planned Inference Calls:  24 (12 cases x 2 conditions)" in captured
    assert "DRY-RUN VERIFICATION SUCCESSFUL" in captured
    assert "Zero live model calls or network endpoints were contacted." in captured


def test_structure_prompt_experiment_mock_decisions_and_malformed_handling():
    """
    Task 29 (Item 6): Verifies mocked relevant, irrelevant, uncertain, and malformed
    decision handling under control_structure_prompt and treatment_structure_prompt.
    """
    case_9 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-09-Multi-Message-Exchange")

    with SafeContextGradingEnvironment() as env:
        doc = build_candidate_document(case_9)
        ctx = rag_graph.get_grading_preceding_context(doc, database_path=str(env.db_path))

        # 1. Test Control with mocked relevant response
        class MockCtrlRelevant:
            def invoke(self, prompt, *args, **kwargs):
                return AIMessage(content=json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Control relevant"}]))

        res_ctrl = evaluate_case_condition(
            case=case_9,
            condition="control_structure_prompt",
            grader_llm=MockCtrlRelevant(),
            database_path=str(env.db_path),
            prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
            candidate_formatter=format_candidate_structure_aware,
            preceding_context=ctx,
        )
        assert res_ctrl.parsed_decision == "relevant"
        assert res_ctrl.status == "success"

        # 2. Test Treatment with mocked uncertain response
        class MockTreatUncertain:
            def invoke(self, prompt, *args, **kwargs):
                return AIMessage(content=json.dumps([{"candidate": 1, "decision": "uncertain", "reason": "Treatment uncertain"}]))

        res_treat = evaluate_case_condition(
            case=case_9,
            condition="treatment_structure_prompt",
            grader_llm=MockTreatUncertain(),
            database_path=str(env.db_path),
            prompt_template=EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
            candidate_formatter=format_candidate_structure_aware,
            preceding_context=ctx,
        )
        assert res_treat.parsed_decision == "uncertain"
        assert res_treat.status == "success"

        # 3. Test Treatment with malformed output failing open safely
        class MockMalformed:
            def invoke(self, prompt, *args, **kwargs):
                return AIMessage(content="[CORRUPT_NOT_JSON]")

        res_malformed = evaluate_case_condition(
            case=case_9,
            condition="treatment_structure_prompt",
            grader_llm=MockMalformed(),
            database_path=str(env.db_path),
            prompt_template=EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE,
            candidate_formatter=format_candidate_structure_aware,
            preceding_context=ctx,
        )
        assert res_malformed.parsed_decision == "uncertain"
        assert res_malformed.status == "failure"
        assert res_malformed.reasoning == "parsing_fallback"


def test_production_graph_and_defaults_remain_unchanged():
    """
    Task 29 (Item 8): Verifies that production graph defaults and behavior are completely unaltered:
      - rag_graph.GRADE_PROMPT_TEMPLATE is unchanged.
      - grade_candidate_batch default prompt_template is None (resolving to GRADE_PROMPT_TEMPLATE).
      - grade_candidate_batch default candidate_formatter is None (resolving to format_candidate_for_grader).
    """
    # 1. GRADE_PROMPT_TEMPLATE verification
    assert "Evaluation Guidelines:\n1. Evaluate each candidate individually." in rag_graph.GRADE_PROMPT_TEMPLATE
    assert "3. Conversational chats are informal: short replies (e.g., 'yes', '9pm', 'sure')" in rag_graph.GRADE_PROMPT_TEMPLATE
    assert "When a candidate is formatted as a Chronological Conversation Transcript" not in rag_graph.GRADE_PROMPT_TEMPLATE

    # 2. Signature and default arguments of grade_candidate_batch
    sig = inspect.signature(rag_graph.grade_candidate_batch)
    assert sig.parameters["prompt_template"].default is None
    assert sig.parameters["candidate_formatter"].default is None
    assert sig.parameters["preceding_contexts"].default is None


# ---------------------------------------------------------------------
# Task 32: Isolated Dialogue-Unit Relevance Experiment Tests
# ---------------------------------------------------------------------

def test_format_candidate_dialogue_unit_cases_1_2_4():
    """
    Task 32 (Item 4): Verifies that format_candidate_dialogue_unit formats:
      - CASE-01: the certificate-storage question and "In Vault 9." in the same ordered dialogue unit.
      - CASE-02: the maintenance-window context and "9pm." preserved faithfully.
      - CASE-04: the staging-deployment question and its short answer presented together.
    """
    with SafeContextGradingEnvironment() as env:
        # 1. CASE-01: Terse location
        c1 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-01-Terse-Location")
        doc_1 = build_candidate_document(c1)
        ctx_1 = rag_graph.get_grading_preceding_context(doc_1, database_path=str(env.db_path))

        out_1 = format_candidate_dialogue_unit(1, doc_1, preceding_context=ctx_1)
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 102)" in out_1
        assert "Dialogue Exchange:\n" in out_1
        assert "[ID 101] [14/10/2026 09:10 AM] Dev_Alex: Where did we store the backup SSL certificate?" in out_1
        assert "[Response to Evaluate] [ID 102] [14/10/2026 09:11 AM] Dev_Sam: In Vault 9." in out_1
        # Chronological order: question appears before candidate answer
        idx_q1 = out_1.index("Where did we store the backup SSL certificate?")
        idx_a1 = out_1.index("[Response to Evaluate]")
        assert idx_q1 < idx_a1

        # 2. CASE-02: Short numerical / time
        c2 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-02-Short-Time")
        doc_2 = build_candidate_document(c2)
        ctx_2 = rag_graph.get_grading_preceding_context(doc_2, database_path=str(env.db_path))

        out_2 = format_candidate_dialogue_unit(1, doc_2, preceding_context=ctx_2)
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 104)" in out_2
        assert "When are we kicking off the production deployment window tonight?" in out_2
        assert "[Response to Evaluate] [ID 104] [14/10/2026 02:31 PM] Dev_Lead: 9pm." in out_2
        idx_q2 = out_2.index("When are we kicking off the production deployment window tonight?")
        idx_a2 = out_2.index("[Response to Evaluate]")
        assert idx_q2 < idx_a2

        # 3. CASE-04: Yes/no confirmation
        c4 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-04-Yes-No-Confirmation")
        doc_4 = build_candidate_document(c4)
        ctx_4 = rag_graph.get_grading_preceding_context(doc_4, database_path=str(env.db_path))

        out_4 = format_candidate_dialogue_unit(1, doc_4, preceding_context=ctx_4)
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 108)" in out_4
        assert "Did the database migration script finish successfully on staging?" in out_4
        assert "[Response to Evaluate] [ID 108] [14/10/2026 04:20 PM] Dev_Sarah: Yes, completely clean." in out_4
        idx_q4 = out_4.index("Did the database migration script finish successfully on staging?")
        idx_a4 = out_4.index("[Response to Evaluate]")
        assert idx_q4 < idx_a4


def test_format_candidate_dialogue_unit_case_05_and_case_08_unrelated_intact():
    """
    Task 32 (Item 4): Verifies that format_candidate_dialogue_unit does not mutate
    or invent facts for unrelated candidates:
      - CASE-05: unrelated message does not acquire facts from the keys context.
      - CASE-08: unrelated deployment-failure/weather message remains intact.
    """
    with SafeContextGradingEnvironment() as env:
        # 1. CASE-05 fixture: cafeteria keys context + kitchen counter candidate
        c5 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-05-Misleading-Context")
        doc_5 = build_candidate_document(c5)
        ctx_5 = rag_graph.get_grading_preceding_context(doc_5, database_path=str(env.db_path))

        out_5 = format_candidate_dialogue_unit(1, doc_5, preceding_context=ctx_5)
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 110)" in out_5
        assert "Where did you leave the office cafeteria keys?" in out_5
        assert "[Response to Evaluate] [ID 110] [14/10/2026 12:05 PM] Dev_Sam: On the kitchen counter." in out_5

        # 2. CASE-05 variation: library meeting message with keys context
        doc_lib = Document(
            page_content="[ID 502] [2026-10-14T17:00:00] Dev_Friend: Let's meet at the library at 5pm.",
            metadata={"source": "hybrid_search", "message_id": 502, "sender": "Dev_Friend"},
        )
        out_lib = format_candidate_dialogue_unit(1, doc_lib, preceding_context=ctx_5)
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 502)" in out_lib
        assert "Where did you leave the office cafeteria keys?" in out_lib
        assert "[Response to Evaluate] [ID 502] [14/10/2026 05:00 PM] Dev_Friend: Let's meet at the library at 5pm." in out_lib
        # Assert the candidate part itself does not acquire facts about keys
        cand_part = out_lib.split("[Response to Evaluate]")[1]
        assert "keys" not in cand_part.lower()
        assert "cafeteria" not in cand_part.lower()

        # 3. CASE-08: firewall question context + weather candidate
        c8 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-08-Unrelated-Candidate-Matching-Context")
        doc_8 = build_candidate_document(c8)
        # Bypassed lookup (length > 6 words) -> None
        ctx_8_bypassed = rag_graph.get_grading_preceding_context(doc_8, database_path=str(env.db_path))
        assert ctx_8_bypassed is None

        # Standalone formatting when bypassed
        out_8_bypassed = format_candidate_dialogue_unit(1, doc_8, preceding_context=ctx_8_bypassed)
        assert "[Candidate 1] (Standalone Message ID: 116)" in out_8_bypassed
        assert "Dev_Dana: I think the weather is going to be stormy this evening." in out_8_bypassed

        # When preceding context is provided (e.g. from fixture preceding_messages)
        raw_ctx_8 = "[ID 115] [2026-10-14T17:04:00] Dev_Security: Who approved the firewall port change yesterday?"
        out_8 = format_candidate_dialogue_unit(1, doc_8, preceding_context=raw_ctx_8)
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 116)" in out_8
        assert "Who approved the firewall port change yesterday?" in out_8
        assert "[Response to Evaluate] [ID 116] [14/10/2026 05:05 PM] Dev_Dana: I think the weather is going to be stormy this evening." in out_8

        # 4. CASE-08 variation: deployment-failure message
        doc_fail = Document(
            page_content="[ID 802] [2026-10-14T17:05:00] Dev_CI: The deployment pipeline failed again.",
            metadata={"source": "hybrid_search", "message_id": 802, "sender": "Dev_CI"},
        )
        out_fail = format_candidate_dialogue_unit(1, doc_fail, preceding_context=raw_ctx_8)
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 802)" in out_fail
        assert "[Response to Evaluate] [ID 802] [14/10/2026 05:05 PM] Dev_CI: The deployment pipeline failed again." in out_fail
        cand_fail_part = out_fail.split("[Response to Evaluate]")[1]
        assert "firewall" not in cand_fail_part.lower()


def test_format_candidate_dialogue_unit_case_09_multi_turn_chronology():
    """
    Task 32 (Item 4): Verifies that format_candidate_dialogue_unit includes all necessary
    preceding turns and the target message in chronological order for multi-turn exchanges (CASE-09).
    """
    with SafeContextGradingEnvironment() as env:
        c9 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-09-Multi-Message-Exchange")
        doc_9 = build_candidate_document(c9)
        ctx_9 = rag_graph.get_grading_preceding_context(doc_9, database_path=str(env.db_path))

        out_9 = format_candidate_dialogue_unit(1, doc_9, preceding_context=ctx_9)

        # Header asserts dialogue unit with target response ID
        assert "[Candidate 1] (Dialogue Unit | Response Message ID: 119)" in out_9
        assert "Dialogue Exchange:\n" in out_9

        # Both preceding messages must be present
        assert "[ID 117] [14/10/2026 01:40 PM] Dev_Monitor: Rack C4 power sensor is throwing warnings." in out_9
        assert "[ID 118] [14/10/2026 01:41 PM] Dev_Ops: Can anyone confirm on site?" in out_9

        # Target message must be present as the evaluated response
        assert "[Response to Evaluate] [ID 119] [14/10/2026 01:42 PM] Dev_Facilities: Confirmed, heavy voltage drops." in out_9

        # Strictly chronological order: Turn 1 < Turn 2 < Response to Evaluate
        pos_turn1 = out_9.index("Rack C4 power sensor is throwing warnings.")
        pos_turn2 = out_9.index("Can anyone confirm on site?")
        pos_cand = out_9.index("[Response to Evaluate]")

        assert pos_turn1 < pos_turn2 < pos_cand, "Dialogue turns are not in strict chronological order!"


def test_format_candidate_dialogue_unit_case_12_no_context_fallback():
    """
    Task 32 (Item 4): Verifies that format_candidate_dialogue_unit safely falls back
    to a clearly labelled standalone candidate when context is unavailable (CASE-12),
    without fabricating conversation turns.
    """
    case_12 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-012-Ordinary-Length-Bypass")
    doc_12 = build_candidate_document(case_12)

    # 1. With None preceding context
    out_none = format_candidate_dialogue_unit(1, doc_12, preceding_context=None)
    assert "[Candidate 1] (Standalone Message ID: 125)" in out_none
    assert "[ID 125] [14/10/2026 10:15 AM] Dev_Compliance: Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage." in out_none
    assert "Dialogue Exchange:" not in out_none
    assert "[Response to Evaluate]" not in out_none
    assert "[Turn" not in out_none

    # 2. With empty / whitespace preceding context
    out_empty = format_candidate_dialogue_unit(1, doc_12, preceding_context="   \n  \t  ")
    assert "[Candidate 1] (Standalone Message ID: 125)" in out_empty
    assert "Dialogue Exchange:" not in out_empty
    assert "[Response to Evaluate]" not in out_empty

    # 3. Verify exact output equality between None and whitespace
    assert out_none == out_empty


def test_format_candidate_dialogue_unit_multiline_fidelity_and_edge_cases():
    """
    Task 32 (Item 4): Verifies multiline message bodies, blank lines, indentation,
    sender labels, IDs, punctuation, and malformed inputs are preserved faithfully.
    """
    # 1. Multiline preceding context with indentation and trailing spaces
    multiline_ctx = (
        "[ID 201] [2026-10-14T09:00:00] Dev_Alice: Config details:\n"
        "  port: 8080\n"
        "  host: localhost\n\n"
        "[ID 202] [2026-10-14T09:01:00] Dev_Bob: Verified config."
    )
    # Multiline candidate with code block and punctuation
    cand_doc = Document(
        page_content="[ID 203] [2026-10-14T09:02:00] Dev_Charlie: Looks good!\n```bash\ncurl http://localhost:8080\n```",
        metadata={"source": "hybrid_search", "message_id": 203, "sender": "Dev_Charlie"},
    )

    formatted = format_candidate_dialogue_unit(1, cand_doc, preceding_context=multiline_ctx)
    assert "[Candidate 1] (Dialogue Unit | Response Message ID: 203)" in formatted
    assert "Dialogue Exchange:\n" in formatted
    assert "  port: 8080\n  host: localhost" in formatted
    assert "[Response to Evaluate] [ID 203] [14/10/2026 09:02 AM] Dev_Charlie: Looks good!\n```bash\ncurl http://localhost:8080\n```" in formatted

    # 2. Conversation chunk candidate source
    chunk_doc = Document(
        page_content="[ID 301] [2026-10-14T10:00:00] Dev_Lead: Team discussion on database indexes.",
        metadata={"source": "conversation_chunk", "chunk_id": "chunk_99"},
    )
    formatted_chunk = format_candidate_dialogue_unit(2, chunk_doc, preceding_context=None)
    assert "[Candidate 2] (Standalone Conversation Chunk ID: chunk_99)" in formatted_chunk

    # 3. Missing metadata fallback
    no_meta_doc = Document(
        page_content="Simple text message without metadata.",
        metadata={},
    )
    formatted_no_meta = format_candidate_dialogue_unit(3, no_meta_doc, preceding_context=None)
    assert "[Candidate 3] (Standalone Candidate | Source: unknown)" in formatted_no_meta
    assert "Simple text message without metadata." in formatted_no_meta

    # 4. Malformed preceding context without standard header
    malformed_ctx = "Raw unformatted log line without message ID"
    formatted_malformed = format_candidate_dialogue_unit(4, cand_doc, preceding_context=malformed_ctx)
    assert "[Candidate 4] (Dialogue Unit | Response Message ID: 203)" in formatted_malformed
    assert "Raw unformatted log line without message ID" in formatted_malformed


def test_dialogue_unit_experiment_mode_isolates_only_candidate_representation_and_reuses_context():
    """
    Task 32 (Items 3, 4): Verifies that dialogue_unit_experiment mode:
      1. Uses format_candidate_for_grader (Control) vs format_candidate_dialogue_unit (Treatment).
      2. Both conditions use the EXACT SAME production GRADE_PROMPT_TEMPLATE.
      3. Context lookup occurs strictly ONCE per case (12 total lookups).
      4. Query, candidate identity, model settings, parser, and decision policy are identical.
      5. Shared call budget enforces exactly 24 attempted calls (12 cases x 2 conditions).
    """
    mock_llm = DeterministicMockGraderLLM()
    lookup_count = 0
    orig_lookup = rag_graph.get_grading_preceding_context

    def spy_lookup(*args, **kwargs):
        nonlocal lookup_count
        lookup_count += 1
        return orig_lookup(*args, **kwargs)

    with SafeContextGradingEnvironment() as env:
        with patch("rag_graph.get_grading_preceding_context", side_effect=spy_lookup):
            report = run_context_grading_suite(
                grader_llm=mock_llm,
                env=env,
                cases=EVALUATION_CASES,
                max_budget=24,
                mode="dialogue_unit_experiment",
            )

    assert report.is_complete is True
    assert report.cases_evaluated == 12

    # Context lookup is performed strictly ONCE per candidate (12 times total, NOT 24)
    assert lookup_count == 12, f"Expected exactly 12 context lookups, got {lookup_count}"

    assert report.metrics_a.condition_name == "Control (Standard Formatter + Production Prompt)"
    assert report.metrics_b.condition_name == "Treatment (Dialogue-Unit Formatter + Production Prompt)"

    # Shared call budget: exactly 24 calls attempted (12 cases x 2 conditions)
    assert report.attempted_calls == 24
    assert mock_llm.call_count == 24

    # Verify per-case isolation and exact prompt template identity across all 12 cases
    for res_ctrl, res_treat in zip(report.condition_a_results, report.condition_b_results):
        assert res_ctrl.case_id == res_treat.case_id
        assert res_ctrl.candidate_id == res_treat.candidate_id
        assert res_ctrl.candidate_text == res_treat.candidate_text
        assert res_ctrl.query == res_treat.query
        assert res_ctrl.expected_label == res_treat.expected_label
        assert res_ctrl.has_preceding_context == res_treat.has_preceding_context

        # Invariant: Prompt guidelines in Control and Treatment MUST BE 100% BYTE-FOR-BYTE IDENTICAL
        # Both use production GRADE_PROMPT_TEMPLATE
        ctrl_guidelines = res_ctrl.prompt.split("Evaluation Guidelines:")[1]
        treat_guidelines = res_treat.prompt.split("Evaluation Guidelines:")[1]
        assert ctrl_guidelines == treat_guidelines, (
            f"Case {res_ctrl.case_id}: Evaluation guidelines differ between Control and Treatment!"
        )

        assert "1. Evaluate each candidate individually." in res_ctrl.prompt
        assert "1. Evaluate each candidate individually." in res_treat.prompt
        assert "When a candidate is formatted as a Chronological Conversation Transcript" not in res_treat.prompt

        # The candidate representation differs between Control and Treatment for cases with context
        ctrl_cand = res_ctrl.prompt.split("Retrieved Candidates:\n")[1].split("\n\nEvaluation Guidelines:")[0]
        treat_cand = res_treat.prompt.split("Retrieved Candidates:\n")[1].split("\n\nEvaluation Guidelines:")[0]

        if res_ctrl.has_preceding_context:
            assert "Preceding Conversational Context:" in ctrl_cand
            assert "Candidate Message to Evaluate:" in ctrl_cand
            assert "Dialogue Exchange:" in treat_cand
            assert "[Response to Evaluate]" in treat_cand
        else:
            # For cases without context (CASE-12), both format as standalone
            assert "[Response to Evaluate]" not in treat_cand
            assert "Dialogue Exchange:" not in treat_cand


def test_dialogue_unit_experiment_cli_dry_run_and_mode_dispatch(capsys):
    """
    Task 32 (Item 4): Verifies CLI --mode dialogue_unit_experiment executes safely in dry-run mode
    with 0 model calls and complete sandbox isolation.
    """
    with patch.object(sys, "argv", ["run_context_grading_live.py", "--mode", "dialogue_unit_experiment"]):
        with patch("langchain_ollama.ChatOllama") as mock_ollama:
            exit_code = main()
            assert exit_code == 0
            mock_ollama.assert_not_called()

    captured = capsys.readouterr().out
    assert "Evaluation Mode:          dialogue_unit_experiment" in captured
    assert "Planned Inference Calls:  24 (12 cases x 2 conditions)" in captured
    assert "DRY-RUN VERIFICATION SUCCESSFUL" in captured
    assert "Zero live model calls or network endpoints were contacted." in captured


def test_dialogue_unit_experiment_mock_decisions_and_malformed_handling():
    """
    Task 32 (Item 4): Verifies mocked relevant, irrelevant, uncertain, and malformed
    decision handling under control_dialogue_unit and treatment_dialogue_unit.
    """
    case_1 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-01-Terse-Location")

    with SafeContextGradingEnvironment() as env:
        doc = build_candidate_document(case_1)
        ctx = rag_graph.get_grading_preceding_context(doc, database_path=str(env.db_path))

        # 1. Test Control with mocked relevant response
        class MockCtrlRelevant:
            def invoke(self, prompt, *args, **kwargs):
                return AIMessage(content=json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Control relevant"}]))

        res_ctrl = evaluate_case_condition(
            case=case_1,
            condition="control_dialogue_unit",
            grader_llm=MockCtrlRelevant(),
            database_path=str(env.db_path),
            prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
            candidate_formatter=rag_graph.format_candidate_for_grader,
            preceding_context=ctx,
        )
        assert res_ctrl.parsed_decision == "relevant"
        assert res_ctrl.status == "success"

        # 2. Test Treatment with mocked relevant response
        class MockTreatRelevant:
            def invoke(self, prompt, *args, **kwargs):
                return AIMessage(content=json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Treatment relevant"}]))

        res_treat = evaluate_case_condition(
            case=case_1,
            condition="treatment_dialogue_unit",
            grader_llm=MockTreatRelevant(),
            database_path=str(env.db_path),
            prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
            candidate_formatter=format_candidate_dialogue_unit,
            preceding_context=ctx,
        )
        assert res_treat.parsed_decision == "relevant"
        assert res_treat.status == "success"

        # 3. Test Treatment with malformed output failing open safely
        class MockMalformed:
            def invoke(self, prompt, *args, **kwargs):
                return AIMessage(content="[NOT_VALID_JSON]")

        res_malformed = evaluate_case_condition(
            case=case_1,
            condition="treatment_dialogue_unit",
            grader_llm=MockMalformed(),
            database_path=str(env.db_path),
            prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
            candidate_formatter=format_candidate_dialogue_unit,
            preceding_context=ctx,
        )
        assert res_malformed.parsed_decision == "uncertain"
        assert res_malformed.status == "failure"
        assert res_malformed.reasoning == "parsing_fallback"


def test_dialogue_unit_experiment_flags_case_07_for_separate_adjudication():
    """
    Task 32 (Item 4): Confirms that CASE-07 ground-truth label remains strictly 'irrelevant'
    as defined in EVALUATION_CASES, and is flagged for separate relevance-label adjudication
    rather than being silently modified.
    """
    case_7 = next(c for c in EVALUATION_CASES if c.case_id == "CASE-07-Self-Contained-Irrelevant")
    assert case_7.expected_label == "irrelevant"
    assert case_7.category == "Self-Contained Negative"
    # Confirmation: Expected label in fixture has NOT been modified
    assert case_7.expected_standalone == "irrelevant"
    assert case_7.expected_context_aware == "irrelevant"
