"""
test_dialogue_evidence_live_harness.py
======================================
Deterministic offline tests for the Task 40 evaluation harness:
  - Budget ceiling counter verification and pre-invocation blocking.
  - Fail-closed storage isolation guards (detecting and blocking production db/chroma access).
  - Teardown & global restoration on success and exception.
  - Evaluation case manifest validation (all 12 canonical cases).
"""

import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from run_dialogue_evidence_live import (
    EVAL_CASES,
    HARD_CEILING_CALLS,
    NOMINAL_BUDGET,
    GlobalBudgetEnforcingLLM,
    InferenceBudgetExceededError,
    IsolationBreachError,
    SafeEvaluationSandbox,
)
import chat_database
import hybrid_retriever
import rag_graph
import vectorstore


def test_budget_counter_tracks_and_blocks():
    """Verify GlobalBudgetEnforcingLLM intercepts calls and halts before exceeding ceiling."""
    mock_inner = MagicMock()
    mock_inner.invoke.return_value = MagicMock(content="Mock response")

    counter = GlobalBudgetEnforcingLLM(inner_llm=mock_inner, max_calls=3)

    # 1st call
    res1 = counter.invoke("You are a relevance grader for retrieved candidates.")
    assert counter.total_attempted == 1
    assert counter.operation_counts["grading"] == 1
    assert mock_inner.invoke.call_count == 1

    # 2nd call
    res2 = counter.invoke("Answer the following question based on strict grounding rules.")
    assert counter.total_attempted == 2
    assert counter.operation_counts["generate"] == 1

    # 3rd call
    res3 = counter.invoke("You are a groundedness checker verifying factual groundedness.")
    assert counter.total_attempted == 3
    assert counter.operation_counts["grounding"] == 1

    # 4th call should be BLOCKED BEFORE execution
    with pytest.raises(InferenceBudgetExceededError) as exc_info:
        counter.invoke("Please rewrite the following question for better retrieval.")

    assert "GLOBAL INFERENCE BUDGET EXHAUSTED" in str(exc_info.value)
    # The inner mock should NOT have been called for the 4th call
    assert mock_inner.invoke.call_count == 3
    assert counter.total_attempted == 3


def test_sandbox_isolation_guards_production_paths():
    """Verify SafeEvaluationSandbox blocks any attempt to connect to production paths."""
    case = EVAL_CASES[0]

    with SafeEvaluationSandbox(case) as env:
        # Verify temporary paths are active
        assert env.tmp_path.exists()
        assert Path(chat_database.DATABASE_PATH).resolve() == env.db_path.resolve()

        # Attempting to access production chat_data.db via get_connection must raise IsolationBreachError
        with pytest.raises(IsolationBreachError):
            chat_database.get_connection("./chat_data.db")

        with pytest.raises(IsolationBreachError):
            chat_database.get_connection("chat_data.db")

        # Attempting to access any arbitrary path outside the sandbox must raise IsolationBreachError
        with pytest.raises(IsolationBreachError):
            chat_database.get_connection("C:/Windows/temp/other.db")


def test_sandbox_teardown_restores_globals():
    """Verify sandbox cleans up and restores module globals even on exception."""
    orig_db = chat_database.DATABASE_PATH
    orig_chroma = vectorstore.PERSIST_DIR
    orig_conn = chat_database.get_connection

    case = EVAL_CASES[0]
    sandbox_dir = None

    try:
        with SafeEvaluationSandbox(case) as env:
            sandbox_dir = env.tmp_dir
            assert Path(sandbox_dir).exists()
            assert chat_database.DATABASE_PATH != orig_db
            raise ValueError("Forced error inside sandbox context")
    except ValueError:
        pass

    # Ensure globals restored
    assert chat_database.DATABASE_PATH == orig_db
    assert vectorstore.PERSIST_DIR == orig_chroma
    assert chat_database.get_connection == orig_conn
    # Ensure temporary sandbox folder was removed
    assert not Path(sandbox_dir).exists()


def test_eval_case_manifest_completeness():
    """Verify all 12 canonical cases are defined with required attributes."""
    assert len(EVAL_CASES) == 12

    case_ids = [c.case_id for c in EVAL_CASES]
    assert "CASE-01-Terse-Location" in case_ids
    assert "CASE-02-Short-Time" in case_ids
    assert "CASE-03-Pronoun-Anaphora" in case_ids
    assert "CASE-04-Yes-No-Confirmation" in case_ids
    assert "CASE-010-Ambiguous-Context-Abstention" in case_ids
    assert "CASE-011-Similar-Entity-Negative" in case_ids
    assert "CASE-012-Ordinary-Length-Bypass" in case_ids

    for c in EVAL_CASES:
        assert c.query and len(c.query) > 5
        assert c.target_id > 0
        assert c.target_text
        if c.expected_answerable:
            assert len(c.expected_supporting_ids) > 0
            assert len(c.expected_answer_keywords) > 0
        else:
            assert len(c.expected_supporting_ids) == 0
            assert len(c.expected_answer_keywords) == 0


def test_nominal_budget_fits_within_hard_ceiling():
    """Verify planned nominal budget (72) is strictly below hard ceiling (84)."""
    assert NOMINAL_BUDGET <= 72
    assert HARD_CEILING_CALLS == 84
    # Max cases * max possible calls per case (3 for baseline, 2 for treatment) = 60 nominal
    expected_nominal_total = 12 * 3 + 12 * 2
    assert expected_nominal_total == 60
    assert expected_nominal_total <= NOMINAL_BUDGET
