"""
test_bm03_comparison_harness.py
================================
Deterministic offline unit test suite for the BM-03 comparison harness.
Validates:
  1. Fail-closed path guards that prevent production SQLite or Chroma access.
  2. Safe cleanup on exceptions, restoring all patched globals and deleting temp dirs.
  3. Single retrieval pass feeding identical candidate inputs to both conditions.
  4. Exact prompt differentiation (Condition A contains context; Condition B does not).
  5. Decision recording and differential detection.
  6. Optional end-to-end evaluation with existing benchmark metric definitions.

Zero Ollama/network dependencies.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import chat_database
import vectorstore
import hybrid_retriever
from bm03_comparison_harness import (
    SafeBM03Environment,
    IsolationBreachError,
    run_bm03_grader_comparison,
    run_bm03_e2e_comparison,
    BM_03_CASE,
)


def make_mock_short_reply_llm(context_dependent: bool = True):
    """
    Mock LLM that inspects prompt text:
    - If prompt has 'Preceding Conversational Context:', returns 'relevant'.
    - If prompt lacks preceding context, returns 'irrelevant'.
    """
    mock = MagicMock()

    def invoke_side_effect(prompt, *args, **kwargs):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            if "Preceding Conversational Context:" in prompt_str:
                decisions = [{
                    "candidate": 1,
                    "decision": "relevant",
                    "reason": "Preceding question establishes context for Vault 9.",
                }]
            else:
                decisions = [{
                    "candidate": 1,
                    "decision": "irrelevant",
                    "reason": "Standalone short fragment lacks topic context.",
                }]
            return AIMessage(content=json.dumps(decisions))
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="Decision: yes")
        else:
            # Generation
            return AIMessage(content="The backup SSL certificate is stored in Vault 9 [ID 3].")

    mock.invoke.side_effect = invoke_side_effect
    return mock


# =====================================================================
# 1. Fail-Closed Path Guards
# =====================================================================

def test_fail_closed_guard_blocks_access_to_production_or_external_sqlite():
    """Verify that any database path outside the temporary sandbox raises IsolationBreachError."""
    with SafeBM03Environment(build_real_chroma=False) as env:
        # 1. Normal access to the isolated database works
        conn = chat_database.get_connection()
        assert conn is not None
        conn.close()

        # 2. Accessing production chat_data.db is safely redirected to sandbox
        conn_redirected = chat_database.get_connection(str(env.prod_db_path))
        assert conn_redirected is not None
        conn_redirected.close()

        # 3. Accessing an arbitrary path outside tmp_dir raises IsolationBreachError
        external_path = Path("C:/unauthorized_external_path/other.db").resolve()
        with pytest.raises(IsolationBreachError, match="outside isolated sandbox"):
            chat_database.get_connection(str(external_path))


def test_fail_closed_guard_asserts_global_pointers_remain_in_sandbox():
    """Verify assert_fail_closed_isolation raises if globals are tampered with."""
    with SafeBM03Environment(build_real_chroma=False) as env:
        # Initially valid
        env.assert_fail_closed_isolation()

        # Tampering with chat_database.DATABASE_PATH to point to production
        try:
            chat_database.DATABASE_PATH = "./chat_data.db"
            with pytest.raises(IsolationBreachError):
                env.assert_fail_closed_isolation()
        finally:
            # Restore to sandbox
            chat_database.DATABASE_PATH = str(env.db_path)


# =====================================================================
# 2. Cleanup and Teardown Guarantees
# =====================================================================

def test_teardown_restores_all_globals_and_deletes_temp_dir_on_exception():
    """Verify try/finally teardown cleans up completely even when an exception occurs."""
    orig_db = chat_database.DATABASE_PATH
    orig_chroma = vectorstore.PERSIST_DIR
    orig_get_conn = chat_database.get_connection

    captured_tmp_path = None
    try:
        with SafeBM03Environment(build_real_chroma=False) as env:
            captured_tmp_path = env.tmp_path
            assert captured_tmp_path.exists()
            assert chat_database.DATABASE_PATH == str(env.db_path)
            # Simulate unexpected failure
            raise ValueError("Simulated unexpected crash during benchmark setup")
    except ValueError:
        pass

    # Assert all original module pointers are restored
    assert chat_database.DATABASE_PATH == orig_db
    assert vectorstore.PERSIST_DIR == orig_chroma
    assert chat_database.get_connection == orig_conn_identity(orig_get_conn)
    # Assert temporary folder was removed
    assert captured_tmp_path is not None
    assert not captured_tmp_path.exists()


def orig_conn_identity(fn):
    return fn


# =====================================================================
# 3. Grader A/B Comparison Logic
# =====================================================================

def test_bm03_grader_comparison_evaluates_identical_candidates_and_differentiates_prompts():
    """
    Verify that:
      1. Both conditions receive the same candidate documents.
      2. Condition A includes preceding context in its prompt.
      3. Condition B omits preceding context from its prompt.
      4. Decision difference is captured accurately.
    """
    candidate_doc = Document(
        page_content="[ID 3] [14/10/2026, 09:11] Dev_Sam: In Vault 9.",
        metadata={
            "source": "hybrid_search",
            "message_id": 3,
            "timestamp": "14/10/2026, 09:11",
            "sender": "Dev_Sam",
            "hybrid_score": 0.89,
        },
    )

    mock_llm = make_mock_short_reply_llm()

    with SafeBM03Environment(build_real_chroma=False) as env:
        # Run comparison with candidate document
        res = run_bm03_grader_comparison(
            env=env,
            retrieved_docs=[candidate_doc],
            grader_llm=mock_llm,
        )

    assert res.case_id == "BM-03-Short-Reply-Context"
    assert res.retrieved_candidate_ids == [3]

    # Prompt check
    assert res.prompt_has_preceding_context_A is True
    assert res.prompt_has_preceding_context_B is False
    assert "Preceding Conversational Context:" in res.context_aware.prompt
    assert "Preceding Conversational Context:" not in res.standalone_baseline.prompt

    # Preceding context in tech_ops is Msg 2: "Where did we store the backup SSL certificate?"
    assert "[ID 2]" in res.context_aware.prompt
    assert "Where did we store the backup SSL certificate?" in res.context_aware.prompt

    # Decision difference
    assert res.context_aware.retained_candidate_ids == [3]
    assert res.context_aware.rejected_candidate_ids == []
    assert res.standalone_baseline.retained_candidate_ids == []
    assert res.standalone_baseline.rejected_candidate_ids == [3]
    assert res.decision_difference_detected is True


# =====================================================================
# 4. Optional End-to-End Evaluation Logic
# =====================================================================

def test_bm03_e2e_comparison_computes_existing_benchmark_metrics():
    """
    Verify the optional end-to-end evaluation calculates metrics
    consistent with benchmark_suite.py without live models.
    """
    candidate_doc = Document(
        page_content="[ID 3] [14/10/2026, 09:11] Dev_Sam: In Vault 9.",
        metadata={
            "source": "hybrid_search",
            "message_id": 3,
            "timestamp": "14/10/2026, 09:11",
            "sender": "Dev_Sam",
            "hybrid_score": 0.89,
        },
    )

    mock_llm = make_mock_short_reply_llm()

    with SafeBM03Environment(build_real_chroma=False) as env:
        e2e_res = run_bm03_e2e_comparison(
            env=env,
            retrieved_docs=[candidate_doc],
            llm_instance=mock_llm,
        )

    assert e2e_res.case_id == "BM-03-Short-Reply-Context"
    # Condition A should have retained [ID 3] and cited it
    assert e2e_res.context_aware.valid_citation_ids == [3]
    assert e2e_res.context_aware.citations_valid is True
    assert e2e_res.context_aware.grounded is True
    assert e2e_res.context_aware.attribution_precision == 1.0
    assert e2e_res.context_aware.attribution_recall == 1.0
    assert e2e_res.context_aware.e2e_passed is True

    # Per-stage latencies recorded
    assert "grade_documents" in e2e_res.context_aware.stage_latencies
    assert "expand_context" in e2e_res.context_aware.stage_latencies
    assert "generate" in e2e_res.context_aware.stage_latencies
    assert "check_grounded" in e2e_res.context_aware.stage_latencies
    assert e2e_res.context_aware.total_latency_seconds > 0.0
