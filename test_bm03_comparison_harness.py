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
import inspect
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import chat_database
import vectorstore
import hybrid_retriever
import rag_graph
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
    """Verify default access succeeds, but explicit production or external SQLite paths raise IsolationBreachError."""
    with SafeBM03Environment(build_real_chroma=False) as env:
        # 1. Safe default access resolves to the isolated database
        conn = chat_database.get_connection()
        assert conn is not None
        conn.close()

        conn_none = chat_database.get_connection(None)
        assert conn_none is not None
        conn_none.close()

        # 2. Explicit request for production chat_data.db MUST raise IsolationBreachError (no silent redirect)
        with pytest.raises(IsolationBreachError, match="Explicit access attempted to production database"):
            chat_database.get_connection(str(env.prod_db_path))

        with pytest.raises(IsolationBreachError, match="Explicit access attempted to production database"):
            chat_database.get_connection("./chat_data.db")

        # 3. Accessing an arbitrary path outside tmp_dir raises IsolationBreachError
        external_path = Path("C:/unauthorized_external_path/other.db").resolve()
        with pytest.raises(IsolationBreachError, match="outside isolated sandbox"):
            chat_database.get_connection(str(external_path))


def test_fail_closed_guard_blocks_production_and_external_chroma_mock_mode():
    """Verify that in mock mode, explicit production or external Chroma paths raise IsolationBreachError."""
    with SafeBM03Environment(build_real_chroma=False) as env:
        # 1. Safe default access inside sandbox returns mock store
        msg_store = vectorstore.load_message_vectorstore()
        assert msg_store is not None
        chunk_store = vectorstore.load_chunk_vectorstore()
        assert chunk_store is not None

        hybrid_msg = hybrid_retriever.load_message_vectorstore()
        assert hybrid_msg is not None
        hybrid_chunk = hybrid_retriever.load_chunk_vectorstore()
        assert hybrid_chunk is not None

        # 2. Explicit sandbox path returns mock store
        sandbox_store = vectorstore.load_message_vectorstore(str(env.chroma_path))
        assert sandbox_store is not None

        # 3. Explicit production access MUST raise IsolationBreachError
        with pytest.raises(IsolationBreachError, match="Explicit access attempted to production Chroma directory"):
            vectorstore.load_message_vectorstore(str(env.prod_chroma_path))

        with pytest.raises(IsolationBreachError, match="Explicit access attempted to production Chroma directory"):
            vectorstore.load_message_vectorstore("./chroma_db")

        with pytest.raises(IsolationBreachError, match="Explicit access attempted to production Chroma directory"):
            hybrid_retriever.load_message_vectorstore("./chroma_db")

        with pytest.raises(IsolationBreachError, match="Explicit access attempted to production Chroma directory"):
            vectorstore.load_chunk_vectorstore("./chroma_db")

        # 4. External Chroma path outside sandbox raises IsolationBreachError
        external_chroma = Path("C:/unauthorized_external_path/chroma").resolve()
        with pytest.raises(IsolationBreachError, match="outside isolated sandbox"):
            vectorstore.load_message_vectorstore(str(external_chroma))


def test_fail_closed_guard_blocks_production_and_external_chroma_real_store_mode():
    """Verify that in real-store mode, explicit production or external Chroma paths are blocked before opening."""
    with patch("bm03_comparison_harness.build_vectorstore"):
        with SafeBM03Environment(build_real_chroma=True) as env:
            # 1. Explicit production Chroma path raises IsolationBreachError
            with pytest.raises(IsolationBreachError, match="Explicit access attempted to production Chroma directory"):
                vectorstore.load_message_vectorstore("./chroma_db")

            with pytest.raises(IsolationBreachError, match="Explicit access attempted to production Chroma directory"):
                vectorstore.load_message_vectorstore(str(env.prod_chroma_path))

            # 2. External Chroma path outside sandbox raises IsolationBreachError
            external_chroma = Path("C:/unauthorized_external_path/chroma").resolve()
            with pytest.raises(IsolationBreachError, match="outside isolated sandbox"):
                vectorstore.load_message_vectorstore(str(external_chroma))


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
    """Verify try/finally teardown cleans up completely even when an exception occurs inside context."""
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
            raise ValueError("Simulated unexpected crash during benchmark run")
    except ValueError:
        pass

    # Assert all original module pointers are restored
    assert chat_database.DATABASE_PATH == orig_db
    assert vectorstore.PERSIST_DIR == orig_chroma
    assert chat_database.get_connection == orig_conn_identity(orig_get_conn)
    # Assert temporary folder was removed
    assert captured_tmp_path is not None
    assert not captured_tmp_path.exists()


def test_setup_failure_cleans_up_and_restores_globals():
    """Verify that a failure during setup() removes temp dir and restores all globals/defaults."""
    orig_db = chat_database.DATABASE_PATH
    orig_chroma = vectorstore.PERSIST_DIR
    orig_get_conn = chat_database.get_connection
    orig_chrono_default = inspect.signature(chat_database.get_chronological_context).parameters["database_path"].default

    env = SafeBM03Environment(build_real_chroma=False)
    captured_tmp_path = env.tmp_path
    assert captured_tmp_path.exists()

    # Simulate unexpected failure during dataset initialization in setup()
    with patch("bm03_comparison_harness.save_messages", side_effect=RuntimeError("Simulated setup write crash")):
        with pytest.raises(RuntimeError, match="Simulated setup write crash"):
            env.setup()

    # Assert temp dir was removed
    assert not captured_tmp_path.exists()

    # Assert globals restored
    assert chat_database.DATABASE_PATH == orig_db
    assert vectorstore.PERSIST_DIR == orig_chroma
    assert chat_database.get_connection == orig_get_conn
    # Assert chat_database defaults restored
    restored_chrono_default = inspect.signature(chat_database.get_chronological_context).parameters["database_path"].default
    assert restored_chrono_default == orig_chrono_default


def test_normal_exit_restores_all_function_defaults():
    """Verify that a normal successful run restores all 14 chat_database function defaults."""
    orig_defaults = {
        name: inspect.signature(getattr(chat_database, name)).parameters["database_path"].default
        for name in [
            "create_database", "get_chronological_context", "get_connection",
            "get_context", "get_database_stats", "get_message", "get_message_range",
            "get_messages_between", "get_messages_by_sender", "rebuild_fts_index",
            "reset_database", "save_messages", "search_messages", "verify_fts_index",
        ]
    }
    with SafeBM03Environment(build_real_chroma=False) as env:
        # While active, defaults point to sandbox DB
        active_chrono_default = inspect.signature(chat_database.get_chronological_context).parameters["database_path"].default
        assert active_chrono_default == str(env.db_path)

    # After exit, all 14 defaults must be restored
    for name, orig_val in orig_defaults.items():
        restored_val = inspect.signature(getattr(chat_database, name)).parameters["database_path"].default
        assert restored_val == orig_val


def test_intervening_defaults_change_is_preserved_on_teardown():
    """
    Verify that if another component changes a function's __defaults__ while the harness
    is active, teardown avoids blindly overwriting the intervening change.
    """
    orig_chrono_default = inspect.signature(chat_database.get_chronological_context).parameters["database_path"].default
    intervening_val = "./intervening_custom_data.db"

    try:
        with SafeBM03Environment(build_real_chroma=False) as env:
            # Simulate an intervening external component mutating __defaults__
            chat_database.get_chronological_context.__defaults__ = (intervening_val,)

        # Teardown should preserve the intervening change rather than reverting to orig
        current_val = inspect.signature(chat_database.get_chronological_context).parameters["database_path"].default
        assert current_val == intervening_val
    finally:
        # Restore canonical default for test cleanliness
        chat_database.get_chronological_context.__defaults__ = (orig_chrono_default,)


def test_harness_module_alias_get_connection_is_guarded():
    """Verify that get_connection imported into bm03_comparison_harness is also fail-closed guarded."""
    import bm03_comparison_harness
    with SafeBM03Environment(build_real_chroma=False) as env:
        # Calling via module alias with production path must raise IsolationBreachError
        with pytest.raises(IsolationBreachError, match="Explicit access attempted to production database"):
            bm03_comparison_harness.get_connection("./chat_data.db")


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


def test_bm03_e2e_comparison_never_calls_global_production_llm():
    """
    Offline regression test: fails if run_bm03_e2e_comparison leaks or calls
    the global production rag_graph.llm.
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

    mock_injected_llm = make_mock_short_reply_llm()

    forbidden_global_llm = MagicMock()
    forbidden_global_llm.invoke.side_effect = AssertionError(
        "FAIL-CLOSED BREACH: Global rag_graph.llm was invoked instead of injected test LLM!"
    )

    with patch("rag_graph.llm", forbidden_global_llm):
        with SafeBM03Environment(build_real_chroma=False) as env:
            e2e_res = run_bm03_e2e_comparison(
                env=env,
                retrieved_docs=[candidate_doc],
                llm_instance=mock_injected_llm,
            )

    # Assert global LLM was NEVER called
    assert forbidden_global_llm.invoke.call_count == 0
    assert e2e_res.case_id == "BM-03-Short-Reply-Context"
    assert e2e_res.context_aware.e2e_passed is True
    assert e2e_res.context_aware.answer != ""
