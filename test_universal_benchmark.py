"""
test_universal_benchmark.py

Deterministic offline test suite for the Universal-RAG benchmark.
Tests that the benchmark suite components, data fixtures, SQLite persistence,
FTS5 lexical search, context expansion, and deterministic metric calculations
function accurately without requiring a live Ollama server or network calls.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

import chat_database
import rag_graph
from benchmark_suite import (
    TECH_OPS_CONVERSATION,
    TEAM_COLLAB_CONVERSATION,
    BENCHMARK_CASES,
    IsolatedBenchmarkEnvironment,
    CaseExecutionResult,
    evaluate_case_metrics,
    aggregate_benchmark_metrics,
)


@pytest.fixture(scope="module")
def tech_ops_env():
    env = IsolatedBenchmarkEnvironment("test_tech", TECH_OPS_CONVERSATION, build_real_chroma=False)
    env.setup()
    yield env
    env.teardown()


@pytest.fixture(scope="module")
def team_collab_env():
    env = IsolatedBenchmarkEnvironment("test_collab", TEAM_COLLAB_CONVERSATION, build_real_chroma=False)
    env.setup()
    yield env
    env.teardown()


def test_tech_ops_fixture_integrity(tech_ops_env):
    """Verify parsing, enrichment, and SQLite table structure for Tech Ops fixture."""
    conn = chat_database.get_connection(str(tech_ops_env.db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM messages")
    row = cur.fetchone()
    conn.close()
    assert row["cnt"] == 14
    assert len(tech_ops_env.df) == 14


def test_team_collab_fixture_integrity(team_collab_env):
    """Verify parsing, enrichment, and SQLite table structure for Bilingual Collab fixture."""
    conn = chat_database.get_connection(str(team_collab_env.db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM messages")
    row = cur.fetchone()
    conn.close()
    assert row["cnt"] == 6
    assert len(team_collab_env.df) == 6


def test_deterministic_metrics_calculation():
    """Verify metric calculation formulas for precision, recall, validity, and confusion matrix."""
    # Case A: Perfect answerable case
    c1 = BENCHMARK_CASES[0]  # Direct fact: expected [0]
    r1 = CaseExecutionResult(
        case=c1,
        retrieved_message_ids=[0, 8],
        available_context_ids=[0, 8],
        valid_citation_ids=[0],
        invalid_citation_ids=[],
        answer="The database port is 5433 [ID 0].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r1)
    assert r1.retrieval_recall_at_k == 1.0
    assert r1.citation_validity_rate == 1.0
    assert r1.attribution_precision == 1.0
    assert r1.attribution_recall == 1.0
    assert r1.distractor_avoided is True
    assert r1.grounding_category == "TP"
    assert r1.e2e_passed is True

    # Case B: Absent answer case with clean abstention (zero citations)
    c6 = BENCHMARK_CASES[5]  # Absent answer: expected []
    r6 = CaseExecutionResult(
        case=c6,
        retrieved_message_ids=[],
        available_context_ids=[],
        valid_citation_ids=[],
        invalid_citation_ids=[],
        answer="The available chat context is insufficient to answer the question.",
        grounded=False,
        citations_valid=True,
    )
    evaluate_case_metrics(r6)
    assert r6.correct_abstention is True
    assert r6.grounding_category == "TN"
    assert r6.e2e_passed is True

    # Case C: Hallucinated citation ID
    r_bad = CaseExecutionResult(
        case=c1,
        retrieved_message_ids=[0],
        available_context_ids=[0],
        valid_citation_ids=[],
        invalid_citation_ids=[999],
        answer="The port is 5433 [ID 999].",
        grounded=False,
        citations_valid=False,
    )
    evaluate_case_metrics(r_bad)
    assert r_bad.citation_validity_rate == 0.0
    assert r_bad.e2e_passed is False

    # Aggregate
    summary = aggregate_benchmark_metrics([r1, r6, r_bad])
    assert summary.total_cases == 3
    assert summary.passed_cases == 2
    assert summary.tp_count == 1
    assert summary.tn_count == 1
    assert summary.fp_count == 0
    assert summary.fn_count == 1
    assert summary.infrastructure_error_count == 0


def test_chunk_end_id_misattribution_distinction():
    """
    Verifies that the benchmark identifies a cited message that is mechanically valid
    (present in context) but NOT supporting evidence (BM-01 chunk-end issue).
    """
    c1 = BENCHMARK_CASES[0]  # Expected supporting ID is [0] (database port)
    # Model cites ID 4 (Vikram's caching) which is in context, but not supporting evidence
    res = CaseExecutionResult(
        case=c1,
        available_context_ids=[0, 1, 2, 3, 4],
        valid_citation_ids=[4],
        invalid_citation_ids=[],
        answer="Database port for the replica cluster is configured to 5433 [ID 4].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    # Mechanically valid because ID 4 is in context
    assert res.citation_validity_rate == 1.0
    # But attribution precision and recall are ZERO
    assert res.attribution_precision == 0.0
    assert res.attribution_recall == 0.0
    # Therefore E2E passed MUST be False despite grounded=True
    assert res.e2e_passed is False


def test_abstention_with_citations_fails_abstention_check():
    """
    Verifies that an abstention that appends citations (e.g. [ID 4]) is rejected
    as an improper abstention rather than passing as clean abstention.
    """
    c6 = BENCHMARK_CASES[5]  # Absent answer: expected []
    res = CaseExecutionResult(
        case=c6,
        available_context_ids=[0, 1, 2, 3, 4],
        valid_citation_ids=[4],
        invalid_citation_ids=[],
        answer="The root password for the production bastion host was not discussed in the chat. [ID 4]",
        grounded=False,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    # Must NOT be marked as a correct abstention because it cited an ID
    assert res.correct_abstention is False
    assert res.e2e_passed is False
    assert res.grounding_category == "FP"


def test_infrastructure_errors_isolated_from_quality_metrics():
    """
    Verifies that infrastructure exceptions (like Chroma segment reader crash)
    are separated and do not pollute the model grounding confusion matrix as FN.
    """
    c_err = BENCHMARK_CASES[10]  # BM-11
    res = CaseExecutionResult(
        case=c_err,
        exceptions=["Internal error: Error creating hnsw segment reader: Nothing found on disk"],
    )
    evaluate_case_metrics(res)
    assert res.e2e_passed is False
    assert res.grounding_category == "ERROR"
    assert res.retrieval_recall_at_k == 0.0

    summary = aggregate_benchmark_metrics([res])
    assert summary.infrastructure_error_count == 1
    assert summary.fn_count == 0
    assert summary.fp_count == 0
    assert summary.tp_count == 0
    assert summary.tn_count == 0


def test_deterministic_offline_pipeline_execution(tech_ops_env):
    """
    Executes the end-to-end ChatLens-RAG graph pipeline offline against the
    isolated Tech Ops environment using mocked LLM and deterministic FTS5 retrieval.
    """
    tech_ops_env.activate()
    case = BENCHMARK_CASES[0]  # "What port is configured for the replica database cluster?"

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "Mentions database port 5433."},
                {"candidate": 2, "decision": "irrelevant", "reason": "Mentions legacy port scanning."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="yes")
        else:
            return AIMessage(content="The replica database port is 5433 [ID 0].")

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_llm_invoke

    # Real FTS5 will find Msg 0 and Msg 8 for "replica database cluster port"
    with patch.object(rag_graph, "llm", mock_llm), \
         patch("hybrid_retriever.semantic_message_search", return_value=[]), \
         patch("hybrid_retriever.semantic_chunk_search", return_value=[]):

        resp = rag_graph.ask(case.question, df=tech_ops_env.df)

    assert 0 in resp.get("available_citation_ids", [])
    assert resp.get("valid_citation_ids") == [0]
    assert resp.get("invalid_citation_ids") == []
    assert resp.get("grounded") is True
    assert "5433" in resp.get("answer", "")
