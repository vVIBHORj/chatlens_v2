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


def test_one_required_supporting_id_cited_correctly():
    """Case 1: One required supporting ID cited correctly satisfies strict pass contract."""
    c1 = BENCHMARK_CASES[0]  # Expected: [0]
    res = CaseExecutionResult(
        case=c1,
        retrieved_message_ids=[0],
        available_context_ids=[0, 8],
        valid_citation_ids=[0],
        invalid_citation_ids=[],
        answer="The database port is 5433 [ID 0].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.attribution_precision == 1.0
    assert res.attribution_recall == 1.0
    assert res.e2e_passed is True
    assert res.partial_e2e_passed is True


def test_one_of_two_required_supporting_ids_cited_other_omitted():
    """Case 2: One of two required supporting IDs cited (other omitted) fails complete E2E pass."""
    c5 = BENCHMARK_CASES[4]  # Expected: [6, 7] (multi-message synthesis)
    res = CaseExecutionResult(
        case=c5,
        retrieved_message_ids=[6, 7],
        available_context_ids=[6, 7],
        valid_citation_ids=[7],
        invalid_citation_ids=[],
        answer="Release 2.4 sign-off was approved by QA lead [ID 7].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.attribution_precision == 1.0
    assert res.attribution_recall == 0.5
    # Strict contract: complete E2E pass requires ALL required evidence IDs (recall == 1.0)
    assert res.e2e_passed is False
    # Partial pass is acknowledged
    assert res.partial_e2e_passed is True


def test_both_required_ids_cited_correctly():
    """Case 3: Both required supporting IDs cited correctly passes complete E2E contract."""
    c5 = BENCHMARK_CASES[4]  # Expected: [6, 7]
    res = CaseExecutionResult(
        case=c5,
        retrieved_message_ids=[6, 7],
        available_context_ids=[6, 7],
        valid_citation_ids=[6, 7],
        invalid_citation_ids=[],
        answer="Release 2.4 is scheduled for Tuesday 3 AM [ID 6] and QA approved sign-off [ID 7].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.attribution_precision == 1.0
    assert res.attribution_recall == 1.0
    assert res.e2e_passed is True
    assert res.partial_e2e_passed is True


def test_valid_in_context_but_irrelevant_id_cited():
    """Case 4: A valid-in-context but irrelevant ID cited fails attribution and E2E pass."""
    c1 = BENCHMARK_CASES[0]  # Expected: [0]
    res = CaseExecutionResult(
        case=c1,
        retrieved_message_ids=[0],
        available_context_ids=[0, 1, 2, 3, 4],
        valid_citation_ids=[4],  # ID 4 is in context, but unrelated
        invalid_citation_ids=[],
        answer="Database port for the replica cluster is configured to 5433 [ID 4].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.citation_validity_rate == 1.0  # Mechanically valid
    assert res.attribution_precision == 0.0   # But 0% supporting
    assert res.attribution_recall == 0.0
    assert res.e2e_passed is False
    assert res.partial_e2e_passed is False


def test_mixture_of_relevant_and_irrelevant_ids():
    """Case 5: A mixture of relevant and irrelevant IDs cited fails strict attribution precision."""
    c1 = BENCHMARK_CASES[0]  # Expected: [0]
    res = CaseExecutionResult(
        case=c1,
        retrieved_message_ids=[0],
        available_context_ids=[0, 1, 2, 3, 4],
        valid_citation_ids=[0, 4],  # 0 is supporting, 4 is unrelated
        invalid_citation_ids=[],
        answer="Database port is 5433 [ID 0], discussed during caching update [ID 4].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.attribution_precision == 0.5  # Only 1 of 2 cited is supporting
    assert res.attribution_recall == 1.0
    # Complete E2E pass requires precision == 1.0 (no unrelated citations mixed in)
    assert res.e2e_passed is False
    assert res.partial_e2e_passed is False


def test_duplicate_citations_handling():
    """Case 6: Duplicate citations (e.g. [ID 0, ID 0]) do not distort precision or validity."""
    c1 = BENCHMARK_CASES[0]  # Expected: [0]
    res = CaseExecutionResult(
        case=c1,
        retrieved_message_ids=[0],
        available_context_ids=[0],
        valid_citation_ids=[0, 0],
        invalid_citation_ids=[],
        answer="The database port is 5433 [ID 0] as confirmed in [ID 0].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.citation_validity_rate == 1.0
    assert res.attribution_precision == 1.0
    assert res.attribution_recall == 1.0
    assert res.e2e_passed is True


def test_unanswerable_case_with_no_citations():
    """Case 7: An unanswerable case with no citations passes clean abstention contract."""
    c6 = BENCHMARK_CASES[5]  # Expected: []
    res = CaseExecutionResult(
        case=c6,
        retrieved_message_ids=[],
        available_context_ids=[],
        valid_citation_ids=[],
        invalid_citation_ids=[],
        answer="The root password for the production bastion host was not found in the chat.",
        grounded=False,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.correct_abstention is True
    assert res.grounding_category == "TN"
    assert res.attribution_precision == 1.0
    assert res.attribution_recall == 1.0
    assert res.e2e_passed is True


def test_unanswerable_case_with_citation():
    """Case 8: An unanswerable case that cites an ID fails clean abstention."""
    c6 = BENCHMARK_CASES[5]  # Expected: []
    res = CaseExecutionResult(
        case=c6,
        available_context_ids=[4],
        valid_citation_ids=[4],
        invalid_citation_ids=[],
        answer="The root password was not discussed in the chat [ID 4].",
        grounded=False,
        citations_valid=True,
    )
    evaluate_case_metrics(res)
    assert res.correct_abstention is False
    assert res.grounding_category == "FP"
    assert res.attribution_precision == 0.0
    assert res.e2e_passed is False


def test_infrastructure_exception_handling():
    """Case 9: An infrastructure exception is isolated and does not count as a quality success."""
    c_err = BENCHMARK_CASES[10]
    res = CaseExecutionResult(
        case=c_err,
        exceptions=["Internal error: Error creating hnsw segment reader: Nothing found on disk"],
    )
    evaluate_case_metrics(res)
    assert res.e2e_passed is False
    assert res.partial_e2e_passed is False
    assert res.grounding_category == "ERROR"
    assert res.anchor_retrieval_recall_at_k == 0.0
    assert res.context_availability_recall_at_k == 0.0

    summary = aggregate_benchmark_metrics([res])
    assert summary.infrastructure_error_count == 1
    assert summary.e2e_pass_rate == 0.0
    assert summary.fn_count == 0
    assert summary.fp_count == 0


def test_retrieval_metrics_distinction():
    """Verify distinction between direct-anchor recall and evidence/context availability recall."""
    c5 = BENCHMARK_CASES[4]  # Expected: [6, 7]
    res = CaseExecutionResult(
        case=c5,
        retrieved_message_ids=[6],                 # Direct anchor has Msg 6
        retrieved_chunk_ids=[1],                  # Chunk covers 5-9
        expanded_neighbor_ids=[],
        available_context_ids=[5, 6, 7, 8, 9],    # Evidence available to generation has both 6 & 7
    )
    evaluate_case_metrics(res)
    # Direct-anchor retrieval retrieved only 1 of 2 expected anchors
    assert res.anchor_retrieval_recall_at_k == 0.5
    # Context availability recall (available to generation via chunk) has both
    assert res.context_availability_recall_at_k == 1.0


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
