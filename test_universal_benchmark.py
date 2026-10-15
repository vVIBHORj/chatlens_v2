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


def test_bm03_short_reply_minimal_vs_contextual_pair_evidence():
    """BM-03: Minimal evidence [3] and contextual pair [2, 3] pass; question-only [2] or unrelated fails."""
    c3 = [c for c in BENCHMARK_CASES if c.case_id == "BM-03-Short-Reply-Context"][0]

    # 1. Minimal sufficient evidence: citing [3] alone
    r_min = CaseExecutionResult(
        case=c3,
        retrieved_message_ids=[3],
        available_context_ids=[2, 3],
        valid_citation_ids=[3],
        answer="The backup SSL certificate was placed in Vault 9 [ID 3].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_min)
    assert r_min.attribution_precision == 1.0
    assert r_min.attribution_recall == 1.0
    assert r_min.e2e_passed is True

    # 2. Contextual pair: citing [2, 3] (question + answer)
    r_pair = CaseExecutionResult(
        case=c3,
        retrieved_message_ids=[2, 3],
        available_context_ids=[2, 3],
        valid_citation_ids=[2, 3],
        answer="Where did we store the backup SSL certificate? [ID 2] Sam confirmed: In Vault 9 [ID 3].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_pair)
    assert r_pair.attribution_precision == 1.0
    assert r_pair.attribution_recall == 1.0
    assert r_pair.e2e_passed is True

    # 3. Question-only: citing [2] alone (lacks the location answer)
    r_q = CaseExecutionResult(
        case=c3,
        retrieved_message_ids=[2],
        available_context_ids=[2, 3],
        valid_citation_ids=[2],
        answer="The backup SSL certificate was stored in Vault 9 [ID 2].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_q)
    assert r_q.attribution_precision == 1.0
    assert r_q.attribution_recall == 0.5  # Only 1 of 2 in pair {2, 3}
    assert r_q.e2e_passed is False
    assert r_q.partial_e2e_passed is True

    # 4. Unrelated citation mixed in: [3, 4]
    r_unrelated = CaseExecutionResult(
        case=c3,
        retrieved_message_ids=[3],
        available_context_ids=[2, 3, 4],
        valid_citation_ids=[3, 4],
        answer="The certificate was stored in Vault 9 [ID 3], refactored by Vikram [ID 4].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_unrelated)
    assert r_unrelated.attribution_precision == 0.5  # Only ID 3 is acceptable
    assert r_unrelated.attribution_recall == 1.0
    assert r_unrelated.e2e_passed is False
    assert r_unrelated.partial_e2e_passed is False


def test_bm04_pronoun_anaphora_evidence_policy():
    """BM-04: Minimal evidence [5] and pronoun-antecedent pair [4, 5] pass; antecedent-only [4] fails."""
    c4 = [c for c in BENCHMARK_CASES if c.case_id == "BM-04-Pronoun-Anaphora"][0]

    # 1. Minimal evidence: citing [5]
    r_min = CaseExecutionResult(
        case=c4,
        retrieved_message_ids=[5],
        available_context_ids=[4, 5],
        valid_citation_ids=[5],
        answer="The cache TTL was reduced to 10 seconds [ID 5].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_min)
    assert r_min.attribution_precision == 1.0
    assert r_min.attribution_recall == 1.0
    assert r_min.e2e_passed is True

    # 2. Contextual pair: citing [4, 5]
    r_pair = CaseExecutionResult(
        case=c4,
        retrieved_message_ids=[4, 5],
        available_context_ids=[4, 5],
        valid_citation_ids=[4, 5],
        answer="Vikram refactored middleware [ID 4] and reduced cache TTL to 10 seconds [ID 5].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_pair)
    assert r_pair.attribution_precision == 1.0
    assert r_pair.attribution_recall == 1.0
    assert r_pair.e2e_passed is True

    # 3. Antecedent-only: citing [4] (lacks TTL value)
    r_ant = CaseExecutionResult(
        case=c4,
        retrieved_message_ids=[4],
        available_context_ids=[4, 5],
        valid_citation_ids=[4],
        answer="Vikram changed cache TTL to 10 seconds [ID 4].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_ant)
    assert r_ant.attribution_precision == 1.0
    assert r_ant.attribution_recall == 0.5
    assert r_ant.e2e_passed is False


def test_bm05_multi_message_synthesis_requires_both_evidence_ids():
    """BM-05: Multi-message synthesis strictly requires both [6, 7]; citing only one fails complete E2E."""
    c5 = [c for c in BENCHMARK_CASES if c.case_id == "BM-05-Multi-Message-Synthesis"][0]

    # Only Msg 6 cited (omitted sign-off)
    r6 = CaseExecutionResult(
        case=c5,
        available_context_ids=[6, 7],
        valid_citation_ids=[6],
        answer="Release 2.4 is scheduled for Tuesday at 3 AM [ID 6].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r6)
    assert r6.attribution_precision == 1.0
    assert r6.attribution_recall == 0.5
    assert r6.e2e_passed is False
    assert r6.partial_e2e_passed is True

    # Both cited [6, 7]
    r_both = CaseExecutionResult(
        case=c5,
        available_context_ids=[6, 7],
        valid_citation_ids=[6, 7],
        answer="Release 2.4 will deploy Tuesday 3 AM [ID 6], staging signed off by QA lead [ID 7].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_both)
    assert r_both.attribution_precision == 1.0
    assert r_both.attribution_recall == 1.0
    assert r_both.e2e_passed is True

    # Both cited + unrelated [6, 7, 8]
    r_mixed = CaseExecutionResult(
        case=c5,
        available_context_ids=[6, 7, 8],
        valid_citation_ids=[6, 7, 8],
        answer="Release Tuesday [ID 6], QA lead [ID 7], and port scan article [ID 8].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_mixed)
    assert round(r_mixed.attribution_precision, 2) == 0.67
    assert r_mixed.attribution_recall == 1.0
    assert r_mixed.e2e_passed is False


def test_bm08_temporal_revision_rejects_superseded_update():
    """BM-08: Only the rescheduled update [10] is acceptable; citing superseded Msg 9 fails."""
    c8 = [c for c in BENCHMARK_CASES if c.case_id == "BM-08-Temporal-Contradiction"][0]

    # 1. Correct: citing only the rescheduled update [10]
    r_corr = CaseExecutionResult(
        case=c8,
        retrieved_message_ids=[10],
        available_context_ids=[9, 10],
        valid_citation_ids=[10],
        answer="The team standup is rescheduled to 2:00 PM in Meeting Room B [ID 10].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_corr)
    assert r_corr.attribution_precision == 1.0
    assert r_corr.attribution_recall == 1.0
    assert r_corr.superseded_avoided is True
    assert r_corr.e2e_passed is True

    # 2. Citing both superseded [9] and update [10]
    r_both = CaseExecutionResult(
        case=c8,
        retrieved_message_ids=[9, 10],
        available_context_ids=[9, 10],
        valid_citation_ids=[9, 10],
        answer="Standup was at 11:00 AM in Room A [ID 9] but rescheduled to 2:00 PM in Room B [ID 10].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_both)
    assert r_both.attribution_precision == 0.5  # Only Msg 10 is acceptable supporting evidence for rescheduled
    assert r_both.attribution_recall == 1.0
    assert r_both.superseded_avoided is False
    assert r_both.e2e_passed is False

    # 3. Citing only superseded [9]
    r_sup = CaseExecutionResult(
        case=c8,
        retrieved_message_ids=[9],
        available_context_ids=[9, 10],
        valid_citation_ids=[9],
        answer="The rescheduled standup will happen at 11:00 AM in Room A [ID 9].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_sup)
    assert r_sup.attribution_precision == 0.0
    assert r_sup.attribution_recall == 0.0
    assert r_sup.superseded_avoided is False
    assert r_sup.e2e_passed is False


def test_bm11_mixed_language_short_reply_alternatives():
    """BM-11: Citing minimal [4] or contextual pair [3, 4] passes; citing request-only [3] fails."""
    c11 = [c for c in BENCHMARK_CASES if c.case_id == "BM-11-Mixed-Language-Context"][0]

    # Minimal [4]
    r_min = CaseExecutionResult(
        case=c11,
        retrieved_message_ids=[4],
        available_context_ids=[3, 4],
        valid_citation_ids=[4],
        answer="Q3 revenue metrics were added to slide 8 [ID 4].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_min)
    assert r_min.attribution_precision == 1.0
    assert r_min.attribution_recall == 1.0
    assert r_min.e2e_passed is True

    # Contextual pair [3, 4]
    r_pair = CaseExecutionResult(
        case=c11,
        retrieved_message_ids=[3, 4],
        available_context_ids=[3, 4],
        valid_citation_ids=[3, 4],
        answer="Rohan asked to include Q3 metrics [ID 3], and Priya confirmed slide 8 [ID 4].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_pair)
    assert r_pair.attribution_precision == 1.0
    assert r_pair.attribution_recall == 1.0
    assert r_pair.e2e_passed is True

    # Request-only [3]
    r_req = CaseExecutionResult(
        case=c11,
        retrieved_message_ids=[3],
        available_context_ids=[3, 4],
        valid_citation_ids=[3],
        answer="Q3 revenue metrics were added to slide 8 [ID 3].",
        grounded=True,
        citations_valid=True,
    )
    evaluate_case_metrics(r_req)
    assert r_req.attribution_precision == 1.0
    assert r_req.attribution_recall == 0.5
    assert r_req.e2e_passed is False
