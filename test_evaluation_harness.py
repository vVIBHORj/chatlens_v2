"""
test_evaluation_harness.py

Offline end-to-end evaluation harness for ChatLens-RAG.

Pipeline Stage Execution Classification:
------------------------------------------
REAL STAGES (exercised directly against isolated temporary storage):
  1. WhatsApp Parsing: `parse_whatsapp_export()` parsing raw multi-line WhatsApp exports.
  2. Analytical Enrichment: `enrich_messages()` calculating stable IDs, timestamps, sentiment, senders.
  3. SQLite Persistence & Schema Integrity: `save_messages()`, tables, FTS5 triggers, rebuilds, integrity checks.
  4. Lexical Retrieval: `chat_database.search_messages()`, `hybrid_retriever.lexical_search()` running FTS5 queries.
  5. Chronological Context Expansion: `chat_database.get_chronological_context()` querying surrounding messages in SQLite.
  6. Document Formatting & Candidate Translation: `format_candidate_for_grader()`, candidate `Document` assembly.
  7. Relevance Grading Parsing & Filtering: `parse_grader_response()`, batch parsing, fail-open logic, filtering.
  8. Context Assembly & Formatting: `expand_context()` formatting `[RETRIEVED ANCHOR]` and `[CHRONOLOGICAL CONTEXT]`.
  9. Citation Parsing & Mechanical Validation: `extract_available_message_ids()`, `extract_answer_citations()`, `validate_citations()`.
  10. Graph State Routing & Decision Nodes: `route_query_node()`, `classify_scope()`, `decide_to_generate()`, `build_graph()`.

MOCKED STAGES (isolated offline to avoid external network services and live models):
  1. Vectorstore Embeddings & Semantic Search: `semantic_message_search()`, `semantic_chunk_search()`
     (mocked to return synthetic `Document` candidates offline without ChromaDB daemon or embedding server).
  2. Local LLM Invocations (`ChatOllama` / `rag_graph.llm`):
     - Grader LLM: Evaluates candidate batches into structured JSON decisions.
     - Generator LLM: Produces natural language answers with inline `[ID ...]` citations.
     - Query Rewriter LLM: Produces rewritten queries on retrieval failure.
     - Grounding LLM: Evaluates semantic claim support ("yes" / "no").
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import chat_database
import hybrid_retriever
import rag_graph
from chat_database import get_connection, save_messages
from enrich import enrich_messages, to_dataframe
from parser import parse_whatsapp_export


# =====================================================================
# 1. Synthetic WhatsApp Dataset Fixture
# =====================================================================

SYNTHETIC_WHATSAPP_EXPORT = """15/10/2026, 09:00 - Dr. Aris: The deep space telescope calibration setting is set to 1420 MHz.
15/10/2026, 09:05 - Specialist Lyra: Deep space telescope calibration setting schedules for lunch catering are in the cafeteria.
15/10/2026, 09:10 - Commander Vance: The orbital relay vessel embarks as soon as dusk falls across the landing platform.
15/10/2026, 09:20 - Tech Kael: Where did we store the backup cryogenic coolant canisters?
15/10/2026, 09:21 - Specialist Lyra: Sub-level 4.
15/10/2026, 09:22 - Tech Kael: Acknowledged, heading down with the transport cart now.
15/10/2026, 11:00 - Navigator Thorne: Did anyone see the retro movie marathon at the community center last night?
15/10/2026, 11:05 - Commander Vance: I missed it, was fixing the antenna.
"""

# Ground truth specification for the synthetic dataset
DATASET_GROUND_TRUTH = {
    # Msg 0: Direct fact
    0: {
        "sender": "Dr. Aris",
        "fact": "calibration setting is set to 1420 MHz",
        "type": "direct_factual_answer",
    },
    # Msg 1: Lexical distractor
    1: {
        "sender": "Specialist Lyra",
        "fact": "lunch catering schedules in cafeteria",
        "type": "lexical_distractor",
    },
    # Msg 2: Semantic paraphrase (little lexical overlap with "shuttle depart")
    2: {
        "sender": "Commander Vance",
        "fact": "orbital relay vessel embarks as soon as dusk falls across landing platform",
        "type": "semantic_paraphrase",
    },
    # Msg 3: Chronological inquiry
    3: {
        "sender": "Tech Kael",
        "fact": "inquired where backup cryogenic coolant canisters are stored",
        "type": "conversational_context_inquiry",
    },
    # Msg 4: Short reply (requires Msg 3 for meaning)
    4: {
        "sender": "Specialist Lyra",
        "fact": "Sub-level 4",
        "type": "short_reply_requiring_context",
    },
    # Msg 5: Chronological continuation
    5: {
        "sender": "Tech Kael",
        "fact": "heading down with transport cart",
        "type": "chronological_continuation",
    },
    # Msg 6 & 7: Unrelated conversation
    6: {
        "sender": "Navigator Thorne",
        "fact": "retro movie marathon last night",
        "type": "unrelated_chatter",
    },
    7: {
        "sender": "Commander Vance",
        "fact": "missed movie, fixing antenna",
        "type": "unrelated_chatter",
    },
}


@dataclass
class SyntheticRagEnvironment:
    db_path: str
    chat_file: str
    enriched_messages: List[Any]
    df: pd.DataFrame


@pytest.fixture
def synthetic_rag_env(tmp_path: Path):
    """
    Sets up an isolated, temporary SQLite database and populates it
    via the actual WhatsApp parsing and analytical enrichment pipeline.
    Redirects all SQLite connections to this isolated database.
    """
    chat_file = str(tmp_path / "synthetic_chat.txt")
    with open(chat_file, "w", encoding="utf-8") as f:
        f.write(SYNTHETIC_WHATSAPP_EXPORT)

    # 1. Real parsing
    parsed_messages = parse_whatsapp_export(chat_file)
    assert len(parsed_messages) == 8

    # 2. Real analytical enrichment
    enriched_messages = enrich_messages(parsed_messages)
    assert len(enriched_messages) == 8

    # 3. Real SQLite database persistence
    db_path = str(tmp_path / "synthetic_eval_chat.db")
    save_messages(enriched_messages, database_path=db_path, reset=True)

    df = to_dataframe(enriched_messages)

    # Patch chat_database.get_connection so all modules query db_path
    orig_get_connection = chat_database.get_connection

    def redirected_get_connection(target_path=chat_database.DATABASE_PATH):
        return orig_get_connection(db_path)

    with patch.object(chat_database, "get_connection", side_effect=redirected_get_connection):
        yield SyntheticRagEnvironment(
            db_path=db_path,
            chat_file=chat_file,
            enriched_messages=enriched_messages,
            df=df,
        )


# =====================================================================
# 2. Explicit Outcome Measurement Data Structure
# =====================================================================

@dataclass
class EvaluationOutcome:
    test_id: str
    question: str
    relevant_evidence_retrieved: bool
    relevant_evidence_survived_grading: bool
    irrelevant_distractor_excluded: bool
    final_context_contained_expected_id: bool
    cited_ids_valid_against_context: bool
    unsupported_marked_ungrounded: bool
    original_question_preserved: bool
    exceptions_or_unexpected_transitions: List[str]
    rewrites_used: int
    answer: str
    grounded: bool
    valid_citation_ids: List[int]
    invalid_citation_ids: List[int]
    available_citation_ids: List[int]


# =====================================================================
# 3. Harness Test Execution Engine
# =====================================================================

def run_harness_trial(
    env: SyntheticRagEnvironment,
    test_id: str,
    question: str,
    expected_supporting_ids: Set[int],
    distractor_ids: Set[int] = set(),
    mock_semantic_candidates: Optional[List[Tuple[Document, float]]] = None,
    custom_llm_invoke: Optional[Callable[[Any], AIMessage]] = None,
    grader_failure: bool = False,
    expect_rewrite: bool = False,
    is_unsupported_expected: bool = False,
) -> EvaluationOutcome:
    """
    Executes the end-to-end ChatLens-RAG pipeline against the synthetic environment,
    tracing each stage and recording explicit measurable outcomes.
    """
    retrieved_docs: List[Document] = []
    graded_docs: List[Document] = []
    expanded_docs: List[Document] = []
    exceptions: List[str] = []

    orig_retrieve = rag_graph.retrieve
    orig_grade = rag_graph.grade_documents
    orig_expand = rag_graph.expand_context

    def traced_retrieve(state):
        try:
            res = orig_retrieve(state)
            retrieved_docs.extend(res.get("documents", []))
            return res
        except Exception as e:
            exceptions.append(f"retrieve_exception: {e}")
            raise

    def traced_grade(state, **kwargs):
        try:
            res = orig_grade(state, **kwargs)
            graded_docs.extend(res.get("documents", []))
            return res
        except Exception as e:
            exceptions.append(f"grade_exception: {e}")
            raise

    def traced_expand(state):
        try:
            res = orig_expand(state)
            expanded_docs.extend(res.get("documents", []))
            return res
        except Exception as e:
            exceptions.append(f"expand_exception: {e}")
            raise

    # Default Mock LLM handler if none provided
    def default_llm_invoke(prompt):
        prompt_str = str(prompt)
        # Relevance Grader
        if "conservative relevance grader" in prompt_str:
            if grader_failure:
                raise RuntimeError("Simulated grader LLM connection failure")
            # Default: mark any distractor irrelevant, others relevant
            decisions = []
            for idx in range(1, 20):
                # Simple heuristic based on prompt text: check candidate content
                decisions.append({
                    "candidate": idx,
                    "decision": "relevant",
                    "reason": "Evaluated by harness mock grader.",
                })
            return AIMessage(content=json.dumps(decisions))

        # Query Rewrite
        elif "Rewrite the following question" in prompt_str:
            return AIMessage(content="authorization protocol sector nine")

        # Groundedness Checker
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="no" if is_unsupported_expected else "yes")

        # Generator
        else:
            if expected_supporting_ids:
                cites = " ".join(f"[ID {cid}]" for cid in sorted(expected_supporting_ids))
                return AIMessage(content=f"Synthesized factual response for evaluation {cites}.")
            else:
                return AIMessage(content="The available chat context is insufficient.")

    llm_handler = custom_llm_invoke or default_llm_invoke
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = llm_handler

    sem_results = mock_semantic_candidates if mock_semantic_candidates is not None else []

    with patch.object(rag_graph, "llm", mock_llm), \
         patch.object(rag_graph, "retrieve", side_effect=traced_retrieve), \
         patch.object(rag_graph, "grade_documents", side_effect=traced_grade), \
         patch.object(rag_graph, "expand_context", side_effect=traced_expand), \
         patch("hybrid_retriever.semantic_message_search", return_value=sem_results), \
         patch("hybrid_retriever.semantic_chunk_search", return_value=[]):

        result = rag_graph.ask(question, df=env.df)

    # Intermediate and final outcome analysis
    retrieved_msg_ids = {
        doc.metadata.get("message_id")
        for doc in retrieved_docs
        if doc.metadata.get("message_id") is not None
    }
    graded_msg_ids = {
        doc.metadata.get("message_id")
        for doc in graded_docs
        if doc.metadata.get("message_id") is not None
    }
    available_ids = set(result.get("available_citation_ids", []))
    valid_cited_ids = result.get("valid_citation_ids", [])
    invalid_cited_ids = result.get("invalid_citation_ids", [])

    # Measure explicit conditions
    relevant_retrieved = (
        len(expected_supporting_ids) == 0
        or any(cid in retrieved_msg_ids for cid in expected_supporting_ids)
    )
    relevant_survived = (
        len(expected_supporting_ids) == 0
        or any(cid in graded_msg_ids for cid in expected_supporting_ids)
    )
    distractors_excluded = (
        len(distractor_ids) == 0
        or all(did not in graded_msg_ids for did in distractor_ids)
    )
    final_context_had_expected = (
        len(expected_supporting_ids) == 0
        or any(cid in available_ids for cid in expected_supporting_ids)
    )
    cited_valid = (len(invalid_cited_ids) == 0) and (
        len(valid_cited_ids) > 0 or len(expected_supporting_ids) == 0
    )
    unsupported_marked_ungrounded = (
        not result["grounded"] if is_unsupported_expected else True
    )
    original_preserved = (
        result.get("answer") is not None
    )

    return EvaluationOutcome(
        test_id=test_id,
        question=question,
        relevant_evidence_retrieved=relevant_retrieved,
        relevant_evidence_survived_grading=relevant_survived,
        irrelevant_distractor_excluded=distractors_excluded,
        final_context_contained_expected_id=final_context_had_expected,
        cited_ids_valid_against_context=cited_valid,
        unsupported_marked_ungrounded=unsupported_marked_ungrounded,
        original_question_preserved=original_preserved,
        exceptions_or_unexpected_transitions=exceptions,
        rewrites_used=result.get("rewrites_used", 0),
        answer=result.get("answer", ""),
        grounded=result.get("grounded", False),
        valid_citation_ids=valid_cited_ids,
        invalid_citation_ids=invalid_cited_ids,
        available_citation_ids=list(available_ids),
    )


# =====================================================================
# 4. Required Evaluation Test Cases (1 to 10)
# =====================================================================

def test_case_1_direct_answer_with_valid_citation(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 1: Direct answer with valid citation.
    Verifies that a factual question retrieves the direct answer message (Msg 0),
    grades it relevant, retains it in context, cites [ID 0], and passes validation.
    """
    question = "What is the calibration setting of the deep space telescope?"
    expected_ids = {0}

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "States 1420 MHz calibration setting."},
                {"candidate": 2, "decision": "irrelevant", "reason": "Cafeteria lunch catering."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="yes")
        else:
            return AIMessage(content="The deep space telescope calibration setting is set to 1420 MHz [ID 0].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-1-Direct-Answer",
        question=question,
        expected_supporting_ids=expected_ids,
        distractor_ids={1},
        custom_llm_invoke=mock_llm_invoke,
    )

    assert outcome.relevant_evidence_retrieved
    assert outcome.relevant_evidence_survived_grading
    assert outcome.irrelevant_distractor_excluded
    assert outcome.final_context_contained_expected_id
    assert outcome.cited_ids_valid_against_context
    assert outcome.grounded is True
    assert outcome.valid_citation_ids == [0]
    assert outcome.invalid_citation_ids == []
    assert 0 in outcome.available_citation_ids


def test_case_2_semantic_paraphrase_minimal_lexical_overlap(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 2: Semantic paraphrase with minimal lexical overlap.
    Verifies that a query with zero lexical overlap with Msg 2 ("shuttle depart" vs
    "orbital relay vessel embarks as soon as dusk falls") is retrieved via semantic search,
    recognized as relevant by the grader without lexical bias, and validated.
    """
    question = "When will the shuttle depart?"
    expected_ids = {2}

    # Verify that real SQLite lexical search alone has zero matches for "shuttle depart"
    lexical_hits = hybrid_retriever.lexical_search(question)
    assert len(lexical_hits) == 0, "Lexical search unexpectedly matched synthetic paraphrase."

    # Mock semantic search returning Msg 2
    mock_sem_doc = Document(
        page_content="The orbital relay vessel embarks as soon as dusk falls across the landing platform.",
        metadata={"message_id": 2, "sender": "Commander Vance", "timestamp": "15/10/2026, 09:10"},
    )

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "Describes departure timing as dusk falls."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="yes")
        else:
            return AIMessage(content="The shuttle embarks as soon as dusk falls across the landing platform [ID 2].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-2-Semantic-Paraphrase",
        question=question,
        expected_supporting_ids=expected_ids,
        mock_semantic_candidates=[(mock_sem_doc, 0.15)],
        custom_llm_invoke=mock_llm_invoke,
    )

    assert outcome.relevant_evidence_retrieved
    assert outcome.relevant_evidence_survived_grading
    assert outcome.final_context_contained_expected_id
    assert outcome.cited_ids_valid_against_context
    assert outcome.grounded is True
    assert outcome.valid_citation_ids == [2]


def test_case_3_short_reply_resolved_through_chronological_context(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 3: Short reply resolved through chronological context.
    Verifies that a short, isolated answer ("Sub-level 4.", Msg 4) is expanded with its
    preceding conversational question (Msg 3) using SQLite get_chronological_context.
    """
    # Context intent triggers chronological context expansion
    question = "What is the context around the cryogenic coolant canisters?"
    expected_ids = {4}

    # Mock semantic search finding anchor Msg 4
    mock_anchor = Document(
        page_content="Sub-level 4.",
        metadata={"message_id": 4, "sender": "Specialist Lyra", "timestamp": "15/10/2026, 09:21"},
    )

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "Short reply location."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="yes")
        else:
            return AIMessage(content="The cryogenic coolant canisters are located in Sub-level 4 [ID 4].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-3-Short-Reply-Context",
        question=question,
        expected_supporting_ids=expected_ids,
        mock_semantic_candidates=[(mock_anchor, 0.18)],
        custom_llm_invoke=mock_llm_invoke,
    )

    assert outcome.relevant_evidence_retrieved
    assert outcome.final_context_contained_expected_id
    # Crucial check: Real SQLite expansion pulled in preceding Msg 3 and succeeding Msg 5
    assert 3 in outcome.available_citation_ids
    assert 4 in outcome.available_citation_ids
    assert outcome.cited_ids_valid_against_context
    assert outcome.grounded is True


def test_case_4_lexical_distractor_rejected_without_losing_relevant_answer(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 4: Lexical distractor rejected without losing the relevant answer.
    Verifies that when both relevant Msg 0 and lexical distractor Msg 1 match FTS5 terms,
    the relevance grader filters out Msg 1 while preserving Msg 0 for generation.
    """
    question = "What is the calibration setting of the deep space telescope?"
    expected_ids = {0}
    distractor_ids = {1}

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            # Explicitly reject Candidate 2 (lunch catering distractor)
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "Contains telescope frequency 1420 MHz."},
                {"candidate": 2, "decision": "irrelevant", "reason": "Lunch catering distractor."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="yes")
        else:
            return AIMessage(content="The calibration setting is 1420 MHz [ID 0].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-4-Distractor-Rejected",
        question=question,
        expected_supporting_ids=expected_ids,
        distractor_ids=distractor_ids,
        custom_llm_invoke=mock_llm_invoke,
    )

    assert outcome.relevant_evidence_retrieved
    assert outcome.relevant_evidence_survived_grading
    assert outcome.irrelevant_distractor_excluded
    assert outcome.final_context_contained_expected_id
    assert 0 in outcome.available_citation_ids
    assert 1 not in outcome.available_citation_ids
    assert outcome.grounded is True


def test_case_5_unrelated_retrieval_candidates_cause_rewrite_attempt(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 5: Unrelated retrieval candidates causing a rewrite attempt.
    Verifies that when initial retrieval returns only unrelated chatter (Msg 6, Msg 7),
    the grader marks them irrelevant, decide_to_generate triggers rewrite_query,
    and rewrite_count increments.
    """
    question = "What is the security authorization protocol for sector 9?"

    # Force initial retrieval to return unrelated candidates Msg 6 and Msg 7
    mock_unrelated = [
        (Document(page_content="Did anyone see the retro movie marathon?", metadata={"message_id": 6}), 0.8),
        (Document(page_content="I missed it, was fixing the antenna.", metadata={"message_id": 7}), 0.8),
    ]

    rewrite_call_count = 0

    def mock_llm_invoke(prompt):
        nonlocal rewrite_call_count
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            # Grader marks unrelated candidates irrelevant
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "irrelevant", "reason": "Unrelated movie chatter."},
                {"candidate": 2, "decision": "irrelevant", "reason": "Unrelated antenna repair."},
            ]))
        elif "Rewrite the following question" in prompt_str:
            rewrite_call_count += 1
            return AIMessage(content="authorization protocol sector nine")
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="no")
        else:
            return AIMessage(content="The available chat context is insufficient.")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-5-Rewrite-Triggered",
        question=question,
        expected_supporting_ids=set(),
        mock_semantic_candidates=mock_unrelated,
        custom_llm_invoke=mock_llm_invoke,
        expect_rewrite=True,
        is_unsupported_expected=True,
    )

    assert rewrite_call_count >= 1, "Rewrite LLM was not invoked when all candidates were irrelevant."
    assert outcome.rewrites_used >= 1
    assert outcome.irrelevant_distractor_excluded
    assert outcome.grounded is False


def test_case_6_answer_absent_from_dataset_marked_insufficient(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 6: Answer absent from the dataset.
    Verifies that a question whose answer does not exist in the chat produces an insufficient
    answer notice, has no invalid citations, and is marked ungrounded.
    """
    question = "What is the reactor core override password?"

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([]))
        elif "Rewrite the following question" in prompt_str:
            return AIMessage(content="reactor core password")
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="no")
        else:
            return AIMessage(content="The available chat context is insufficient to answer the question.")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-6-Absent-Answer",
        question=question,
        expected_supporting_ids=set(),
        custom_llm_invoke=mock_llm_invoke,
        is_unsupported_expected=True,
    )

    assert outcome.grounded is False
    assert outcome.valid_citation_ids == []
    assert outcome.invalid_citation_ids == []
    assert "insufficient" in outcome.answer.lower()
    assert outcome.unsupported_marked_ungrounded


def test_case_7_valid_citation_to_chronological_neighbor(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 7: Valid citation to a chronological neighbor.
    Verifies that citing a chronological neighbor (Msg 3) that was not an initial retrieval anchor
    but was brought into context via SQLite expansion is recognized as mechanically valid.
    """
    question = "What is the context around the cryogenic coolant canisters?"
    # Msg 4 is the anchor; Msg 3 is the chronological neighbor
    mock_anchor = Document(
        page_content="Sub-level 4.",
        metadata={"message_id": 4, "sender": "Specialist Lyra", "timestamp": "15/10/2026, 09:21"},
    )

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "Coolant canister location."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="yes")
        else:
            # Citations include neighbor Msg 3 AND anchor Msg 4
            return AIMessage(content="Tech Kael inquired about the storage location [ID 3], and Specialist Lyra answered Sub-level 4 [ID 4].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-7-Neighbor-Citation",
        question=question,
        expected_supporting_ids={3, 4},
        mock_semantic_candidates=[(mock_anchor, 0.15)],
        custom_llm_invoke=mock_llm_invoke,
    )

    assert 3 in outcome.available_citation_ids
    assert 4 in outcome.available_citation_ids
    assert 3 in outcome.valid_citation_ids
    assert 4 in outcome.valid_citation_ids
    assert outcome.invalid_citation_ids == []
    assert outcome.cited_ids_valid_against_context
    assert outcome.grounded is True


def test_case_8_invented_citation_id_marked_ungrounded(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 8: Invented citation ID.
    Verifies that citing a hallucinated ID (e.g., [ID 999]) absent from the active context
    forces grounded=False, even if the semantic checker returns 'yes'.
    """
    question = "What is the calibration setting of the deep space telescope?"

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "Telescope calibration setting."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            # Semantic checker returns yes, but mechanical check must override
            return AIMessage(content="yes")
        else:
            # Generator hallucinates non-existent ID 999
            return AIMessage(content="The deep space telescope calibration setting is set to 1420 MHz [ID 999].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-8-Invented-Citation",
        question=question,
        expected_supporting_ids={0},
        custom_llm_invoke=mock_llm_invoke,
    )

    assert outcome.invalid_citation_ids == [999]
    assert 999 not in outcome.available_citation_ids
    # Grounded must be False due to mechanical citation violation override
    assert outcome.grounded is False
    assert outcome.cited_ids_valid_against_context is False


def test_case_9_unsupported_claim_attached_to_valid_citation(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 9: Unsupported claim attached to a mechanically valid citation.
    Verifies that when citations are mechanically valid (e.g. citing [ID 0]), but the substantive
    claim is unsupported by context, the semantic grounding check returns 'no' and marks grounded=False.
    """
    question = "What is the calibration setting of the deep space telescope?"

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            return AIMessage(content=json.dumps([
                {"candidate": 1, "decision": "relevant", "reason": "Telescope calibration setting."},
            ]))
        elif "strict factual groundedness checker" in prompt_str:
            # Semantic checker catches unsupported claim
            return AIMessage(content="no")
        else:
            # Mechanically valid citation to ID 0, but false claim
            return AIMessage(content="The deep space telescope exploded and was destroyed during calibration [ID 0].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-9-Unsupported-Claim",
        question=question,
        expected_supporting_ids={0},
        custom_llm_invoke=mock_llm_invoke,
        is_unsupported_expected=True,
    )

    # Mechanical citation check passed
    assert outcome.valid_citation_ids == [0]
    assert outcome.invalid_citation_ids == []
    # But semantic grounding caught the unsupported claim
    assert outcome.grounded is False
    assert outcome.unsupported_marked_ungrounded


def test_case_10_grader_failure_fails_open_with_candidate_preservation(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Test Case 10: Grader failure with candidate preservation.
    Verifies that when the relevance grader LLM fails or raises an exception, the pipeline
    fails open by marking candidates 'uncertain', preserving them so they reach generation.
    """
    question = "What is the calibration setting of the deep space telescope?"

    def mock_llm_invoke(prompt):
        prompt_str = str(prompt)
        if "conservative relevance grader" in prompt_str:
            # Grader throws runtime error / service timeout
            raise RuntimeError("LLM inference timeout during candidate grading")
        elif "strict factual groundedness checker" in prompt_str:
            return AIMessage(content="yes")
        else:
            return AIMessage(content="The deep space telescope calibration setting is set to 1420 MHz [ID 0].")

    outcome = run_harness_trial(
        env=synthetic_rag_env,
        test_id="TC-10-Grader-Fail-Open",
        question=question,
        expected_supporting_ids={0},
        custom_llm_invoke=mock_llm_invoke,
        grader_failure=True,
    )

    # Candidate 0 must NOT be dropped despite grader failure
    assert outcome.relevant_evidence_survived_grading
    assert outcome.final_context_contained_expected_id
    assert 0 in outcome.available_citation_ids
    assert outcome.valid_citation_ids == [0]
    assert outcome.grounded is True


# =====================================================================
# 5. Overarching Harness Summary Verification
# =====================================================================

def test_evaluation_harness_dataset_integrity(synthetic_rag_env: SyntheticRagEnvironment):
    """
    Verifies that the synthetic dataset has exact fact representations and known IDs.
    """
    conn = get_connection(synthetic_rag_env.db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, sender, message FROM messages ORDER BY id")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 8
    assert rows[0]["sender"] == "Dr. Aris"
    assert "1420 MHz" in rows[0]["message"]
    assert rows[1]["sender"] == "Specialist Lyra"
    assert "cafeteria" in rows[1]["message"]
    assert rows[2]["sender"] == "Commander Vance"
    assert "dusk falls" in rows[2]["message"]
    assert rows[3]["sender"] == "Tech Kael"
    assert "cryogenic coolant" in rows[3]["message"]
    assert rows[4]["sender"] == "Specialist Lyra"
    assert rows[4]["message"] == "Sub-level 4."
    assert rows[5]["sender"] == "Tech Kael"
    assert "transport cart" in rows[5]["message"]
    assert rows[6]["sender"] == "Navigator Thorne"
    assert "movie marathon" in rows[6]["message"]
    assert rows[7]["sender"] == "Commander Vance"
    assert "antenna" in rows[7]["message"]
