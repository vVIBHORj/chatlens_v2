import pytest
from unittest.mock import patch
from langchain_core.documents import Document

from hybrid_retriever import (
    HybridCandidate,
    calculate_term_coverage,
    calculate_message_quality,
    calculate_evidence_score,
    extract_query_terms,
    SEMANTIC_WEIGHT,
    LEXICAL_WEIGHT,
    TERM_COVERAGE_WEIGHT,
    MESSAGE_QUALITY_WEIGHT,
    MULTI_SOURCE_BONUS,
)
from rag_graph import (
    retrieve,
    decide_to_generate,
    GraphState,
    MAX_REWRITES,
)


# =====================================================================
# Helper for creating a baseline GraphState
# =====================================================================

def make_test_state(documents=None, rewrite_count=0, question="Test query") -> GraphState:
    return {
        "question": question,
        "original_question": question,
        "documents": documents or [],
        "generation": "",
        "rewrite_count": rewrite_count,
        "grounded": False,
        "df": None,
        "scope": "specific",
        "primary_intent": "semantic_rag",
        "secondary_intents": [],
        "query_spec": None,
    }


# =====================================================================
# Scenario 1: Relevant evidence scoring
# =====================================================================

def test_scenario_1_relevant_evidence_scoring():
    """Synthetic query and candidate that clearly answers it with plausible semantic and lexical metadata."""
    query = "What database version was deployed?"
    query_terms = extract_query_terms(query)
    assert query_terms == ["database", "version", "deployed"]

    candidate = HybridCandidate(
        message_id=101,
        message="We deployed database version 16 to the server yesterday.",
        sender="DevOpsLead",
        timestamp="2026-10-10 12:00:00",
        semantic_rank=1,
        lexical_rank=1,
        semantic_evidence=1.0,  # 1.0 / 1
        lexical_evidence=1.0,   # 1.0 / 1
        multi_source_bonus=MULTI_SOURCE_BONUS,  # 0.10
        sources=["semantic_message", "lexical"],
    )

    # 1. Term coverage: all extracted terms are present in the candidate message
    coverage, matched = calculate_term_coverage(query_terms, candidate.message)
    assert coverage == 1.0
    assert matched == ["database", "version", "deployed"]
    candidate.term_coverage = coverage
    candidate.matched_terms = matched

    # 2. Message quality: 9 words -> 1.0
    quality = calculate_message_quality(candidate.message)
    assert quality == 1.0
    candidate.message_quality = quality

    # 3. Direct evidence score: max lexical matches (0.40) + max query matches (0.30) + agreement (0.10) + quality (0.10) = 0.90
    ev_score = calculate_evidence_score(candidate, query)
    assert ev_score == pytest.approx(0.9000)
    candidate.evidence_score = ev_score

    # 4. Hybrid score calculation
    candidate.score = (
        (candidate.semantic_evidence * SEMANTIC_WEIGHT)
        + (candidate.lexical_evidence * LEXICAL_WEIGHT)
        + (candidate.term_coverage * TERM_COVERAGE_WEIGHT)
        + ((candidate.message_quality - 1.0) * MESSAGE_QUALITY_WEIGHT)
        + candidate.multi_source_bonus
    )
    # Expected: (1.0 * 0.35) + (1.0 * 0.35) + (1.0 * 0.20) + (0.0 * 0.05) + 0.10 = 1.00
    assert candidate.score == pytest.approx(1.00)


# =====================================================================
# Scenario 2: Lexical distractor with nontrivial score
# =====================================================================

def test_scenario_2_lexical_distractor_nontrivial_score_without_relevance():
    """Candidate sharing several query terms but discussing something different.

    Demonstrates that lexical overlap produces a high score without proving semantic relevance.
    """
    query = "What database version was deployed?"
    query_terms = extract_query_terms(query)

    # Distractor explicitly mentions the query words but asserts a different topic
    distractor = HybridCandidate(
        message_id=102,
        message="I deployed a new version of the frontend UI, not the database.",
        sender="FrontendDev",
        timestamp="2026-10-10 12:05:00",
        semantic_rank=None,
        lexical_rank=2,
        semantic_evidence=0.0,
        lexical_evidence=0.5,  # 1.0 / 2
        multi_source_bonus=0.0,
        sources=["lexical"],
    )

    coverage, matched = calculate_term_coverage(query_terms, distractor.message)
    # All 3 terms ('database', 'version', 'deployed') appear in the distractor message text!
    assert coverage == 1.0
    assert matched == ["database", "version", "deployed"]
    distractor.term_coverage = coverage
    distractor.matched_terms = matched

    quality = calculate_message_quality(distractor.message)
    assert quality == 1.0
    distractor.message_quality = quality

    ev_score = calculate_evidence_score(distractor, query)
    # Evidence score is high (0.40 matched + 0.30 query words + 0.0 agreement + 0.10 quality = 0.80)
    # even though the candidate disclaims database deployment!
    assert ev_score >= 0.70
    assert ev_score == pytest.approx(0.8000)

    distractor.score = (
        (distractor.semantic_evidence * SEMANTIC_WEIGHT)
        + (distractor.lexical_evidence * LEXICAL_WEIGHT)
        + (distractor.term_coverage * TERM_COVERAGE_WEIGHT)
        + ((distractor.message_quality - 1.0) * MESSAGE_QUALITY_WEIGHT)
        + distractor.multi_source_bonus
    )
    # (0.0 * 0.35) + (0.5 * 0.35) + (1.0 * 0.20) + 0.0 + 0.0 = 0.175 + 0.20 = 0.375
    assert distractor.score == pytest.approx(0.375)


# =====================================================================
# Scenario 3: Semantic-only candidate with zero lexical overlap
# =====================================================================

def test_scenario_3_semantic_only_candidate_zero_lexical_overlap():
    """Relevant synthetic candidate with no lexical term overlap but plausible semantic metadata."""
    query = "Departure schedule?"
    query_terms = extract_query_terms(query)
    assert query_terms == ["Departure", "schedule"]

    # Candidate answers the question in natural conversational language without using 'departure' or 'schedule'
    candidate = HybridCandidate(
        message_id=103,
        message="Flight takes off at 08:30 tomorrow.",
        sender="TravelAgent",
        timestamp="2026-10-10 12:10:00",
        semantic_rank=2,
        lexical_rank=None,
        semantic_evidence=0.5,  # 1.0 / 2
        lexical_evidence=0.0,
        multi_source_bonus=0.0,
        sources=["semantic_message"],
    )

    coverage, matched = calculate_term_coverage(query_terms, candidate.message)
    # Term coverage is strictly 0.0
    assert coverage == 0.0
    assert matched == []
    candidate.term_coverage = coverage
    candidate.matched_terms = matched

    quality = calculate_message_quality(candidate.message)
    assert quality == 0.85  # 6 words (4-7 words range)
    candidate.message_quality = quality

    # Hybrid score remains strictly positive from semantic evidence
    candidate.score = (
        (candidate.semantic_evidence * SEMANTIC_WEIGHT)
        + (candidate.lexical_evidence * LEXICAL_WEIGHT)
        + (candidate.term_coverage * TERM_COVERAGE_WEIGHT)
        + ((candidate.message_quality - 1.0) * MESSAGE_QUALITY_WEIGHT)
        + candidate.multi_source_bonus
    )
    # 0.5 * 0.35 + ((0.85 - 1.0) * 0.05) = 0.175 - 0.0075 = 0.1675
    assert candidate.score == pytest.approx(0.1675)

    # Direct evidence score still awards message quality even with zero lexical matches
    ev_score = calculate_evidence_score(candidate, query)
    assert ev_score == pytest.approx(0.0850)  # 0.0 matched + 0.0 query words + 0.0 agreement + (0.85 * 0.10)
    # Confirms zero lexical overlap does not cause candidate score to be zero or negative


# =====================================================================
# Scenario 4: Irrelevant long message and negative score
# =====================================================================

def test_scenario_4_irrelevant_long_message_and_negative_score():
    """Verify unconditional message-quality contribution in evidence_score and negative hybrid_score possibility."""
    # Part A: Long conversational filler message with 0 query overlap (no stopword overlap)
    query_unrelated = "Quantum telemetry calibration?"
    long_filler = HybridCandidate(
        message_id=104,
        message="Fresh organic apples and bananas were harvested yesterday from orchards.",
        sender="Roommate",
        timestamp="2026-10-10 12:15:00",
        semantic_rank=None,
        lexical_rank=None,
        sources=[],
    )
    long_filler.message_quality = calculate_message_quality(long_filler.message)
    assert long_filler.message_quality == 1.0

    ev_score = calculate_evidence_score(long_filler, query_unrelated)
    # Verified behavior: long candidate with no retrieval evidence and no query matches
    # receives 0.0000 evidence score (unconditional quality contribution removed).
    assert ev_score == pytest.approx(0.0000)

    # Part B: Stopword overlap elevates evidence_score further
    # Query words > 2 letters like 'the' are matched without stopword filtering
    query_with_stopword = "Where is the rover?"
    filler_with_the = HybridCandidate(
        message_id=1041,
        message="Fresh organic apples and bananas were harvested yesterday from the farm.",
        sender="Roommate",
        timestamp="2026-10-10 12:15:00",
        message_quality=1.0,
    )
    ev_score_stopword = calculate_evidence_score(filler_with_the, query_with_stopword)
    # 'the' matches (0.15) + quality (0.10) = 0.25
    assert ev_score_stopword == pytest.approx(0.2500)

    # Part C: Demonstrate that candidate.score CAN be negative
    short_candidate = HybridCandidate(
        message_id=105,
        message="ok",  # 1 word -> quality = 0.25 -> quality penalty = (0.25 - 1.0) * 0.05 = -0.0375
        semantic_rank=20,  # 1/20 = 0.05 -> semantic_evidence * 0.35 = 0.0175
        semantic_evidence=0.05,
        lexical_rank=None,
        lexical_evidence=0.0,
        term_coverage=0.0,
        multi_source_bonus=0.0,
        message_quality=0.25,
    )

    short_candidate.score = (
        (short_candidate.semantic_evidence * SEMANTIC_WEIGHT)
        + (short_candidate.lexical_evidence * LEXICAL_WEIGHT)
        + (short_candidate.term_coverage * TERM_COVERAGE_WEIGHT)
        + ((short_candidate.message_quality - 1.0) * MESSAGE_QUALITY_WEIGHT)
        + short_candidate.multi_source_bonus
    )
    # 0.0175 - 0.0375 = -0.0200
    assert short_candidate.score == pytest.approx(-0.0200)
    assert short_candidate.score < 0.0


# =====================================================================
# Scenario 5: Mixed candidates independent representation
# =====================================================================

def test_scenario_5_mixed_candidates_independent_representation():
    """Pool containing one relevant candidate and several unrelated candidates."""
    query = "Where was the conference held?"
    query_terms = extract_query_terms(query)

    # 1. Relevant candidate
    cand_relevant = HybridCandidate(
        message_id=201,
        message="The conference was held at the Berlin Convention Hall on main street.",
        semantic_rank=1,
        lexical_rank=1,
        semantic_evidence=1.0,
        lexical_evidence=1.0,
        multi_source_bonus=MULTI_SOURCE_BONUS,
        sources=["semantic_message", "lexical"],
    )
    cov1, m1 = calculate_term_coverage(query_terms, cand_relevant.message)
    cand_relevant.term_coverage = cov1
    cand_relevant.matched_terms = m1
    cand_relevant.message_quality = calculate_message_quality(cand_relevant.message)
    cand_relevant.score = (
        (cand_relevant.semantic_evidence * SEMANTIC_WEIGHT)
        + (cand_relevant.lexical_evidence * LEXICAL_WEIGHT)
        + (cand_relevant.term_coverage * TERM_COVERAGE_WEIGHT)
        + ((cand_relevant.message_quality - 1.0) * MESSAGE_QUALITY_WEIGHT)
        + cand_relevant.multi_source_bonus
    )
    cand_relevant.evidence_score = calculate_evidence_score(cand_relevant, query)

    # 2. Lexical distractor
    cand_distractor = HybridCandidate(
        message_id=202,
        message="I held my breath during the press conference about the movie release.",
        semantic_rank=15,
        lexical_rank=2,
        semantic_evidence=1.0 / 15,
        lexical_evidence=0.5,
        multi_source_bonus=MULTI_SOURCE_BONUS,
        sources=["semantic_message", "lexical"],
    )
    cov2, m2 = calculate_term_coverage(query_terms, cand_distractor.message)
    cand_distractor.term_coverage = cov2
    cand_distractor.matched_terms = m2
    cand_distractor.message_quality = calculate_message_quality(cand_distractor.message)
    cand_distractor.score = (
        (cand_distractor.semantic_evidence * SEMANTIC_WEIGHT)
        + (cand_distractor.lexical_evidence * LEXICAL_WEIGHT)
        + (cand_distractor.term_coverage * TERM_COVERAGE_WEIGHT)
        + ((cand_distractor.message_quality - 1.0) * MESSAGE_QUALITY_WEIGHT)
        + cand_distractor.multi_source_bonus
    )
    cand_distractor.evidence_score = calculate_evidence_score(cand_distractor, query)

    # 3. Unrelated short acknowledgement
    cand_short = HybridCandidate(
        message_id=203,
        message="ok see you",
        semantic_rank=20,
        lexical_rank=None,
        semantic_evidence=0.05,
        lexical_evidence=0.0,
        multi_source_bonus=0.0,
        sources=["semantic_message"],
    )
    cov3, m3 = calculate_term_coverage(query_terms, cand_short.message)
    cand_short.term_coverage = cov3
    cand_short.matched_terms = m3
    cand_short.message_quality = calculate_message_quality(cand_short.message)
    cand_short.score = (
        (cand_short.semantic_evidence * SEMANTIC_WEIGHT)
        + (cand_short.lexical_evidence * LEXICAL_WEIGHT)
        + (cand_short.term_coverage * TERM_COVERAGE_WEIGHT)
        + ((cand_short.message_quality - 1.0) * MESSAGE_QUALITY_WEIGHT)
        + cand_short.multi_source_bonus
    )
    cand_short.evidence_score = calculate_evidence_score(cand_short, query)

    candidates = [cand_relevant, cand_distractor, cand_short]
    sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

    # Verify candidates maintain distinct IDs, independent evidence components, and sorting
    assert sorted_candidates[0].message_id == 201
    assert sorted_candidates[1].message_id == 202
    assert sorted_candidates[2].message_id == 203
    assert sorted_candidates[0].score > sorted_candidates[1].score > sorted_candidates[2].score


# =====================================================================
# Scenario 6: Graph routing for empty vs. nonempty retrieval
# =====================================================================

def test_scenario_6_graph_routing_empty_vs_nonempty_retrieval():
    """Verify decide_to_generate routing on empty vs. nonempty candidate lists."""
    # 1. Empty retrieval at attempt 0 routes to rewrite_query
    state_empty_0 = make_test_state(documents=[], rewrite_count=0)
    assert decide_to_generate(state_empty_0) == "rewrite_query"

    # 2. Empty retrieval at attempt 1 routes to rewrite_query
    state_empty_1 = make_test_state(documents=[], rewrite_count=1)
    assert decide_to_generate(state_empty_1) == "rewrite_query"

    # 3. Empty retrieval at attempt MAX_REWRITES (2) stops rewriting and forces generate
    state_empty_2 = make_test_state(documents=[], rewrite_count=MAX_REWRITES)
    assert decide_to_generate(state_empty_2) == "generate"

    # 4. Nonempty retrieval (even with 1 totally irrelevant document) routes to generate!
    # Documents existing behavior: nonempty candidate list skips rewriting unconditionally.
    irrelevant_doc = Document(
        page_content="[ID 999] [2026-10-10] User: completely unrelated random chat message",
        metadata={"source": "hybrid_search", "message_id": 999, "evidence_score": 0.10},
    )
    state_nonempty = make_test_state(documents=[irrelevant_doc], rewrite_count=0)
    assert decide_to_generate(state_nonempty) == "generate"


# =====================================================================
# Scenario 7: Chunk metadata compatibility and absent score fields
# =====================================================================

@patch("rag_graph.hybrid_search")
def test_scenario_7_chunk_metadata_compatibility_and_scale_differences(mock_hybrid_search):
    """Verify representation of message and chunk candidates in retrieve().

    Confirms chunk documents lack hybrid_score/evidence_score without raising KeyError.
    """
    message_cand = HybridCandidate(
        message_id=301,
        message="Alpha protocol initiated.",
        sender="AgentA",
        timestamp="2026-10-10 14:00:00",
        score=0.6500,
        evidence_score=0.7000,
        semantic_rank=1,
        lexical_rank=1,
        semantic_distance=0.15,
        matched_terms=["alpha", "protocol"],
        sources=["semantic_message", "lexical"],
    )

    chunk_cand = {
        "rank": 1,
        "distance": 0.2345,
        "chunk_id": 401,
        "start_message_id": 1,
        "end_message_id": 15,
        "episode_id": 10,
        "participants": "AgentA, AgentB",
        "document": "Full conversation transcript chunk between AgentA and AgentB.",
    }

    mock_hybrid_search.return_value = {
        "query": "alpha protocol",
        "candidates": [message_cand],
        "chunks": [chunk_cand],
    }

    initial_state = make_test_state(question="alpha protocol")
    result_state = retrieve(initial_state)
    documents = result_state["documents"]

    assert len(documents) == 2

    # Document 0: message candidate
    msg_doc = documents[0]
    assert msg_doc.metadata["source"] == "hybrid_search"
    assert msg_doc.metadata["message_id"] == 301
    assert msg_doc.metadata["hybrid_score"] == 0.6500
    assert msg_doc.metadata["evidence_score"] == 0.7000
    assert msg_doc.metadata["matched_terms"] == ["alpha", "protocol"]

    # Document 1: chunk candidate
    chunk_doc = documents[1]
    assert chunk_doc.metadata["source"] == "conversation_chunk"
    assert chunk_doc.metadata["chunk_rank"] == 1
    assert chunk_doc.metadata["semantic_distance"] == 0.2345

    # Confirm score fields are absent on chunk document without KeyError when using .get()
    assert chunk_doc.metadata.get("hybrid_score") is None
    assert chunk_doc.metadata.get("evidence_score") is None
    assert chunk_doc.metadata.get("matched_terms") is None

    # Confirm that direct dict key access on chunk_doc raises KeyError
    with pytest.raises(KeyError):
        _ = chunk_doc.metadata["evidence_score"]
    with pytest.raises(KeyError):
        _ = chunk_doc.metadata["hybrid_score"]

    # Document that chunk candidate scale (Chroma distance float >= 0.0) is
    # fundamentally distinct from message candidate hybrid_score (engineered composite rank/coverage).
    assert isinstance(chunk_doc.metadata["semantic_distance"], float)
    assert isinstance(msg_doc.metadata["hybrid_score"], float)
