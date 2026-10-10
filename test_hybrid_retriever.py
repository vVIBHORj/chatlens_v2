import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from hybrid_retriever import (
    QUERY_EXPANSIONS,
    expand_query_terms,
    build_semantic_query,
    hybrid_search,
    HybridCandidate,
)


# =====================================================================
# 1. Verification of expand_query_terms pass-through
# =====================================================================

def test_expand_query_terms_no_hardcoded_or_hinglish_expansions():
    """Verify expand_query_terms does NOT inject sports, relationship, laptop, brand, or Hinglish terms."""
    # Former trigger words
    terms = ["sports", "laptop", "girlfriend", "boyfriend", "friend"]
    expanded = expand_query_terms(terms)

    # Must preserve exact original input terms
    assert expanded == terms

    # Check for forbidden hardcoded terms that used to be in QUERY_EXPANSIONS
    forbidden_terms = {
        # sports
        "sport", "game", "khel", "tennis", "badminton", "cricket", "football", "basketball",
        # laptop & brands
        "notebook", "computer", "asus", "hp", "omen", "lenovo", "dell", "acer",
        # relationship & Hinglish
        "girl", "bandi", "wali", "meri wali", "relationship", "dating", "partner",
        "ladka", "banda", "wala", "mera wala", "dost", "yaar", "buddy",
    }
    for item in expanded:
        assert item.lower() not in forbidden_terms or item in terms, (
            f"Unexpected expansion term injected: {item}"
        )
    # Ensure no extra terms were added beyond the input
    assert len(expanded) == len(terms)


def test_expand_query_terms_deduplication_and_empty():
    """Verify expand_query_terms handles duplicates and empty input correctly."""
    assert expand_query_terms([]) == []
    assert expand_query_terms(["laptop", "omen", "laptop"]) == ["laptop", "omen"]


# =====================================================================
# 2. Verification of build_semantic_query
# =====================================================================

def test_build_semantic_query_preserves_original_query():
    """Verify build_semantic_query preserves the original query without appending expansions."""
    queries_and_terms = [
        ("What did we discuss about sports?", ["discuss", "sports"]),
        ("What laptop was EXODIA considering buying?", ["laptop", "exodia", "considering", "buying"]),
        ("Who was EXODIA's girlfriend?", ["exodia", "girlfriend"]),
        ("Tell me about my friend", ["tell", "friend"]),
    ]

    for raw_query, terms in queries_and_terms:
        result = build_semantic_query(raw_query, terms)
        assert result == raw_query, (
            f"Expected '{raw_query}', got '{result}'"
        )


# =====================================================================
# 3. Arbitrary and Multilingual Queries
# =====================================================================

def test_arbitrary_concepts_and_multilingual_pass_through():
    """Verify arbitrary concepts and multilingual query strings pass through without additions."""
    # Arbitrary scientific / domain concepts
    scientific_terms = ["quantum", "telemetry", "spectroscopy", "orbital"]
    assert expand_query_terms(scientific_terms) == scientific_terms

    # Multilingual queries
    multilingual_cases = [
        ("¿Cuál fue la última decisión tomada sobre el proyecto?", ["última", "decisión", "proyecto"]),
        ("Comment configurer le serveur de base de données ?", ["configurer", "serveur", "base", "données"]),
        ("Kal sham ko movie dekhne kaun chalega?", ["sham", "movie", "dekhne", "chalega"]),
    ]

    for q, terms in multilingual_cases:
        assert expand_query_terms(terms) == terms
        assert build_semantic_query(q, terms) == q


def test_query_expansions_dict_is_empty():
    """Verify QUERY_EXPANSIONS contains no default hardcoded topic entries."""
    assert isinstance(QUERY_EXPANSIONS, dict)
    assert len(QUERY_EXPANSIONS) == 0


# =====================================================================
# 4. hybrid_search invokes retrieval paths with unexpanded terms
# =====================================================================

@patch("hybrid_retriever.semantic_chunk_search")
@patch("hybrid_retriever.lexical_search")
@patch("hybrid_retriever.semantic_message_search")
def test_hybrid_search_invokes_paths_with_unexpanded_terms(
    mock_semantic_msg,
    mock_lexical,
    mock_chunk,
):
    """Verify hybrid_search calls semantic, lexical, and chunk retrieval with unexpanded query and terms."""
    mock_semantic_msg.return_value = []
    mock_lexical.return_value = []
    mock_chunk.return_value = []

    query = "What did we discuss about sports?"
    result = hybrid_search(query)

    # 1. Semantic message search called with the exact unexpanded query
    mock_semantic_msg.assert_called_once()
    called_semantic_query = mock_semantic_msg.call_args[0][0]
    assert called_semantic_query == query
    assert "badminton" not in called_semantic_query
    assert "khel" not in called_semantic_query

    # 2. Lexical search called with unexpanded query_terms
    mock_lexical.assert_called_once()
    _, lexical_kwargs = mock_lexical.call_args
    assert lexical_kwargs["query_terms"] == ["sports"]
    assert "tennis" not in lexical_kwargs["query_terms"]
    assert "game" not in lexical_kwargs["query_terms"]
    assert "badminton" not in lexical_kwargs["query_terms"]

    # 3. Semantic chunk search called with the raw query
    mock_chunk.assert_called_once_with(query, top_k=5)

    # 4. Result dict metadata
    assert result["query"] == query
    assert result["expanded_query_terms"] == ["sports"]
    assert result["semantic_query"] == query


# =====================================================================
# 5. Candidate deduplication, multi-source bonus, and ranking
# =====================================================================

@patch("hybrid_retriever.semantic_chunk_search")
@patch("hybrid_retriever.lexical_search")
@patch("hybrid_retriever.semantic_message_search")
def test_hybrid_search_deduplication_and_ranking_preserved(
    mock_semantic_msg,
    mock_lexical,
    mock_chunk,
):
    """Verify candidate deduplication, multi-source bonuses, and ranking remain intact."""
    # Semantic message results: Message 10 (rank 1), Message 20 (rank 2)
    doc10 = Document(page_content="We discussed basketball and sports.", metadata={"message_id": 10, "sender": "Alice"})
    doc20 = Document(page_content="Sports are great exercise.", metadata={"message_id": 20, "sender": "Bob"})
    mock_semantic_msg.return_value = [
        (doc10, 0.15),
        (doc20, 0.25),
    ]

    # Lexical results: Message 20 (rank 1), Message 30 (rank 2)
    # Message 20 appears in both semantic and lexical -> duplicate merge!
    row20 = {"id": 20, "message": "Sports are great exercise.", "sender": "Bob", "timestamp": "2026-10-10 10:00"}
    row30 = {"id": 30, "message": "Just watching sports on TV.", "sender": "Charlie", "timestamp": "2026-10-10 11:00"}
    mock_lexical.return_value = [row20, row30]

    # Chunk results: Chunk 101 (rank 1), Chunk 101 again (rank 2 - duplicate), Chunk 102 (rank 3)
    chunk1_doc = Document(page_content="Chunk 1", metadata={"chunk_id": 101})
    chunk1_dup = Document(page_content="Chunk 1 dup", metadata={"chunk_id": 101})
    chunk2_doc = Document(page_content="Chunk 2", metadata={"chunk_id": 102})
    mock_chunk.return_value = [
        (chunk1_doc, 0.2),
        (chunk1_dup, 0.3),
        (chunk2_doc, 0.4),
    ]

    result = hybrid_search("sports")
    candidates = result["candidates"]
    chunks = result["chunks"]

    # Candidate deduplication checks
    candidate_ids = [c.message_id for c in candidates]
    assert len(candidate_ids) == len(set(candidate_ids)) == 3
    assert set(candidate_ids) == {10, 20, 30}

    # Candidate 20 was found by both sources
    cand20 = next(c for c in candidates if c.message_id == 20)
    assert "semantic_message" in cand20.sources
    assert "lexical" in cand20.sources
    assert cand20.multi_source_bonus > 0.0

    # Ranking is strictly descending by candidate.score
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)

    # Chunk deduplication checks
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert chunk_ids == [101, 102]
