from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import re

from vectorstore import (
    load_message_vectorstore,
    load_chunk_vectorstore,
)
from chat_database import search_messages


# ============================================================
# CONFIG
# ============================================================

MESSAGE_SEMANTIC_TOP_K = 20
LEXICAL_TOP_K = 20
CHUNK_TOP_K = 5

# RRF is still useful as one piece of evidence,
# but it is no longer the final ranking mechanism.
RRF_K = 60

# Weights for final evidence score.
SEMANTIC_WEIGHT = 0.35
LEXICAL_WEIGHT = 0.35
TERM_COVERAGE_WEIGHT = 0.20
MULTI_SOURCE_BONUS = 0.10


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "what",
    "whats",
    "what's",
    "who",
    "whom",
    "whose",
    "which",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "am",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "about",
    "discuss",
    "discussed",
    "discussing",
    "tell",
    "told",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "have",
    "has",
    "had",
    "was",
    "were",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "and",
    "or",
}


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class HybridCandidate:

    message_id: int

    message: Optional[str] = None
    sender: Optional[str] = None
    timestamp: Optional[Any] = None

    # Final evidence score
    score: float = 0.0

    # Individual evidence components
    semantic_evidence: float = 0.0
    lexical_evidence: float = 0.0
    term_coverage: float = 0.0
    multi_source_bonus: float = 0.0

    # Original retrieval information
    semantic_rank: Optional[int] = None
    lexical_rank: Optional[int] = None

    semantic_distance: Optional[float] = None

    sources: List[str] = field(default_factory=list)

    # For debugging
    matched_terms: List[str] = field(default_factory=list)


# ============================================================
# QUERY TERMS
# ============================================================

def extract_query_terms(query: str) -> List[str]:
    """
    Extract meaningful terms from the query.

    This intentionally removes conversational question words
    such as "what", "who", "how", etc.
    """

    if not query:
        return []

    words = re.findall(
        r"[\w@#']+",
        query,
        flags=re.UNICODE,
    )

    terms = []

    for word in words:

        normalized = word.lower().strip()

        if not normalized:
            continue

        if normalized in STOPWORDS:
            continue

        if len(normalized) <= 1:
            continue

        if normalized not in [
            term.lower()
            for term in terms
        ]:
            terms.append(word)

    return terms


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Optional[str]) -> str:

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text).lower(),
    ).strip()


# ============================================================
# TERM COVERAGE
# ============================================================

def calculate_term_coverage(
    query_terms: List[str],
    message: str,
):
    """
    Calculate how many meaningful query terms occur in the
    retrieved message.

    Example:

        query:
        "What laptop was EXODIA considering buying?"

        terms:
        ["laptop", "EXODIA", "considering", "buying"]

    """

    if not query_terms:
        return 0.0, []

    normalized_message = normalize_text(message)

    matched = []

    for term in query_terms:

        normalized_term = normalize_text(term)

        if not normalized_term:
            continue

        if normalized_term in normalized_message:
            matched.append(term)

    coverage = len(matched) / len(query_terms)

    return coverage, matched


# ============================================================
# RRF
# ============================================================

def rrf_score(
    rank: int,
    k: int = RRF_K,
):
    return 1.0 / (k + rank)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_message_search(
    query: str,
    top_k: int = MESSAGE_SEMANTIC_TOP_K,
):

    store = load_message_vectorstore()

    return store.similarity_search_with_score(
        query,
        k=top_k,
    )


# ============================================================
# LEXICAL SEARCH
# ============================================================

def lexical_search(
    query: str,
    top_k: int = LEXICAL_TOP_K,
):

    return search_messages(
        query,
        limit=top_k,
    )


# ============================================================
# CHUNK SEARCH
# ============================================================

def semantic_chunk_search(
    query: str,
    top_k: int = CHUNK_TOP_K,
):

    store = load_chunk_vectorstore()

    return store.similarity_search_with_score(
        query,
        k=top_k,
    )


# ============================================================
# MESSAGE ID EXTRACTION
# ============================================================

def extract_message_id(metadata):

    if not metadata:
        return None

    possible_keys = [
        "message_id",
        "id",
        "original_message_id",
    ]

    for key in possible_keys:

        value = metadata.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            pass

    return None


# ============================================================
# DATABASE ROW HELPERS
# ============================================================

def get_row_value(
    row,
    key,
    default=None,
):

    try:
        return row[key]
    except (
        KeyError,
        TypeError,
        IndexError,
    ):
        pass

    try:
        return getattr(
            row,
            key,
        )
    except AttributeError:
        return default


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    message_top_k: int = MESSAGE_SEMANTIC_TOP_K,
    lexical_top_k: int = LEXICAL_TOP_K,
    chunk_top_k: int = CHUNK_TOP_K,
):

    query_terms = extract_query_terms(query)

    candidates: Dict[
        int,
        HybridCandidate,
    ] = {}

    # ========================================================
    # 1. SEMANTIC RETRIEVAL
    # ========================================================

    semantic_results = semantic_message_search(
        query,
        top_k=message_top_k,
    )

    # ========================================================
    # 2. PROCESS SEMANTIC RESULTS
    # ========================================================

    for rank, (
        document,
        distance,
    ) in enumerate(
        semantic_results,
        start=1,
    ):

        metadata = document.metadata or {}

        message_id = extract_message_id(
            metadata
        )

        if message_id is None:
            continue

        if message_id not in candidates:

            candidates[message_id] = (
                HybridCandidate(
                    message_id=message_id
                )
            )

        candidate = candidates[
            message_id
        ]

        candidate.semantic_rank = rank
        candidate.semantic_distance = distance

        # RRF-like normalized semantic evidence.
        candidate.semantic_evidence = (
            rrf_score(rank)
        )

        candidate.message = (
            metadata.get("message")
            or document.page_content
        )

        candidate.sender = metadata.get(
            "sender"
        )

        candidate.timestamp = metadata.get(
            "timestamp"
        )

        if "semantic_message" not in candidate.sources:

            candidate.sources.append(
                "semantic_message"
            )

    # ========================================================
    # 3. LEXICAL RETRIEVAL
    # ========================================================

    lexical_results = lexical_search(
        query,
        top_k=lexical_top_k,
    )

    # ========================================================
    # 4. PROCESS LEXICAL RESULTS
    # ========================================================

    for rank, row in enumerate(
        lexical_results,
        start=1,
    ):

        message_id = get_row_value(
            row,
            "id",
        )

        if message_id is None:
            continue

        try:
            message_id = int(message_id)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if message_id not in candidates:

            candidates[message_id] = (
                HybridCandidate(
                    message_id=message_id
                )
            )

        candidate = candidates[
            message_id
        ]

        candidate.lexical_rank = rank

        candidate.lexical_evidence = (
            rrf_score(rank)
        )

        if "lexical" not in candidate.sources:

            candidate.sources.append(
                "lexical"
            )

        # Fill information from SQLite.
        db_message = get_row_value(
            row,
            "message",
        )

        if db_message:

            candidate.message = (
                candidate.message
                or db_message
            )

        db_sender = get_row_value(
            row,
            "sender",
        )

        if db_sender:

            candidate.sender = (
                candidate.sender
                or db_sender
            )

        db_timestamp = get_row_value(
            row,
            "timestamp",
        )

        if db_timestamp:

            candidate.timestamp = (
                candidate.timestamp
                or db_timestamp
            )

    # ========================================================
    # 5. CALCULATE TERM COVERAGE
    # ========================================================

    for candidate in candidates.values():

        coverage, matched = (
            calculate_term_coverage(
                query_terms,
                candidate.message or "",
            )
        )

        candidate.term_coverage = coverage
        candidate.matched_terms = matched

    # ========================================================
    # 6. MULTI-SOURCE BONUS
    # ========================================================

    for candidate in candidates.values():

        has_semantic = (
            candidate.semantic_rank is not None
        )

        has_lexical = (
            candidate.lexical_rank is not None
        )

        if has_semantic and has_lexical:

            candidate.multi_source_bonus = (
                MULTI_SOURCE_BONUS
            )

    # ========================================================
    # 7. FINAL SCORE
    # ========================================================

    for candidate in candidates.values():

        candidate.score = (

            (
                candidate.semantic_evidence
                * SEMANTIC_WEIGHT
            )

            +

            (
                candidate.lexical_evidence
                * LEXICAL_WEIGHT
            )

            +

            (
                candidate.term_coverage
                * TERM_COVERAGE_WEIGHT
            )

            +

            candidate.multi_source_bonus
        )

    # ========================================================
    # 8. SORT
    # ========================================================

    ranked_candidates = sorted(
        candidates.values(),
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    # ========================================================
    # 9. CONVERSATION CHUNKS
    # ========================================================

    chunk_results = semantic_chunk_search(
        query,
        top_k=chunk_top_k,
    )

    chunks = []

    for rank, (
        document,
        distance,
    ) in enumerate(
        chunk_results,
        start=1,
    ):

        metadata = document.metadata or {}

        chunks.append(
            {
                "rank": rank,
                "distance": distance,
                "start_message_id": metadata.get(
                    "start_message_id"
                ),
                "end_message_id": metadata.get(
                    "end_message_id"
                ),
                "episode_id": metadata.get(
                    "episode_id"
                ),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "participants": metadata.get(
                    "participants"
                ),
                "document": document.page_content,
            }
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "query": query,
        "query_terms": query_terms,
        "candidates": ranked_candidates,
        "chunks": chunks,
        "semantic_message_results": semantic_results,
        "lexical_results": lexical_results,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_hybrid_results(
    result,
    top_k=10,
):

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL — EVIDENCE RANKING")
    print("=" * 80)

    print(
        f"\nQuery: {result['query']}"
    )

    print(
        f"Query terms: "
        f"{result['query_terms']}"
    )

    print("\n" + "-" * 80)
    print("RANKED CANDIDATES")
    print("-" * 80)

    candidates = result[
        "candidates"
    ][:top_k]

    if not candidates:

        print("No candidates found.")

    for display_rank, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\n#{display_rank} "
            f"message_id={candidate.message_id}"
        )

        print(
            f"   FINAL SCORE: "
            f"{candidate.score:.6f}"
        )

        print(
            f"   Semantic evidence: "
            f"{candidate.semantic_evidence:.6f}"
        )

        print(
            f"   Lexical evidence: "
            f"{candidate.lexical_evidence:.6f}"
        )

        print(
            f"   Term coverage: "
            f"{candidate.term_coverage:.2f}"
        )

        print(
            f"   Multi-source bonus: "
            f"{candidate.multi_source_bonus:.2f}"
        )

        print(
            f"   Sources: "
            f"{', '.join(candidate.sources)}"
        )

        if candidate.semantic_rank is not None:

            print(
                f"   Semantic rank: "
                f"#{candidate.semantic_rank}"
            )

        if candidate.lexical_rank is not None:

            print(
                f"   Lexical rank: "
                f"#{candidate.lexical_rank}"
            )

        if candidate.semantic_distance is not None:

            print(
                f"   Semantic distance: "
                f"{candidate.semantic_distance:.6f}"
            )

        if candidate.matched_terms:

            print(
                f"   Matched terms: "
                f"{candidate.matched_terms}"
            )

        if candidate.sender:

            print(
                f"   Sender: "
                f"{candidate.sender}"
            )

        if candidate.timestamp:

            print(
                f"   Timestamp: "
                f"{candidate.timestamp}"
            )

        if candidate.message:

            print(
                f"   Message: "
                f"{candidate.message}"
            )

    # ========================================================
    # CHUNKS
    # ========================================================

    print("\n" + "-" * 80)
    print("CONVERSATION CHUNK SUPPORT")
    print("-" * 80)

    if not result["chunks"]:

        print("No chunk results.")

    else:

        for chunk in result["chunks"]:

            print(
                f"\nChunk #{chunk['rank']}"
            )

            print(
                f"   Distance: "
                f"{chunk['distance']:.6f}"
            )

            print(
                f"   Message range: "
                f"{chunk['start_message_id']} → "
                f"{chunk['end_message_id']}"
            )

            print(
                f"   Episode: "
                f"{chunk['episode_id']}"
            )

    print("\n" + "=" * 80)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    TEST_QUERIES = [

        "What laptop was EXODIA considering buying?",

        "Who is udit?",

        "Who was EXODIA's girlfriend?",

        "What did we discuss about sports?",
    ]

    for query in TEST_QUERIES:

        result = hybrid_search(query)

        print_hybrid_results(
            result,
            top_k=10,
        )