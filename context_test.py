from hybrid_retriever import hybrid_search
from context_expander import (
    expand_candidates,
    print_context_regions,
)


TEST_QUERIES = [
    "What laptop was EXODIA considering buying?",
    "Who is udit?",
    "Who was EXODIA's girlfriend?",
    "What did we discuss about sports?",
]


def extract_candidates(result, top_k=10):
    raw_candidates = result.get("candidates", [])

    if not isinstance(raw_candidates, list):
        raise TypeError(
            "Expected result['candidates'] to be a list, "
            f"got {type(raw_candidates).__name__}"
        )

    candidates = []

    for index, candidate in enumerate(
        raw_candidates[:top_k],
        start=1,
    ):
        if not hasattr(candidate, "message_id"):
            print(
                f"WARNING: Candidate #{index} "
                f"missing message_id - skipped"
            )
            continue

        message_id = candidate.message_id

        if message_id is None:
            print(
                f"WARNING: Candidate #{index} "
                f"has message_id=None - skipped"
            )
            continue

        try:
            message_id = int(message_id)
        except (TypeError, ValueError):
            print(
                f"WARNING: Candidate #{index} "
                f"has invalid message_id={message_id!r} - skipped"
            )
            continue

        if message_id <= 0:
            print(
                f"WARNING: Candidate #{index} "
                f"has non-positive message_id={message_id} - skipped"
            )
            continue

        if not hasattr(candidate, "score"):
            print(
                f"WARNING: Candidate #{index} "
                f"missing score - skipped"
            )
            continue

        try:
            score = float(candidate.score)
        except (TypeError, ValueError):
            print(
                f"WARNING: Candidate #{index} "
                f"has invalid score={candidate.score!r} - skipped"
            )
            continue

        if not hasattr(candidate, "sources"):
            print(
                f"WARNING: Candidate #{index} "
                f"missing sources - skipped"
            )
            continue

        sources = candidate.sources

        if sources is None:
            sources = []
        elif isinstance(sources, list):
            sources = sources.copy()
        elif isinstance(sources, (tuple, set)):
            sources = list(sources)
        else:
            sources = [str(sources)]

        candidates.append(
            {
                "message_id": message_id,
                "score": score,
                "sources": sources,
            }
        )

    return candidates


def print_candidate_validation(candidates):
    print("\nValidated candidates:")

    if not candidates:
        print("  NONE")
        return

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"  #{rank} "
            f"message_id={candidate['message_id']} "
            f"score={candidate['score']:.6f} "
            f"sources={candidate['sources']}"
        )


def main():
    print("=" * 80)
    print("ChatLens-RAG Step 6.1")
    print("Boundary-Aware Context Expansion Test")
    print("=" * 80)

    for query in TEST_QUERIES:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        result = hybrid_search(query)

        raw_candidates = result.get(
            "candidates",
            [],
        )

        print(
            f"\nHybrid candidates returned: "
            f"{len(raw_candidates)}"
        )

        candidates = extract_candidates(
            result,
            top_k=10,
        )

        print(
            f"Validated candidates: "
            f"{len(candidates)}"
        )

        print_candidate_validation(
            candidates
        )

        if not candidates:
            print(
                "\nNo valid candidates available "
                "for context expansion."
            )
            continue

        anchor_ids = [
            candidate["message_id"]
            for candidate in candidates
        ]

        print("\nAnchor IDs:")
        print(anchor_ids)

        regions = expand_candidates(
            candidates
        )

        if not regions:
            print(
                "\nNo context regions found."
            )
            continue

        print(
            f"\nContext regions: "
            f"{len(regions)}"
        )

        print_context_regions(
            regions
        )


if __name__ == "__main__":
    main()