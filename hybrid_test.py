from hybrid_retriever import (
    hybrid_search,
    print_hybrid_results,
)


TEST_QUERIES = [
    "What laptop was EXODIA considering buying?",
    "Who is udit?",
    "Who was EXODIA's girlfriend?",
    "What did we discuss about sports?",
]


def main():

    for query in TEST_QUERIES:

        result = hybrid_search(query)

        print_hybrid_results(
            result,
            top_k=10,
        )


if __name__ == "__main__":
    main()