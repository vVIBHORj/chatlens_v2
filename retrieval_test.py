"""
retrieval_test.py

Diagnostic tool for ChatLens-RAG.

IMPORTANT:
    This script does NOT use the LLM.

It tests three retrieval mechanisms:

    1. Individual-message semantic retrieval
    2. Conversation-chunk semantic retrieval
    3. SQLite lexical retrieval

The goal is to determine whether the information is being retrieved
before we modify the RAG generation pipeline.
"""

from typing import List
from langchain_core.documents import Document

from vectorstore import (
    load_message_vectorstore,
    load_chunk_vectorstore,
)
from chat_database import (
    search_messages,
    extract_lexical_terms,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MESSAGE_TOP_K = 10
CHUNK_TOP_K = 5
KEYWORD_TOP_K = 10


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def print_separator(char="-", width=90):
    print(char * width)


def print_message_result(
    rank: int,
    document: Document,
):
    """
    Display an individual-message semantic result.
    """

    metadata = document.metadata

    message_id = metadata.get(
        "message_id",
        "?",
    )

    sender = metadata.get(
        "sender",
        "?",
    )

    timestamp = metadata.get(
        "timestamp",
        "?",
    )

    print(
        f"\n[{rank}] "
        f"message_id={message_id}"
    )

    print(
        f"    Sender:    {sender}"
    )

    print(
        f"    Timestamp: {timestamp}"
    )

    print(
        f"    Text:      {document.page_content}"
    )


def print_chunk_result(
    rank: int,
    document: Document,
):
    """
    Display a conversation-level semantic result.
    """

    metadata = document.metadata

    episode_id = metadata.get(
        "episode_id",
        "?",
    )

    chunk_index = metadata.get(
        "chunk_index",
        "?",
    )

    start_id = metadata.get(
        "start_message_id",
        "?",
    )

    end_id = metadata.get(
        "end_message_id",
        "?",
    )

    date = metadata.get(
        "date",
        "?",
    )

    num_messages = metadata.get(
        "num_messages",
        "?",
    )

    print(
        f"\n[{rank}] "
        f"episode={episode_id} "
        f"chunk={chunk_index}"
    )

    print(
        f"    Date:     {date}"
    )

    print(
        f"    IDs:      {start_id} → {end_id}"
    )

    print(
        f"    Messages: {num_messages}"
    )

    print(
        "\n    Transcript:"
    )

    print(
        indent_text(
            document.page_content,
            "        ",
        )
    )


def print_keyword_result(
    rank: int,
    row,
):
    """
    Display a SQLite lexical result.

    Works with either sqlite3.Row or dictionary-like rows.
    """

    try:
        message_id = row["id"]
    except Exception:
        message_id = "?"

    try:
        sender = row["sender"]
    except Exception:
        sender = "?"

    try:
        timestamp = row["timestamp"]
    except Exception:
        timestamp = "?"

    try:
        message = row["message"]
    except Exception:
        message = "?"

    print(
        f"\n[{rank}] message_id={message_id}"
    )

    print(
        f"    Sender:    {sender}"
    )

    print(
        f"    Timestamp: {timestamp}"
    )

    print(
        f"    Text:      {message}"
    )


def indent_text(
    text: str,
    prefix: str,
) -> str:
    """
    Indent multiline text.
    """

    return "\n".join(
        prefix + line
        for line in text.splitlines()
    )


# ============================================================================
# MESSAGE SEMANTIC SEARCH
# ============================================================================

def semantic_message_search(
    query: str,
) -> List[Document]:
    """
    Search individual-message vectors.
    """

    store = load_message_vectorstore()

    return store.similarity_search(
        query,
        k=MESSAGE_TOP_K,
    )


# ============================================================================
# CHUNK SEMANTIC SEARCH
# ============================================================================

def semantic_chunk_search(
    query: str,
) -> List[Document]:
    """
    Search conversation-level vectors.
    """

    store = load_chunk_vectorstore()

    return store.similarity_search(
        query,
        k=CHUNK_TOP_K,
    )


# ============================================================================
# SQLITE KEYWORD SEARCH
# ============================================================================

def keyword_search(
    query: str,
):
    """
    Search SQLite FTS after lexical query preprocessing.
    """

    return search_messages(
        query,
        limit=KEYWORD_TOP_K,
    )


# ============================================================================
# RUN ONE QUERY
# ============================================================================

def run_query(
    query: str,
):
    """
    Run all three retrieval methods.
    """

    print("\n")
    print("=" * 90)
    print("QUERY")
    print("=" * 90)

    print(f"\n{query}")

    # ------------------------------------------------------------------
    # Individual messages
    # ------------------------------------------------------------------

    print("\n")
    print_separator()
    print(
        f"1. INDIVIDUAL MESSAGE SEMANTIC SEARCH "
        f"(top {MESSAGE_TOP_K})"
    )
    print_separator()

    try:

        message_results = semantic_message_search(
            query
        )

        print(
            f"\nRetrieved {len(message_results)} results."
        )

        for rank, document in enumerate(
            message_results,
            start=1,
        ):
            print_message_result(
                rank,
                document,
            )

    except Exception as e:

        print(
            "\n❌ Individual message search failed:"
        )

        print(e)

    # ------------------------------------------------------------------
    # Conversation chunks
    # ------------------------------------------------------------------

    print("\n")
    print_separator()
    print(
        f"2. CONVERSATION CHUNK SEMANTIC SEARCH "
        f"(top {CHUNK_TOP_K})"
    )
    print_separator()

    try:

        chunk_results = semantic_chunk_search(
            query
        )

        print(
            f"\nRetrieved {len(chunk_results)} results."
        )

        for rank, document in enumerate(
            chunk_results,
            start=1,
        ):
            print_chunk_result(
                rank,
                document,
            )

    except Exception as e:

        print(
            "\n❌ Conversation chunk search failed:"
        )

        print(e)

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    print("\n")
    print_separator()
    print(
        f"3. SQLITE LEXICAL SEARCH "
        f"(top {KEYWORD_TOP_K})"
    )
    print_separator()

    lexical_terms = extract_lexical_terms(query)

    print(
        "\nLexical terms:"
    )

    print(
        f"    {lexical_terms}"
    )

    try:

        keyword_results = keyword_search(
            query
        )

        print(
            f"\nRetrieved {len(keyword_results)} results."
        )

        for rank, row in enumerate(
            keyword_results,
            start=1,
        ):
            print_keyword_result(
                rank,
                row,
            )

    except Exception as e:

        print(
            "\n❌ SQLite search failed:"
        )

        print(e)

    print("\n")
    print("=" * 90)
    print("END QUERY")
    print("=" * 90)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("=" * 90)
    print("ChatLens-RAG Retrieval Diagnostic")
    print("=" * 90)

    print(
        "\nIMPORTANT:"
        "\nThis test does NOT use qwen2.5:3b."
        "\nNo grading."
        "\nNo rewriting."
        "\nNo generation."
        "\n"
    )

    test_queries = [
        "What laptop was EXODIA considering buying?",
        "Who is udit?",
        "Who was EXODIA's girlfriend?",
        "What did we discuss about sports?",
    ]

    for query in test_queries:

        run_query(query)

        print("\n")

    print("=" * 90)
    print("ALL RETRIEVAL TESTS COMPLETE")
    print("=" * 90)