"""
test_storage_path_propagation.py

Deterministic offline regression suite for Task 60:
Explicit Database and Vector-Store Path Propagation.

Verifies all 11 criteria from prompt Section 5:
1. ask() stores both explicit paths in GraphState.
2. retrieve() forwards both paths to hybrid_search().
3. hybrid_search() forwards database path to lexical_search().
4. hybrid_search() forwards vector directory to semantic_message_search().
5. hybrid_search() forwards vector directory to semantic_chunk_search().
6. Context expansion (expand_context, assemble_dialogue_evidence_units) uses the same explicit database path.
7. Custom database_path without custom persist_dir raises StorageConfigurationError.
8. Custom persist_dir without custom database_path raises StorageConfigurationError.
9. Calls without custom paths preserve existing default behavior.
10. Existing callers of ask() and hybrid_search() without new args remain compatible.
11. No retrieval branch silently uses global defaults when explicit paths are present.
"""
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch
import pytest

from langchain_core.documents import Document

from hybrid_retriever import (
    HybridCandidate,
    LEXICAL_TOP_K,
    StorageConfigurationError,
    hybrid_search,
    lexical_search,
    semantic_chunk_search,
    semantic_message_search,
    validate_storage_paths,
)
from rag_graph import (
    GraphState,
    ask,
    expand_context,
    keyword_retrieve,
    retrieve,
)
from context_expander import assemble_dialogue_evidence_units


# =============================================================================
# 1. ask() stores both explicit paths in GraphState
# =============================================================================
def test_ask_stores_both_explicit_paths_in_graph_state():
    custom_db = "/custom/path/isolated.db"
    custom_chroma = "/custom/path/chroma_dir"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "generation": "test answer",
        "grounded": True,
        "documents": [],
        "rewrite_count": 0,
        "scope": "test",
        "database_path": custom_db,
        "persist_dir": custom_chroma,
    }

    with patch("rag_graph.build_graph", return_value=mock_graph):
        ask(
            question="What is the deploy time?",
            database_path=custom_db,
            persist_dir=custom_chroma,
        )

        assert mock_graph.invoke.call_count == 1
        initial_state = mock_graph.invoke.call_args[0][0]
        assert initial_state["database_path"] == custom_db
        assert initial_state["persist_dir"] == custom_chroma


# =============================================================================
# 2. retrieve() forwards both paths to hybrid_search()
# =============================================================================
def test_retrieve_forwards_both_paths_to_hybrid_search():
    custom_db = "/custom/path/isolated.db"
    custom_chroma = "/custom/path/chroma_dir"

    state: GraphState = {
        "question": "Where is the secret key?",
        "database_path": custom_db,
        "persist_dir": custom_chroma,
    }

    mock_candidate = HybridCandidate(
        message_id=101,
        sender="Alice",
        message="key in vault",
        timestamp="2026-10-31T20:00:00",
        score=0.95,
        evidence_score=0.95,
        sources=["lexical"],
    )

    mock_result = {
        "candidates": [mock_candidate],
        "chunks": [],
        "rrf_scores": {101: 0.95},
        "query_terms": ["secret", "key"],
        "semantic_query": "secret key",
    }

    with patch("rag_graph.hybrid_search", return_value=mock_result) as mock_hs:
        res = retrieve(state)

        assert mock_hs.call_count == 1
        _, kwargs = mock_hs.call_args
        assert kwargs["database_path"] == custom_db
        assert kwargs["persist_dir"] == custom_chroma
        assert len(res["documents"]) == 1


# =============================================================================
# 3. hybrid_search() forwards database path to lexical_search()
# =============================================================================
def test_hybrid_search_forwards_database_path_to_lexical_search():
    custom_db = "/custom/path/isolated.db"
    custom_chroma = "/custom/path/chroma_dir"

    with patch("hybrid_retriever.semantic_message_search", return_value=[]), \
         patch("hybrid_retriever.semantic_chunk_search", return_value=[]), \
         patch("hybrid_retriever.lexical_search", return_value=[]) as mock_lex:

        hybrid_search(
            query="Alice deployment",
            database_path=custom_db,
            persist_dir=custom_chroma,
        )

        assert mock_lex.call_count == 1
        _, kwargs = mock_lex.call_args
        assert kwargs["database_path"] == custom_db


# =============================================================================
# 4. hybrid_search() forwards vector directory to semantic_message_search()
# =============================================================================
def test_hybrid_search_forwards_vector_directory_to_semantic_message_search():
    custom_db = "/custom/path/isolated.db"
    custom_chroma = "/custom/path/chroma_dir"

    with patch("hybrid_retriever.semantic_message_search", return_value=[]) as mock_sem_msg, \
         patch("hybrid_retriever.semantic_chunk_search", return_value=[]), \
         patch("hybrid_retriever.lexical_search", return_value=[]):

        hybrid_search(
            query="Alice deployment",
            database_path=custom_db,
            persist_dir=custom_chroma,
        )

        assert mock_sem_msg.call_count == 1
        _, kwargs = mock_sem_msg.call_args
        assert kwargs["persist_dir"] == custom_chroma


# =============================================================================
# 5. hybrid_search() forwards vector directory to semantic_chunk_search()
# =============================================================================
def test_hybrid_search_forwards_vector_directory_to_semantic_chunk_search():
    custom_db = "/custom/path/isolated.db"
    custom_chroma = "/custom/path/chroma_dir"

    with patch("hybrid_retriever.semantic_message_search", return_value=[]), \
         patch("hybrid_retriever.semantic_chunk_search", return_value=[]) as mock_sem_chunk, \
         patch("hybrid_retriever.lexical_search", return_value=[]):

        hybrid_search(
            query="Alice deployment",
            database_path=custom_db,
            persist_dir=custom_chroma,
        )

        assert mock_sem_chunk.call_count == 1
        _, kwargs = mock_sem_chunk.call_args
        assert kwargs["persist_dir"] == custom_chroma


# =============================================================================
# 6. Context expansion uses the same explicit database path
# =============================================================================
def test_context_expansion_uses_explicit_database_path():
    custom_db = "/custom/path/isolated.db"

    # 6a. expand_context
    state: GraphState = {
        "primary_intent": "semantic_rag",
        "database_path": custom_db,
        "documents": [
            Document(
                page_content="9pm",
                metadata={"message_id": 42, "source": "semantic_message"},
            )
        ],
    }

    with patch("rag_graph.get_chronological_context", return_value=[
        {"id": 41, "timestamp": "2026-10-31T20:59:00", "sender": "Bob", "message": "When?"},
        {"id": 42, "timestamp": "2026-10-31T21:00:00", "sender": "Alice", "message": "9pm"},
    ]) as mock_chrono:
        expand_context(state)
        assert mock_chrono.call_count == 1
        _, kwargs = mock_chrono.call_args
        assert kwargs.get("database_path") == custom_db

    # 6b. assemble_dialogue_evidence_units
    anchor_doc = Document(
        page_content="9pm",
        metadata={"message_id": 42, "source": "semantic_message"},
    )
    with patch("context_expander.os.path.exists", return_value=True), \
         patch("context_expander.sqlite3.connect") as mock_connect:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # Safe return to complete function
        mock_connect.return_value.cursor.return_value = mock_cursor

        assemble_dialogue_evidence_units([anchor_doc], database_path=custom_db)
        assert mock_connect.call_count == 1
        assert mock_connect.call_args[0][0] == custom_db


# =============================================================================
# 7. Custom database_path without custom persist_dir raises StorageConfigurationError
# =============================================================================
def test_custom_database_path_without_persist_dir_raises_error():
    custom_db = "/custom/path/isolated.db"

    with pytest.raises(StorageConfigurationError) as exc_info:
        validate_storage_paths(database_path=custom_db, persist_dir=None)
    assert "Incomplete custom storage configuration" in str(exc_info.value)

    with pytest.raises(StorageConfigurationError):
        hybrid_search(query="test", database_path=custom_db, persist_dir=None)

    with pytest.raises(StorageConfigurationError):
        ask(question="test", database_path=custom_db, persist_dir=None)

    with pytest.raises(StorageConfigurationError):
        retrieve({"question": "test", "database_path": custom_db, "persist_dir": None})


# =============================================================================
# 8. Custom persist_dir without custom database_path raises StorageConfigurationError
# =============================================================================
def test_custom_persist_dir_without_database_path_raises_error():
    custom_chroma = "/custom/path/chroma_dir"

    with pytest.raises(StorageConfigurationError) as exc_info:
        validate_storage_paths(database_path=None, persist_dir=custom_chroma)
    assert "Incomplete custom storage configuration" in str(exc_info.value)

    with pytest.raises(StorageConfigurationError):
        hybrid_search(query="test", database_path=None, persist_dir=custom_chroma)

    with pytest.raises(StorageConfigurationError):
        ask(question="test", database_path=None, persist_dir=custom_chroma)

    with pytest.raises(StorageConfigurationError):
        retrieve({"question": "test", "database_path": None, "persist_dir": custom_chroma})


# =============================================================================
# 9. Calls without custom paths preserve existing default behavior
# =============================================================================
def test_calls_without_custom_paths_preserve_defaults():
    # Validation passes cleanly when both are None
    validate_storage_paths(database_path=None, persist_dir=None)

    with patch("hybrid_retriever.load_message_vectorstore") as mock_load_msg, \
         patch("hybrid_retriever.load_chunk_vectorstore") as mock_load_chunk, \
         patch("hybrid_retriever.search_messages") as mock_search_msg, \
         patch("rag_graph.search_messages") as mock_rg_search_msg:

        mock_load_msg.return_value.similarity_search_with_score.return_value = []
        mock_load_chunk.return_value.similarity_search_with_score.return_value = []
        mock_search_msg.return_value = []
        mock_rg_search_msg.return_value = []

        # When None is passed to sub-functions, they call loaders without persist_dir/database_path
        semantic_message_search("test", top_k=2)
        assert mock_load_msg.call_args == call()

        semantic_chunk_search("test", top_k=2)
        assert mock_load_chunk.call_args == call()

        lexical_search("test", top_k=2)
        assert mock_search_msg.call_args == call("test", limit=2)

        # Keyword retrieve preserves default search_messages call
        keyword_retrieve("test", limit=5)
        assert mock_rg_search_msg.call_args == call("test", limit=5)


# =============================================================================
# 10. Existing callers of ask() and hybrid_search() without new args remain compatible
# =============================================================================
def test_existing_callers_remain_fully_compatible():
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "generation": "default response",
        "grounded": True,
        "documents": [],
        "rewrite_count": 0,
        "scope": "test",
    }

    with patch("rag_graph.build_graph", return_value=mock_graph):
        # Legacy caller calling ask() with only question
        res = ask("What is the project name?")
        assert res["answer"] == "default response"
        init_state = mock_graph.invoke.call_args[0][0]
        assert init_state["database_path"] is None
        assert init_state["persist_dir"] is None

    with patch("hybrid_retriever.semantic_message_search", return_value=[]), \
         patch("hybrid_retriever.semantic_chunk_search", return_value=[]), \
         patch("hybrid_retriever.lexical_search", return_value=[]):

        # Legacy caller calling hybrid_search() without database_path or persist_dir
        res_hs = hybrid_search("project status")
        assert "candidates" in res_hs
        assert "chunks" in res_hs


# =============================================================================
# 11. No retrieval branch silently uses global defaults when explicit paths are present
# =============================================================================
def test_no_retrieval_branch_silently_uses_global_defaults():
    custom_db = "/isolated/custom.db"
    custom_chroma = "/isolated/custom_chroma"

    with patch("hybrid_retriever.load_message_vectorstore") as mock_load_msg, \
         patch("hybrid_retriever.load_chunk_vectorstore") as mock_load_chunk, \
         patch("hybrid_retriever.search_messages") as mock_search_msg, \
         patch("rag_graph.search_messages") as mock_rg_search_msg:

        mock_load_msg.return_value.similarity_search_with_score.return_value = []
        mock_load_chunk.return_value.similarity_search_with_score.return_value = []
        mock_search_msg.return_value = []
        mock_rg_search_msg.return_value = []

        # 11a. semantic_message_search must forward persist_dir
        semantic_message_search("test query", persist_dir=custom_chroma)
        assert mock_load_msg.call_args == call(persist_dir=custom_chroma)

        # 11b. semantic_chunk_search must forward persist_dir
        semantic_chunk_search("test query", persist_dir=custom_chroma)
        assert mock_load_chunk.call_args == call(persist_dir=custom_chroma)

        # 11c. lexical_search must forward database_path
        lexical_search("test query", database_path=custom_db)
        assert mock_search_msg.call_args == call("test query", limit=LEXICAL_TOP_K, database_path=custom_db)

        # 11d. keyword_retrieve must forward database_path
        keyword_retrieve("test query", database_path=custom_db)
        assert mock_rg_search_msg.call_args == call("test query", limit=10, database_path=custom_db)

        # 11e. hybrid_search forwards both custom paths to all branches
        hybrid_search(
            query="verify isolation",
            database_path=custom_db,
            persist_dir=custom_chroma,
        )
        assert mock_load_msg.call_args == call(persist_dir=custom_chroma)
        assert mock_load_chunk.call_args == call(persist_dir=custom_chroma)
        # Verify search_messages received the explicit custom database path
        assert mock_search_msg.call_args[1].get("database_path") == custom_db
