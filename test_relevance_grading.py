import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag_graph import (
    grade_documents,
    decide_to_generate,
    rewrite_query,
    route_by_scope,
    expand_context,
    validate_citations,
    GraphState,
    MAX_REWRITES,
)


def make_test_state(
    documents=None,
    rewrite_count=0,
    question="When does the conference start?",
    original_question=None,
    scope="specific",
    primary_intent="semantic_rag",
) -> GraphState:
    orig = original_question if original_question is not None else question
    return {
        "question": question,
        "original_question": orig,
        "documents": documents or [],
        "generation": "",
        "rewrite_count": rewrite_count,
        "grounded": False,
        "df": None,
        "scope": scope,
        "primary_intent": primary_intent,
        "secondary_intents": [],
        "query_spec": None,
    }


def make_mock_llm(json_response: str):
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=json_response)
    return mock


# =====================================================================
# 1. Relevant message retained
# =====================================================================

def test_grade_case_1_relevant_message_retained():
    doc = Document(
        page_content="[ID 10] Alice: The conference starts at 9am sharp.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    state = make_test_state(documents=[doc])

    mock_llm = make_mock_llm(
        json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Provides exact start time"}])
    )

    result_state = grade_documents(state, grader_llm=mock_llm)
    assert len(result_state["documents"]) == 1
    retained_doc = result_state["documents"][0]
    assert retained_doc.metadata["message_id"] == 10
    assert retained_doc.metadata["relevance_decision"] == "relevant"


# =====================================================================
# 2. Clearly irrelevant lexical distractor rejected
# =====================================================================

def test_grade_case_2_clearly_irrelevant_distractor_rejected():
    doc = Document(
        page_content="[ID 20] Bob: I love eating conference pears for breakfast.",
        metadata={"source": "hybrid_search", "message_id": 20},
    )
    state = make_test_state(documents=[doc])

    mock_llm = make_mock_llm(
        json.dumps([{"candidate": 1, "decision": "irrelevant", "reason": "Discusses fruit, not the conference"}])
    )

    result_state = grade_documents(state, grader_llm=mock_llm)
    # The confidently irrelevant distractor must be removed
    assert len(result_state["documents"]) == 0


# =====================================================================
# 3. Semantic paraphrase with zero lexical overlap retained
# =====================================================================

def test_grade_case_3_semantic_paraphrase_zero_lexical_overlap_retained():
    query = "What was the departure schedule?"
    doc = Document(
        page_content="[ID 30] Elena: Flight takes off at 08:30 tomorrow morning.",
        metadata={"source": "hybrid_search", "message_id": 30, "matched_terms": []},
    )
    state = make_test_state(question=query, documents=[doc])

    mock_llm = make_mock_llm(
        json.dumps([{"candidate": 1, "decision": "relevant", "reason": "States flight departure time"}])
    )

    result_state = grade_documents(state, grader_llm=mock_llm)
    assert len(result_state["documents"]) == 1
    assert result_state["documents"][0].metadata["message_id"] == 30
    assert result_state["documents"][0].metadata["relevance_decision"] == "relevant"


# =====================================================================
# 4. Short reply and emoji-only candidate not rejected by deterministic rules
# =====================================================================

def test_grade_case_4_short_reply_and_emoji_not_rejected():
    doc_short = Document(
        page_content="[ID 41] Carlos: 9pm",
        metadata={"source": "hybrid_search", "message_id": 41},
    )
    doc_emoji = Document(
        page_content="[ID 42] Carlos: 🏸🔥",
        metadata={"source": "hybrid_search", "message_id": 42},
    )
    state = make_test_state(documents=[doc_short, doc_emoji])

    # Grader marks short reply and emoji as uncertain
    mock_llm = make_mock_llm(
        json.dumps([
            {"candidate": 1, "decision": "uncertain", "reason": "Short time mention without context"},
            {"candidate": 2, "decision": "uncertain", "reason": "Emoji reaction"},
        ])
    )

    result_state = grade_documents(state, grader_llm=mock_llm)
    # Both must be preserved (not dropped by length or emoji content)
    assert len(result_state["documents"]) == 2
    ids = [d.metadata["message_id"] for d in result_state["documents"]]
    assert ids == [41, 42]
    assert result_state["documents"][0].metadata["relevance_decision"] == "uncertain"
    assert result_state["documents"][1].metadata["relevance_decision"] == "uncertain"


# =====================================================================
# 5. Transcript chunk evaluated independently of message score fields
# =====================================================================

def test_grade_case_5_chunk_evaluated_independently():
    # Chunk has no message_id, no hybrid_score, no evidence_score
    chunk_doc = Document(
        page_content="Alice: When is meeting?\nBob: Tomorrow at noon.",
        metadata={
            "source": "conversation_chunk",
            "chunk_id": 501,
            "chunk_rank": 1,
            "semantic_distance": 0.22,
        },
    )
    state = make_test_state(documents=[chunk_doc])

    mock_llm = make_mock_llm(
        json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Transcript discusses meeting schedule"}])
    )

    result_state = grade_documents(state, grader_llm=mock_llm)
    assert len(result_state["documents"]) == 1
    retained_chunk = result_state["documents"][0]
    assert retained_chunk.metadata["chunk_id"] == 501
    assert retained_chunk.metadata["relevance_decision"] == "relevant"


# =====================================================================
# 6. Mixed pool retains relevant and uncertain, removes only irrelevant
# =====================================================================

def test_grade_case_6_mixed_pool_filtering():
    doc_rel = Document(
        page_content="[ID 1] Carlos: The project deadline is Friday at 5pm.",
        metadata={"source": "hybrid_search", "message_id": 1},
    )
    doc_unc = Document(
        page_content="[ID 2] Carlos: Friday maybe?",
        metadata={"source": "hybrid_search", "message_id": 2},
    )
    doc_irr = Document(
        page_content="[ID 3] Bob: I had pizza for dinner.",
        metadata={"source": "hybrid_search", "message_id": 3},
    )
    state = make_test_state(documents=[doc_rel, doc_unc, doc_irr])

    mock_llm = make_mock_llm(
        json.dumps([
            {"candidate": 1, "decision": "relevant", "reason": "Mentions exact deadline"},
            {"candidate": 2, "decision": "uncertain", "reason": "Ambiguous mention of Friday"},
            {"candidate": 3, "decision": "irrelevant", "reason": "Completely off-topic food comment"},
        ])
    )

    result_state = grade_documents(state, grader_llm=mock_llm)
    # Must retain relevant (ID 1) and uncertain (ID 2), while removing irrelevant (ID 3)
    assert len(result_state["documents"]) == 2
    retained_ids = [d.metadata["message_id"] for d in result_state["documents"]]
    assert retained_ids == [1, 2]


# =====================================================================
# 7. Empty retrieval triggers rewriting while retries remain
# =====================================================================

def test_grade_case_7_empty_retrieval_triggers_rewrite():
    state_0 = make_test_state(documents=[], rewrite_count=0)
    assert decide_to_generate(state_0) == "rewrite_query"

    state_1 = make_test_state(documents=[], rewrite_count=1)
    assert decide_to_generate(state_1) == "rewrite_query"


# =====================================================================
# 8. All candidates rejected triggers rewriting while retries remain
# =====================================================================

def test_grade_case_8_all_rejected_triggers_rewrite():
    doc1 = Document(page_content="[ID 1] random talk", metadata={"source": "hybrid_search", "message_id": 1})
    doc2 = Document(page_content="[ID 2] off topic", metadata={"source": "hybrid_search", "message_id": 2})
    state = make_test_state(documents=[doc1, doc2], rewrite_count=0)

    mock_llm = make_mock_llm(
        json.dumps([
            {"candidate": 1, "decision": "irrelevant", "reason": "Off-topic"},
            {"candidate": 2, "decision": "irrelevant", "reason": "Off-topic"},
        ])
    )

    graded_state = grade_documents(state, grader_llm=mock_llm)
    assert len(graded_state["documents"]) == 0

    # With empty filtered documents and rewrites remaining, decide_to_generate must route to rewrite_query
    decision = decide_to_generate(graded_state)
    assert decision == "rewrite_query"


# =====================================================================
# 9. Grader exception, timeout, malformed output fail open
# =====================================================================

def test_grade_case_9_grader_failure_fails_open():
    doc = Document(page_content="[ID 90] Message content", metadata={"source": "hybrid_search", "message_id": 90})

    # 9a. Exception / Timeout
    mock_fail = MagicMock()
    mock_fail.invoke.side_effect = RuntimeError("Ollama connection timeout")
    state_err = make_test_state(documents=[doc])
    res_err = grade_documents(state_err, grader_llm=mock_fail)
    assert len(res_err["documents"]) == 1
    assert res_err["documents"][0].metadata["message_id"] == 90
    assert "grader_exception" in res_err["documents"][0].metadata["relevance_reason"]

    # 9b. Malformed non-JSON output
    mock_malformed = make_mock_llm("I cannot output JSON, here is an explanation...")
    state_malformed = make_test_state(documents=[doc])
    res_malformed = grade_documents(state_malformed, grader_llm=mock_malformed)
    assert len(res_malformed["documents"]) == 1
    assert res_malformed["documents"][0].metadata["relevance_decision"] == "uncertain"

    # 9c. Unknown / unsupported decision keyword
    mock_unknown = make_mock_llm(
        json.dumps([{"candidate": 1, "decision": "partially_maybe", "reason": "unknown token"}])
    )
    state_unk = make_test_state(documents=[doc])
    res_unk = grade_documents(state_unk, grader_llm=mock_unknown)
    assert len(res_unk["documents"]) == 1
    assert res_unk["documents"][0].metadata["relevance_decision"] == "uncertain"


# =====================================================================
# 10. Maximum rewrite count terminates retries
# =====================================================================

def test_grade_case_10_max_rewrites_terminates_to_generate():
    # Documents is empty, but rewrite_count has reached MAX_REWRITES
    state_max = make_test_state(documents=[], rewrite_count=MAX_REWRITES)
    assert decide_to_generate(state_max) == "generate"


# =====================================================================
# 11. Broad scope bypasses grading
# =====================================================================

def test_grade_case_11_broad_scope_bypasses_grading():
    state_broad = make_test_state(
        scope="broad",
        primary_intent="statistics",
    )
    # route_by_scope must return 'profile', bypassing retrieve and grade_documents
    route = route_by_scope(state_broad)
    assert route == "profile"


# =====================================================================
# 12. Original question remains unchanged after query rewriting
# =====================================================================

@patch("rag_graph.llm")
def test_grade_case_12_original_question_remains_unchanged(mock_llm):
    mock_llm.invoke.return_value = MagicMock(content="Rewritten conversational query")
    original = "What did we discuss about sports?"
    state = make_test_state(question=original, original_question=original, rewrite_count=0)

    rewritten_state = rewrite_query(state)
    assert rewritten_state["question"] == "Rewritten conversational query"
    # original_question must be strictly preserved
    assert rewritten_state["original_question"] == original
    assert rewritten_state["rewrite_count"] == 1


# =====================================================================
# 13. Relevant lexical evidence during grader failure is not discarded
# =====================================================================

def test_fallback_case_1_relevant_lexical_candidate_preserved_on_grader_failure():
    """
    A relevant candidate with strong lexical evidence must not be discarded
    merely because the grader is unavailable (exception/timeout).
    It is retained as uncertain with explicit grader_status='failure'.
    """
    doc = Document(
        page_content="[ID 101] We deployed database version 16 to the server yesterday.",
        metadata={
            "source": "hybrid_search",
            "message_id": 101,
            "lexical_rank": 1,
            "semantic_rank": 1,
            "matched_terms": ["database", "version", "deployed"],
            "evidence_score": 0.90,
        },
    )
    state = make_test_state(
        question="What database version was deployed?",
        documents=[doc],
    )

    mock_fail = MagicMock()
    mock_fail.invoke.side_effect = RuntimeError("Ollama connection timeout")

    result = grade_documents(state, grader_llm=mock_fail)

    # Candidate must be preserved, not discarded
    assert len(result["documents"]) == 1
    retained = result["documents"][0]
    assert retained.metadata["message_id"] == 101
    assert retained.metadata["relevance_decision"] == "uncertain"
    # Grader failure status must be recorded
    assert retained.metadata.get("grader_status") == "failure"
    assert retained.metadata.get("grader_failure") is True
    assert result.get("grader_status") == "failed"


# =====================================================================
# 14. Relevant semantic paraphrase preserved during grader failure
# =====================================================================

def test_fallback_case_2_semantic_paraphrase_preserved_on_grader_failure():
    """
    A candidate answering the question via semantic paraphrase (low/zero lexical
    overlap) must NOT be rejected when the grader fails.
    """
    doc = Document(
        page_content="[ID 103] Flight takes off at 08:30 tomorrow morning.",
        metadata={
            "source": "hybrid_search",
            "message_id": 103,
            "semantic_rank": 1,
            "lexical_rank": None,
            "matched_terms": [],
            "evidence_score": 0.085,
        },
    )
    state = make_test_state(
        question="What was the departure schedule?",
        documents=[doc],
    )

    mock_fail = MagicMock()
    mock_fail.invoke.side_effect = TimeoutError("Grading LLM timed out")

    result = grade_documents(state, grader_llm=mock_fail)

    # Paraphrase candidate with 0 lexical overlap must be preserved
    assert len(result["documents"]) == 1
    retained = result["documents"][0]
    assert retained.metadata["message_id"] == 103
    assert retained.metadata["relevance_decision"] == "uncertain"
    assert retained.metadata.get("grader_status") == "failure"
    assert retained.metadata.get("grader_failure") is True


# =====================================================================
# 15. Obvious lexical distractor not promoted to relevant during failure
# =====================================================================

def test_fallback_case_3_lexical_distractor_not_promoted_to_relevant_on_grader_failure():
    """
    An unrelated candidate with superficial word overlap must NOT automatically
    be considered relevant during grader failure. It remains 'uncertain'.
    """
    doc = Document(
        page_content="[ID 102] I deployed a new version of the frontend UI, not the database.",
        metadata={
            "source": "hybrid_search",
            "message_id": 102,
            "matched_terms": ["database", "version", "deployed"],
            "lexical_rank": 2,
            "evidence_score": 0.80,
        },
    )
    state = make_test_state(
        question="What database version was deployed?",
        documents=[doc],
    )

    mock_fail = MagicMock()
    mock_fail.invoke.side_effect = RuntimeError("Ollama failure")

    result = grade_documents(state, grader_llm=mock_fail)

    assert len(result["documents"]) == 1
    retained = result["documents"][0]
    # Crucial: superficial lexical overlap must NOT cause the distractor to be marked 'relevant'
    assert retained.metadata["relevance_decision"] != "relevant"
    assert retained.metadata["relevance_decision"] == "uncertain"
    assert retained.metadata.get("grader_status") == "failure"


# =====================================================================
# 16. Malformed grader output preserves conservative behavior safely
# =====================================================================

def test_fallback_case_4_malformed_grader_output_safe_conservative_behavior():
    """
    Malformed non-JSON output fails open safely to uncertain and records
    grader parsing fallback without crashing.
    """
    doc = Document(
        page_content="[ID 105] Some message text",
        metadata={"source": "hybrid_search", "message_id": 105},
    )
    state = make_test_state(documents=[doc])

    mock_malformed = make_mock_llm("<<< INVALID NON-JSON OUTPUT >>>")
    result = grade_documents(state, grader_llm=mock_malformed)

    assert len(result["documents"]) == 1
    retained = result["documents"][0]
    assert retained.metadata["relevance_decision"] == "uncertain"
    assert retained.metadata.get("grader_status") == "failure"
    assert retained.metadata.get("grader_failure") is True
    assert "parsing_fallback" in retained.metadata.get("relevance_reason", "")
    assert result.get("grader_status") == "failed"


# =====================================================================
# 17. Mixed candidate pool predictable behavior under grader failure
# =====================================================================

def test_fallback_case_5_mixed_candidate_pool_partial_and_total_grader_failure():
    """
    In a mixed pool (relevant, uncertain, distractor):
    - When all batches fail, all candidates are conservatively preserved as uncertain.
    - When one batch succeeds and another fails, state reflects 'degraded' status.
    """
    doc_rel = Document(
        page_content="[ID 1] The project deadline is Friday at 5pm.",
        metadata={"source": "hybrid_search", "message_id": 1},
    )
    doc_irr = Document(
        page_content="[ID 2] I had pizza for dinner.",
        metadata={"source": "hybrid_search", "message_id": 2},
    )
    state = make_test_state(documents=[doc_rel, doc_irr])

    # 17a: Total grader failure -> all retained as uncertain
    mock_fail = MagicMock()
    mock_fail.invoke.side_effect = RuntimeError("Service unavailable")
    res_fail = grade_documents(state, grader_llm=mock_fail)
    assert len(res_fail["documents"]) == 2
    assert res_fail["documents"][0].metadata.get("grader_status") == "failure"
    assert res_fail["documents"][1].metadata.get("grader_status") == "failure"
    assert res_fail.get("grader_status") == "failed"

    # 17b: Partial failure simulation across multiple batches
    doc_batch1 = Document(
        page_content="[ID 3] Direct answer to query",
        metadata={"source": "hybrid_search", "message_id": 3},
    )
    doc_batch2 = Document(
        page_content="[ID 4] Second candidate in separate batch",
        metadata={"source": "hybrid_search", "message_id": 4},
    )
    state_multi = make_test_state(documents=[doc_batch1, doc_batch2])

    # First invoke succeeds (marks ID 3 relevant), second invoke raises exception
    mock_partial = MagicMock()
    mock_partial.invoke.side_effect = [
        MagicMock(content=json.dumps([{"candidate": 1, "decision": "relevant", "reason": "Exact match"}])),
        RuntimeError("Timeout on second batch"),
    ]

    with patch("rag_graph.GRADE_BATCH_SIZE", 1):
        res_partial = grade_documents(state_multi, grader_llm=mock_partial)
        assert len(res_partial["documents"]) == 2
        # Doc 1 graded successfully
        assert res_partial["documents"][0].metadata["relevance_decision"] == "relevant"
        assert res_partial["documents"][0].metadata.get("grader_status") == "success"
        assert res_partial["documents"][0].metadata.get("grader_failure") is False
        # Doc 2 failed open
        assert res_partial["documents"][1].metadata["relevance_decision"] == "uncertain"
        assert res_partial["documents"][1].metadata.get("grader_status") == "failure"
        assert res_partial["documents"][1].metadata.get("grader_failure") is True
        # Overall status is degraded
        assert res_partial.get("grader_status") == "degraded"


# =====================================================================
# 18. Message and chunk candidates without metadata assumptions
# =====================================================================

def test_fallback_case_6_chunk_and_message_candidates_without_metadata_assumptions():
    """
    Handling grader failure must not make message-only metadata assumptions
    on conversation chunk candidates (e.g. absent hybrid_score/evidence_score).
    """
    msg_doc = Document(
        page_content="[ID 201] Carlos: Alpha protocol active",
        metadata={
            "source": "hybrid_search",
            "message_id": 201,
            "hybrid_score": 0.85,
            "evidence_score": 0.90,
            "matched_terms": ["alpha", "protocol"],
        },
    )
    chunk_doc = Document(
        page_content="Carlos: Good morning\nDiana: Protocol is active",
        metadata={
            "source": "conversation_chunk",
            "chunk_id": "chunk_99",
            "chunk_rank": 1,
            "semantic_distance": 0.18,
            # Notably lacks: message_id, hybrid_score, evidence_score, matched_terms
        },
    )
    state = make_test_state(documents=[msg_doc, chunk_doc])

    mock_fail = MagicMock()
    mock_fail.invoke.side_effect = TimeoutError("Grading timeout")

    # Must process both safely without KeyError or attribute errors
    result = grade_documents(state, grader_llm=mock_fail)
    assert len(result["documents"]) == 2
    res_msg, res_chunk = result["documents"]

    assert res_msg.metadata["message_id"] == 201
    assert res_msg.metadata.get("grader_status") == "failure"

    assert res_chunk.metadata["chunk_id"] == "chunk_99"
    assert res_chunk.metadata.get("grader_status") == "failure"
    assert res_chunk.metadata.get("hybrid_score") is None


# =====================================================================
# 19. Citation integrity removes unretained candidates from available IDs
# =====================================================================

def test_fallback_case_7_citation_integrity_removes_unretained_candidates_from_available_ids():
    """
    Any candidate removed by relevance grading must NOT become a valid source
    in citation validation, even if its ID appears in earlier state.
    """
    doc_keep = Document(
        page_content="[ID 10] Alice: The conference starts at 9am sharp.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    doc_remove = Document(
        page_content="[ID 20] Bob: I had pizza for dinner.",
        metadata={"source": "hybrid_search", "message_id": 20},
    )
    state = make_test_state(documents=[doc_keep, doc_remove])

    # Grader successfully rejects ID 20 as irrelevant
    mock_llm = make_mock_llm(
        json.dumps([
            {"candidate": 1, "decision": "relevant", "reason": "Conference time"},
            {"candidate": 2, "decision": "irrelevant", "reason": "Food comment"},
        ])
    )
    graded_state = grade_documents(state, grader_llm=mock_llm)
    assert len(graded_state["documents"]) == 1

    # Pass remaining documents through expand_context
    expanded_state = expand_context(graded_state)

    # An answer citing both ID 10 and the rejected ID 20
    answer = "The conference starts at 9am [ID 10, ID 20]."
    val = validate_citations(answer, expanded_state["documents"])

    assert val.all_cited_valid is False
    assert val.invalid_cited_ids == [20]
    assert 20 not in val.available_ids
    assert 10 in val.available_ids


# =====================================================================
# 20. Failure-state handling keeps graph usable and reflects limits
# =====================================================================

def test_fallback_case_8_failure_state_handling_graph_usable_and_reflects_limits():
    """
    When the grader completely fails, the graph remains fully usable:
    - Retained candidates carry grader_status='failure'
    - decide_to_generate routes to 'generate' without deadlocking
    - Downstream expand_context processes the candidates safely
    """
    doc = Document(
        page_content="[ID 50] System online",
        metadata={"source": "hybrid_search", "message_id": 50},
    )
    state = make_test_state(documents=[doc])

    mock_fail = MagicMock()
    mock_fail.invoke.side_effect = ConnectionError("Ollama daemon unreachable")

    # 1. grade_documents handles failure safely
    graded_state = grade_documents(state, grader_llm=mock_fail)
    assert graded_state.get("grader_status") == "failed"
    assert len(graded_state["documents"]) == 1

    # 2. decide_to_generate proceeds to generate (graph remains usable)
    next_step = decide_to_generate(graded_state)
    assert next_step == "generate"

    # 3. expand_context processes documents safely
    expanded_state = expand_context(graded_state)
    assert len(expanded_state["documents"]) == 1
    assert expanded_state["documents"][0].metadata.get("grader_status") == "failure"

