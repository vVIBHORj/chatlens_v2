import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from rag_graph import (
    grade_documents,
    decide_to_generate,
    rewrite_query,
    route_by_scope,
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
