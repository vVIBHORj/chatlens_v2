import re
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from langchain_core.documents import Document

from enrich import EnrichedMessage
from vectorstore import _format_message, _create_chunk_document, build_conversation_chunks
from rag_graph import (
    validate_citations,
    extract_available_message_ids,
    extract_answer_citations,
    extract_message_body,
    check_grounded,
    build_generation_prompt,
    generate,
    CitationValidationResult,
    GraphState,
)


def make_test_state(
    answer: str,
    documents=None,
    scope="specific",
    primary_intent="semantic_rag",
) -> GraphState:
    return {
        "question": "What time is the meeting?",
        "original_question": "What time is the meeting?",
        "documents": documents or [],
        "generation": answer,
        "rewrite_count": 0,
        "grounded": False,
        "semantic_grounded": False,
        "df": None,
        "scope": scope,
        "primary_intent": primary_intent,
        "secondary_intents": [],
        "query_spec": None,
        "citation_validation": None,
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "available_citation_ids": [],
    }


def make_mock_llm(response_text: str):
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=response_text)
    return mock


# =====================================================================
# 1. Valid citation to a retained message
# =====================================================================

def test_citation_case_1_valid_citation_retained_message():
    doc = Document(
        page_content="[ID 10] [2022-09-19 10:00 AM] Alice: The conference starts at 9am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The conference starts at 9am [ID 10]."
    val = validate_citations(answer, [doc])

    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.cited_ids == [10]
    assert val.valid_cited_ids == [10]
    assert val.invalid_cited_ids == []
    assert val.malformed_citations == []
    assert 10 in val.available_ids

    # Integration test with check_grounded
    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is True
    assert result["semantic_grounded"] is True
    assert result["valid_citation_ids"] == [10]
    assert result["invalid_citation_ids"] == []


# =====================================================================
# 2. Invented citation ID
# =====================================================================

def test_citation_case_2_invented_citation_id():
    doc = Document(
        page_content="[ID 10] Alice: The conference starts at 9am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The conference starts at 9am [ID 999]."
    val = validate_citations(answer, [doc])

    assert val.has_citations is True
    assert val.all_cited_valid is False
    assert val.cited_ids == [999]
    assert val.valid_cited_ids == []
    assert val.invalid_cited_ids == [999]

    # Integration: even if semantic grader returns 'yes', final grounded must be False
    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is False
    assert result["semantic_grounded"] is True
    assert result["invalid_citation_ids"] == [999]


# =====================================================================
# 3. Citation to a candidate removed before generation
# =====================================================================

def test_citation_case_3_citation_to_filtered_candidate():
    # Only ID 10 survived relevance grading and reached final context
    doc10 = Document(
        page_content="[ID 10] Alice: The conference starts at 9am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    # Answer cites ID 20, which was removed prior to generation
    answer = "The conference starts at 9am [ID 20]."
    val = validate_citations(answer, [doc10])

    assert val.all_cited_valid is False
    assert val.invalid_cited_ids == [20]
    assert 20 not in val.available_ids

    state = make_test_state(answer=answer, documents=[doc10])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is False
    assert result["invalid_citation_ids"] == [20]


# =====================================================================
# 4. Valid citation to a chronological neighbor present in final context
# =====================================================================

def test_citation_case_4_valid_chronological_neighbor():
    expanded_doc = Document(
        page_content=(
            "[RETRIEVED ANCHOR]\n"
            "[ID 10] [2022-09-19 10:00 AM] Alice: Hey team\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 9] [2022-09-19 09:59 AM] Charlie: Are we ready?\n"
            "[ID 10] [2022-09-19 10:00 AM] Alice: Hey team\n"
            "[ID 12] [2022-09-19 10:02 AM] Bob: Don't forget the presentation notes."
        ),
        metadata={
            "source": "hybrid_search",
            "message_id": 10,
            "context_expanded": True,
            "context_start_message_id": 9,
            "context_end_message_id": 12,
        },
    )
    answer = "Bob reminded everyone to bring the presentation notes [ID 12]."
    val = validate_citations(answer, [expanded_doc])

    assert val.has_citations is True
    assert val.all_cited_valid is True
    assert val.cited_ids == [12]
    assert val.valid_cited_ids == [12]
    assert val.invalid_cited_ids == []
    # Both anchor and neighboring messages are available
    assert set(val.available_ids) == {9, 10, 12}

    state = make_test_state(answer=answer, documents=[expanded_doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is True
    assert 12 in result["available_citation_ids"]


# =====================================================================
# 5. Chunk ID incorrectly used as a message citation
# =====================================================================

def test_citation_case_5_chunk_id_incorrectly_used_as_citation():
    chunk_doc = Document(
        page_content=(
            "[ID 1] [2022-09-19] Alice: Good morning\n"
            "[ID 2] [2022-09-19] Bob: Hello everyone"
        ),
        metadata={
            "source": "conversation_chunk",
            "chunk_id": "chunk_abc123",
            "start_message_id": 1,
            "end_message_id": 2,
        },
    )

    # 5a: Citing non-numeric chunk string token
    answer_chunk_str = "The participants greeted each other [ID chunk_abc123]."
    val_a = validate_citations(answer_chunk_str, [chunk_doc])
    assert val_a.all_cited_valid is False
    assert "[ID chunk_abc123]" in val_a.malformed_citations

    # 5b: Citing raw chunk notation
    answer_chunk_raw = "The participants greeted each other [chunk_abc123]."
    val_b = validate_citations(answer_chunk_raw, [chunk_doc])
    assert val_b.all_cited_valid is False
    assert "[chunk_abc123]" in val_b.malformed_citations

    # 5c: Citing numeric chunk_id that is NOT an individual message ID in the chunk
    chunk_doc_numeric = Document(
        page_content="[ID 1] Alice: Hi\n[ID 2] Bob: Bye",
        metadata={
            "source": "conversation_chunk",
            "chunk_id": 777,
            "start_message_id": 1,
            "end_message_id": 2,
        },
    )
    # Available IDs should ONLY be message IDs {1, 2}, NOT chunk_id 777
    avail = extract_available_message_ids([chunk_doc_numeric])
    assert avail == {1, 2}
    assert 777 not in avail

    answer_numeric_chunk = "They said goodbye [ID 777]."
    val_c = validate_citations(answer_numeric_chunk, [chunk_doc_numeric])
    assert val_c.all_cited_valid is False
    assert val_c.invalid_cited_ids == [777]


# =====================================================================
# 6. Answer with no citations
# =====================================================================

def test_citation_case_6_answer_with_no_citations():
    doc = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The meeting is scheduled for 10am."

    # Specific query: missing_citations should be True
    val_specific = validate_citations(answer, [doc], scope="specific")
    assert val_specific.has_citations is False
    assert val_specific.cited_ids == []
    assert val_specific.missing_citations is True
    assert "No citations provided in answer for specific retrieval query." in val_specific.details

    # Broad query: whole-conversation statistics/profile does not require message citations
    val_broad = validate_citations(answer, [doc], scope="broad")
    assert val_broad.has_citations is False
    assert val_broad.missing_citations is False


# =====================================================================
# 7. Mixed valid and invalid citations
# =====================================================================

def test_citation_case_7_mixed_valid_and_invalid_citations():
    doc = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    answer = "The meeting starts at 10am [ID 10] and lunch is served at noon [ID 50]."
    val = validate_citations(answer, [doc])

    assert val.has_citations is True
    assert val.all_cited_valid is False
    assert val.cited_ids == [10, 50]
    assert val.valid_cited_ids == [10]
    assert val.invalid_cited_ids == [50]

    state = make_test_state(answer=answer, documents=[doc])
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["grounded"] is False
    assert result["semantic_grounded"] is True


# =====================================================================
# 8. Valid ID attached to an unsupported claim (semantic check fails)
# =====================================================================

def test_citation_case_8_valid_id_with_unsupported_claim():
    doc1 = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    doc2 = Document(
        page_content="[ID 11] Bob: I had cereal for breakfast.",
        metadata={"source": "hybrid_search", "message_id": 11},
    )
    # The citation ID 11 exists mechanically, but Bob buying a yacht is completely unsupported
    answer = "The meeting is at 10am [ID 10], and Bob bought a yacht [ID 11]."
    val = validate_citations(answer, [doc1, doc2])

    # Mechanical validation passes!
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [10, 11]
    assert val.invalid_cited_ids == []

    # But semantic grounding check identifies the hallucinated claim and returns 'no'
    state = make_test_state(answer=answer, documents=[doc1, doc2])
    mock_llm = make_mock_llm("no")
    result = check_grounded(state, grounding_llm=mock_llm)
    assert result["semantic_grounded"] is False
    # Final grounded must be False because semantic verification failed
    assert result["grounded"] is False


# =====================================================================
# 9. Missing metadata and malformed citation syntax
# =====================================================================

def test_citation_case_9_missing_metadata_and_malformed_syntax():
    # Documents with None and empty metadata
    doc_no_meta = Document(
        page_content="[ID 10] Alice: Meeting at 10",
        metadata={},
    )
    doc_none_meta = Document(
        page_content="[ID 11] Bob: Understood",
    )
    doc_none_meta.metadata = None

    avail = extract_available_message_ids([doc_no_meta, doc_none_meta])
    assert avail == {10, 11}

    # Answers with various malformed citation attempts
    answer_malformed = "Meeting is at 10 [ID ] or [ID abc] or [ID -5]."
    val = validate_citations(answer_malformed, [doc_no_meta, doc_none_meta])

    assert val.all_cited_valid is False
    assert len(val.malformed_citations) >= 2
    # Does not crash on empty / None answer
    val_empty = validate_citations("", [])
    assert val_empty.has_citations is False
    assert val_empty.all_cited_valid is False


# =====================================================================
# 10. Semantic grounding checker returns true but citation is invalid
# =====================================================================

def test_citation_case_10_semantic_yes_overridden_by_invalid_citation():
    doc = Document(
        page_content="[ID 10] Alice: The meeting is at 10am.",
        metadata={"source": "hybrid_search", "message_id": 10},
    )
    # The factual claim is true, but cited ID 999 is fabricated
    answer = "The meeting is at 10am [ID 999]."
    state = make_test_state(answer=answer, documents=[doc])

    # Semantic LLM evaluates prose claims and returns 'yes'
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)

    # Mechanical validation must override semantic 'yes' to False
    assert result["semantic_grounded"] is True
    assert result["grounded"] is False
    assert result["invalid_citation_ids"] == [999]


# =====================================================================
# 11. Generation prompt receives correct permitted IDs for synthetic evidence
# =====================================================================

def test_generation_prompt_permitted_ids_synthetic_evidence():
    doc0 = Document(
        page_content="[ID 0] Commander Vance: The primary deflector shield operates at 450 GHz.",
        metadata={"source": "hybrid_search", "message_id": 0},
    )
    doc1 = Document(
        page_content="[ID 1] Dr. Thorne: Vessel departure commences at dawn.",
        metadata={"source": "hybrid_search", "message_id": 1},
    )
    doc2 = Document(
        page_content="[ID 2] Elena: Secondary plasma conduits stored in Pod 7.",
        metadata={"source": "hybrid_search", "message_id": 2},
    )

    state = make_test_state(answer="", documents=[doc0, doc1, doc2])
    state["original_question"] = "What frequency does the primary deflector shield operate at?"

    prompt = build_generation_prompt(state, documents=[doc0, doc1, doc2])

    # 1. Permitted IDs must match the evidence IDs
    available_ids = extract_available_message_ids([doc0, doc1, doc2])
    assert available_ids == {0, 1, 2}
    assert "Permitted citation IDs in this context: [0, 1, 2]" in prompt

    # 2. Strict instructions against fabrication must be present
    assert "Never invent, infer, or extrapolate a message ID" in prompt
    assert "state that the information was not found" in prompt

    # 3. generate() invokes LLM with the constructed prompt containing permitted IDs
    mock_llm = make_mock_llm("The shield operates at 450 GHz [ID 0].")
    gen_result = generate(state, generation_llm=mock_llm)

    mock_llm.invoke.assert_called_once()
    called_prompt = mock_llm.invoke.call_args[0][0]
    assert "Permitted citation IDs in this context: [0, 1, 2]" in called_prompt
    assert gen_result["generation"] == "The shield operates at 450 GHz [ID 0]."


# =====================================================================
# 12. Chunk evidence exposes actual embedded message IDs, not chunk_id
# =====================================================================

def test_generation_prompt_chunk_evidence_embedded_ids_not_chunk_id():
    chunk_doc = Document(
        page_content=(
            "[ID 5] Marcus: Cafeteria soup.\n"
            "[ID 6] Elena: Submit maintenance log."
        ),
        metadata={
            "source": "conversation_chunk",
            "chunk_id": 999,
            "start_id": 5,
            "end_id": 6,
        },
    )

    state = make_test_state(answer="", documents=[chunk_doc])
    prompt = build_generation_prompt(state, documents=[chunk_doc])

    available_ids = extract_available_message_ids([chunk_doc])
    assert available_ids == {5, 6}
    assert 999 not in available_ids

    # Permitted IDs must list embedded message IDs 5, 6 and NOT chunk_id 999
    assert "[5, 6]" in prompt
    assert "999" not in prompt


# =====================================================================
# 13. Empty or missing evidence does not cause fabricated permitted IDs
# =====================================================================

def test_generation_prompt_empty_evidence_no_fabricated_ids():
    state_empty = make_test_state(answer="", documents=[])
    state_empty["original_question"] = "What is the passcode?"

    prompt_empty = build_generation_prompt(state_empty, documents=[])

    assert extract_available_message_ids([]) == set()
    assert "insufficient" in prompt_empty.lower()
    # No permitted ID list should be fabricated
    assert "Permitted citation IDs" not in prompt_empty

    # Documents with only whitespace/empty page content
    empty_doc = Document(page_content="   ", metadata={})
    prompt_blank = build_generation_prompt(state_empty, documents=[empty_doc])
    assert "insufficient" in prompt_blank.lower()
    assert "Permitted citation IDs" not in prompt_blank


# =====================================================================
# 14. Existing citation validation continues to reject an invented ID
# =====================================================================

def test_citation_validation_continues_rejecting_invented_id():
    # Context has IDs [0, 1, 2, 3, 4, 5, 6]
    docs = [
        Document(
            page_content=f"[ID {i}] Participant: Message {i}",
            metadata={"source": "hybrid_search", "message_id": i},
        )
        for i in range(7)
    ]
    available_ids = extract_available_message_ids(docs)
    assert available_ids == {0, 1, 2, 3, 4, 5, 6}

    # Answer invents [ID 12]
    answer_with_hallucination = "The primary deflector shield operates at 450 GHz [ID 12]."
    val = validate_citations(answer_with_hallucination, docs)

    assert val.all_cited_valid is False
    assert val.invalid_cited_ids == [12]
    assert 12 not in val.available_ids

    # check_grounded must mark grounded=False even if semantic checker returns 'yes'
    state = make_test_state(answer=answer_with_hallucination, documents=docs)
    mock_llm = make_mock_llm("yes")
    result = check_grounded(state, grounding_llm=mock_llm)

    assert result["grounded"] is False
    assert result["invalid_citation_ids"] == [12]


# =====================================================================
# 15. Generation prompt abstention forbids dummy citations and contains no placeholder examples
# =====================================================================

def test_generation_prompt_abstention_forbids_dummy_citation_and_no_placeholder_examples():
    doc0 = Document(
        page_content="[ID 0] Alice: The weather is sunny today.",
        metadata={"source": "hybrid_search", "message_id": 0},
    )
    state = make_test_state(answer="", documents=[doc0])
    state["original_question"] = "What is the secret passcode?"

    prompt = build_generation_prompt(state, documents=[doc0])

    # Must contain clear abstention instructions
    assert "information was not found" in prompt
    assert "In an abstention, do NOT output any citation brackets" in prompt
    assert "include no citation brackets" in prompt

    # Must NOT contain instructional placeholder numbers that LLMs copy
    assert "[ID 123]" not in prompt
    assert "[ID 12]" not in prompt
    assert "[ID 34]" not in prompt


# =====================================================================
# 16. Semantic claim grounding rejects claim misattributed to another context ID
# =====================================================================

def test_claim_attribution_semantic_grounding_rejects_misattributed_id():
    doc0 = Document(
        page_content="[ID 0] Vance: The shield operates at 450 GHz.",
        metadata={"source": "hybrid_search", "message_id": 0},
    )
    doc1 = Document(
        page_content="[ID 1] Thorne: Vessel departure at dawn.",
        metadata={"source": "hybrid_search", "message_id": 1},
    )

    # Answer asserts the shield frequency but attributes it to [ID 1] (Thorne's departure message)
    misattributed_answer = "[ID 1] The primary deflector shield operates at 450 GHz."

    # 1. Mechanical validation sees that ID 1 is present in context
    val = validate_citations(misattributed_answer, [doc0, doc1])
    assert val.all_cited_valid is True
    assert val.valid_cited_ids == [1]
    assert val.invalid_cited_ids == []

    # 2. Semantic grounding checker evaluates whether message 1 actually supports the claim
    # Simulated semantic grader checking claim against cited ID 1 and finding no support -> 'no'
    state = make_test_state(answer=misattributed_answer, documents=[doc0, doc1])
    mock_semantic_llm = make_mock_llm("no")
    grounded_result = check_grounded(state, grounding_llm=mock_semantic_llm)

    assert grounded_result["semantic_grounded"] is False
    assert grounded_result["grounded"] is False


# =====================================================================
# 17. Generation prompt profile query preserves no-citation policy
# =====================================================================

def test_generation_prompt_profile_query_preserves_no_citation_policy():
    profile_doc = Document(
        page_content="Total messages: 450. Most active participant: Alice (300 messages).",
        metadata={"source": "whole_conversation_profile"},
    )
    state = make_test_state(answer="", documents=[profile_doc], scope="broad")
    state["original_question"] = "Who sends the most messages?"

    prompt = build_generation_prompt(state, documents=[profile_doc])

    assert "statistical profile without individual message IDs" in prompt
    assert "Do not output message ID citations" in prompt
    assert extract_available_message_ids([profile_doc]) == set()


# =====================================================================
# 18. Directly supported factual claim with correct citation verifies grounded
# =====================================================================

def test_groundedness_directly_supported_fact_with_citation():
    doc0 = Document(
        page_content="[ID 0] Commander Vance: The primary deflector shield operates at a frequency of 450 gigahertz.",
        metadata={"source": "hybrid_search", "message_id": 0},
    )
    supported_answer = "The primary deflector shield operates at a frequency of 450 gigahertz, as stated in [ID 0]."
    state = make_test_state(answer=supported_answer, documents=[doc0])

    mock_llm = make_mock_llm("The claim is directly stated in message [ID 0].\nDecision: yes")
    result = check_grounded(state, grounding_llm=mock_llm)

    assert result["grounded"] is True
    assert result["semantic_grounded"] is True
    assert result["valid_citation_ids"] == [0]
    assert result["invalid_citation_ids"] == []


# =====================================================================
# 19. Semantically valid paraphrase with correct citation verifies grounded
# =====================================================================

def test_groundedness_semantic_paraphrase_with_citation():
    doc1 = Document(
        page_content="[ID 1] Dr. Thorne: The vessel's departure from lunar orbit commences at dawn tomorrow.",
        metadata={"source": "hybrid_search", "message_id": 1},
    )
    paraphrased_answer = "The spaceship leaves the moon at dawn tomorrow, as stated in [ID 1]."
    state = make_test_state(answer=paraphrased_answer, documents=[doc1])

    mock_llm = make_mock_llm("The statement is a direct paraphrase of vessel departure from lunar orbit in [ID 1].\nDecision: yes")
    result = check_grounded(state, grounding_llm=mock_llm)

    assert result["grounded"] is True
    assert result["semantic_grounded"] is True
    assert result["valid_citation_ids"] == [1]
    assert result["invalid_citation_ids"] == []


# =====================================================================
# 20. Short contextual reply supported by chronological context verifies grounded
# =====================================================================

def test_groundedness_short_reply_supported_by_chronological_context():
    doc_expanded = Document(
        page_content=(
            "[ID 3] Marcus: Storage Pod 7.\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            "[ID 2] Elena: Where did the engineering team store the secondary plasma conduits?\n"
            "[ID 3] Marcus: Storage Pod 7."
        ),
        metadata={"source": "hybrid_search", "message_id": 3, "context_expanded": True},
    )
    short_reply_answer = "Storage Pod 7 is where the engineering team stored the secondary plasma conduits [ID 3]."
    state = make_test_state(answer=short_reply_answer, documents=[doc_expanded])

    mock_llm = make_mock_llm("Marcus's reply 'Storage Pod 7' directly answers Elena's question in chronological context.\nDecision: yes")
    result = check_grounded(state, grounding_llm=mock_llm)

    assert result["grounded"] is True
    assert result["semantic_grounded"] is True
    assert result["valid_citation_ids"] == [3]


# =====================================================================
# 21. Grader failure preserves downstream citation and grounding checks
# =====================================================================

def test_grader_failure_preserves_citation_and_grounding_checks():
    doc0 = Document(
        page_content="[ID 0] Vance: Shield frequency is 450 GHz.",
        metadata={"source": "hybrid_search", "message_id": 0, "grader_status": "failure"},
    )
    # 1. When cited citation is valid and claim is supported
    valid_answer = "The shield operates at 450 GHz [ID 0]."
    state_valid = make_test_state(answer=valid_answer, documents=[doc0])
    state_valid["grader_status"] = "failed"

    mock_llm_yes = make_mock_llm("Assessment: Claim is directly supported by message 0.\nDecision: yes")
    res_valid = check_grounded(state_valid, grounding_llm=mock_llm_yes)
    assert res_valid["grounded"] is True
    assert res_valid["valid_citation_ids"] == [0]

    # 2. When cited citation is invalid despite grader failure, citation check still forces ungrounded
    invalid_answer = "The shield operates at 450 GHz [ID 99]."
    state_invalid = make_test_state(answer=invalid_answer, documents=[doc0])
    state_invalid["grader_status"] = "failed"

    res_invalid = check_grounded(state_invalid, grounding_llm=mock_llm_yes)
    assert res_invalid["grounded"] is False
    assert res_invalid["invalid_citation_ids"] == [99]


# =====================================================================
# 22. Chunk-contained message ID canonical formatting tests
# =====================================================================

def _make_sample_enriched_message(
    msg_id: int,
    sender: str,
    text: str,
    timestamp_str: str = "2026-10-10 10:00:00",
) -> EnrichedMessage:
    dt = datetime.fromisoformat(timestamp_str)
    return EnrichedMessage(
        id=msg_id,
        timestamp=dt,
        sender=sender,
        message=text,
        message_type="user",
        sentiment_compound=0.0,
        sentiment_label="neutral",
        is_question=text.endswith("?"),
        message_length=len(text),
        response_time_minutes=None,
        reply_time_minutes=None,
    )


def test_chunk_multiple_messages_each_retain_own_canonical_id():
    """
    Multiple messages inside a single conversation chunk must each retain their own
    correct [ID N] canonical header and metadata, without inheriting start/end IDs.
    """
    msgs = [
        _make_sample_enriched_message(10, "Vance", "Primary shield at 450 GHz.", "2026-10-10 10:00:00"),
        _make_sample_enriched_message(11, "Elena", "Acknowledged, telemetry active.", "2026-10-10 10:02:00"),
        _make_sample_enriched_message(12, "Carlos", "Secondary conduits aligned.", "2026-10-10 10:04:00"),
    ]
    chunk_doc = _create_chunk_document(
        messages=msgs,
        episode_id=0,
        chunk_index=0,
        global_chunk_index=0,
    )

    # 1. Page content must contain canonical [ID N] for every message
    assert "[ID 10]" in chunk_doc.page_content
    assert "[ID 11]" in chunk_doc.page_content
    assert "[ID 12]" in chunk_doc.page_content
    assert "[message_id=" not in chunk_doc.page_content

    # 2. Intermediate messages must NOT be overwritten with start_id (10) or end_id (12)
    lines = chunk_doc.page_content.split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("[ID 10]")
    assert lines[1].startswith("[ID 11]")
    assert lines[2].startswith("[ID 12]")
    assert "Elena: Acknowledged, telemetry active." in lines[1]

    # 3. Available IDs must extract all distinct IDs
    avail = extract_available_message_ids([chunk_doc])
    assert avail == {10, 11, 12}


def test_overlapping_chunks_preserve_identical_canonical_ids():
    """
    Overlapping chunks must preserve the identical canonical [ID N] format,
    timestamps, senders, and content for messages that appear in both chunks.
    """
    msgs = [
        _make_sample_enriched_message(i, f"Speaker{i}", f"Statement number {i}.", f"2026-10-10 10:{i:02d}:00")
        for i in range(25)
    ]
    chunks = build_conversation_chunks(msgs)
    assert len(chunks) >= 2, "Expected multiple chunks from 25 messages"

    # Verify every chunk contains only canonical [ID N] headers
    for c in chunks:
        assert "[message_id=" not in c.page_content
        for line in c.page_content.split("\n"):
            assert line.startswith("[ID ")

    # Find messages shared between chunk 0 and chunk 1
    chunk0_ids = extract_available_message_ids([chunks[0]])
    chunk1_ids = extract_available_message_ids([chunks[1]])
    shared_ids = chunk0_ids.intersection(chunk1_ids)
    assert len(shared_ids) > 0, "Expected overlap between consecutive chunks"

    chunk0_lines = {
        int(m.group(1)): m.group(0)
        for m in [re.match(r"^\[ID (\d+)\].*", line) for line in chunks[0].page_content.split("\n")]
        if m
    }
    chunk1_lines = {
        int(m.group(1)): m.group(0)
        for m in [re.match(r"^\[ID (\d+)\].*", line) for line in chunks[1].page_content.split("\n")]
        if m
    }

    for shared_id in shared_ids:
        assert chunk0_lines[shared_id] == chunk1_lines[shared_id], (
            f"Message ID {shared_id} differed across overlapping chunks:\n"
            f"Chunk 0: {chunk0_lines[shared_id]}\n"
            f"Chunk 1: {chunk1_lines[shared_id]}"
        )


def test_message_formatting_does_not_duplicate_ids_or_alter_content():
    """
    Message formatting must not double-prefix IDs or mutate text/timestamps/senders
    when passed through formatting or candidate transformation.
    """
    msg = _make_sample_enriched_message(42, "Dr. Thorne", "Testing orbital mechanics: gamma=1.2.", "2026-10-10 14:00:00")
    formatted = _format_message(msg)

    # 1. Exactly one [ID 42] prefix
    assert formatted.count("[ID 42]") == 1
    assert "[ID 42] [ID 42]" not in formatted
    assert "[message_id=" not in formatted

    # 2. Preserves exact text, sender, and timestamp
    assert "Dr. Thorne: Testing orbital mechanics: gamma=1.2." in formatted

    # 3. Message body extraction returns exact original text
    extracted = extract_message_body(formatted)
    assert extracted == "Testing orbital mechanics: gamma=1.2."

    # 4. If formatted message is passed through candidate stripping regex, no double-prefixing occurs
    clean = re.sub(
        r"^\[(?:message_id|ID)\s*=?\s*\d+\]\s*\[[^\]]+\]\s*[^:]*:\s*",
        "",
        formatted,
        count=1,
    )
    assert clean == "Testing orbital mechanics: gamma=1.2."


def test_generation_evidence_contains_canonical_ids_no_legacy_headers():
    """
    Generation prompt evidence context must present canonical [ID N] headers
    and must not contain legacy [message_id=N] syntax.
    """
    msgs = [
        _make_sample_enriched_message(101, "Vance", "Core frequency locked at 450 GHz.", "2026-10-10 10:00:00"),
        _make_sample_enriched_message(102, "Elena", "Roger that.", "2026-10-10 10:01:00"),
    ]
    chunk_doc = _create_chunk_document(msgs, episode_id=1, chunk_index=0, global_chunk_index=0)
    standalone_doc = Document(
        page_content="[ID 103] [2026-10-10 10:05 AM] Carlos: Backup online.",
        metadata={"source": "hybrid_search", "message_id": 103},
    )

    state = make_test_state(answer="", documents=[chunk_doc, standalone_doc])
    prompt = build_generation_prompt(state, documents=[chunk_doc, standalone_doc])

    assert "[ID 101]" in prompt
    assert "[ID 102]" in prompt
    assert "[ID 103]" in prompt
    assert "[message_id=" not in prompt
    assert "[101, 102, 103]" in prompt

    avail = extract_available_message_ids([chunk_doc, standalone_doc])
    assert avail == {101, 102, 103}


def test_extract_available_message_ids_legacy_backward_compatibility():
    """
    extract_available_message_ids gracefully handles legacy [message_id=N] test fixtures
    without crashing, while strictly excluding non-message IDs like chunk_id.
    """
    legacy_doc = Document(
        page_content=(
            "[message_id=50] Alice: First legacy message.\n"
            "[message_id=51] Bob: Second legacy message."
        ),
        metadata={"source": "conversation_chunk", "chunk_id": 9999},
    )
    avail = extract_available_message_ids([legacy_doc])
    assert avail == {50, 51}
    assert 9999 not in avail
