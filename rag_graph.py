
"""
rag_graph.py

ChatLens-RAG retrieval and reasoning pipeline.

Architecture:

    User Question
          |
          v
    classify_scope
       /       \
   broad       specific
     |             |
     v             v
 whole-chat     hybrid retrieval
 profile          |
                  v
              grade docs
                  |
          +-------+-------+
          |               |
       relevant        irrelevant
          |               |
          v               v
    expand context      rewrite
          |               |
          v               |
       generate <---------+
          |
          v
    grounded check
          |
          v
         END


Retrieval architecture:

    Chroma
      |
      | semantic similarity
      v
    candidate chunks
      |
      +-------------------+
      |                   |
      v                   v
   semantic           SQLite FTS5
                       keyword
      |                   |
      +---------+---------+
                |
                v
         candidate pool
                |
                v
         LLM relevance grade
                |
                v
       SQLite context expansion
                |
                v
             Qwen


Important:

    Chroma = semantic locator
    SQLite = source of truth
    SQLite FTS5 = exact/keyword retrieval
    LLM = reasoning + grading + groundedness


Requirements:

    ollama pull qwen2.5:3b

    Chroma vector store must already exist.

    chat_data.db must already exist.
"""


# =====================================================================
# Imports
# =====================================================================

import json
import re
from dataclasses import dataclass
from typing import List, Optional, TypedDict, Dict, Tuple, Any, Set
from datetime import datetime

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from hybrid_retriever import hybrid_search
from analytics import build_style_profile


from chat_database import (
    search_messages,
    get_chronological_context,
    get_message_range,
)

from query_router import (
    route_query,
    QuerySpec,
    build_query_spec,
)


# =====================================================================
# Configuration
# =====================================================================

MAX_REWRITES = 2

SEMANTIC_TOP_K = 10

KEYWORD_TOP_K = 10

MAX_HYBRID_CANDIDATES = 20

GRADE_BATCH_SIZE = 10

CONTEXT_BEFORE = 10

CONTEXT_AFTER = 10


# =====================================================================
# LLM
# =====================================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)


# =====================================================================
# Graph state
# =====================================================================
class GraphState(TypedDict, total=False):
    question: str
    original_question: str

    documents: List[Document]

    generation: str
    rewrite_count: int
    grounded: bool
    semantic_grounded: bool

    df: Optional[pd.DataFrame]

    scope: str

    primary_intent: str
    secondary_intents: List[str]

    query_spec: Optional[QuerySpec]

    citation_validation: Optional[Dict[str, Any]]
    valid_citation_ids: Optional[List[int]]
    invalid_citation_ids: Optional[List[int]]
    available_citation_ids: Optional[List[int]]

# =====================================================================
# Scope classification
# =====================================================================
# =====================================================================
# Query routing
# =====================================================================

def route_query_node(
    state: GraphState,
) -> GraphState:

    spec = build_query_spec(
        state["question"]
    )

    print(
        f"[QUERY SPEC] "
        f"intent={spec.intent.value} "
        f"target={spec.target!r} "
        f"entities={spec.entities} "
        f"answer_type={spec.answer_type} "
        f"constraints={spec.constraints}"
    )

    return {
        **state,
        "query_spec": spec,
        "primary_intent": spec.intent.value,
        "secondary_intents": [
            intent.value
            for intent in route_query(
                state["question"]
            ).secondary_intents
        ],
    }

def classify_scope(
    state: GraphState,
) -> GraphState:
    """
    Determine whether the query should use whole-conversation
    analysis or retrieval-based analysis.

    The deterministic query router is the primary signal.
    """

    primary_intent = state["primary_intent"]

    # Whole-conversation analytical intents
    broad_intents = {
        "statistics",
        "time_analysis",
        "behavior_analysis",
        "sentiment_analysis",
        "comparison",
        "anomaly_detection",
    }

    if primary_intent in broad_intents:
        scope = "broad"
    else:
        scope = "specific"

    print(
        f"[SCOPE] Primary intent={primary_intent} "
        f"-> Scope={scope}"
    )

    return {
        **state,
        "scope": scope,
    }


# =====================================================================
# Whole-conversation profile
# =====================================================================

def build_profile_context(
    state: GraphState,
) -> GraphState:
    """
    Build a synthetic document containing whole-conversation
    statistics for broad questions.
    """

    df = state.get("df")

    if df is None or df.empty:

        return {
            **state,
            "documents": [],
        }

    profile_text = build_style_profile(
        df
    )

    document = Document(
        page_content=profile_text,

        metadata={
            "source": "whole_conversation_profile",
        },
    )

    return {
        **state,
        "documents": [document],
    }


# =====================================================================
# Expand one semantic document
# =====================================================================

def expand_document_context(
    document: Document,
    before: int = CONTEXT_BEFORE,
    after: int = CONTEXT_AFTER,
) -> Document:
    """
    Expand a Chroma result with surrounding original messages.

    Chroma provides the relevant conversation section.

    SQLite provides the original messages around that section.
    """

    metadata = document.metadata

    start_message_id = metadata.get(
        "start_message_id"
    )

    end_message_id = metadata.get(
        "end_message_id"
    )

    # -------------------------------------------------------------
    # Not a semantic chunk
    # -------------------------------------------------------------

    if (
        start_message_id is None
        or end_message_id is None
    ):

        return document

    start_message_id = int(
        start_message_id
    )

    end_message_id = int(
        end_message_id
    )

    expanded_start = max(
        0,
        start_message_id - before,
    )

    expanded_end = (
        end_message_id + after
    )

    rows = get_message_range(
        expanded_start,
        expanded_end,
    )

    if not rows:

        return document

    # -------------------------------------------------------------
    # Format transcript
    # -------------------------------------------------------------

    lines = []

    for row in rows:

        timestamp = row["timestamp"]

        try:

            dt = datetime.fromisoformat(
                timestamp
            )

            timestamp_text = dt.strftime(
                "%d/%m/%Y %I:%M %p"
            )

        except Exception:

            timestamp_text = (
                timestamp or ""
            )

        sender = row["sender"] or ""

        message = row["message"] or ""

        lines.append(
            f"[ID {row['id']}] "
            f"[{timestamp_text}] "
            f"{sender}: "
            f"{message}"
        )

    expanded_transcript = "\n".join(
        lines
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    new_metadata = dict(
        metadata
    )

    new_metadata[
        "context_expanded"
    ] = True

    new_metadata[
        "expanded_start_message_id"
    ] = expanded_start

    new_metadata[
        "expanded_end_message_id"
    ] = expanded_end

    new_metadata[
        "context_messages_before"
    ] = before

    new_metadata[
        "context_messages_after"
    ] = after

    return Document(
        page_content=expanded_transcript,
        metadata=new_metadata,
    )


# =====================================================================
# Keyword retrieval
# =====================================================================

def keyword_retrieve(
    question: str,
    limit: int = KEYWORD_TOP_K,
) -> List[Document]:
    """
    Retrieve exact/keyword matches from SQLite FTS5.

    This is complementary to semantic retrieval.

    Useful for:

        names
        exact words
        unusual spellings
        slang
        Hinglish
        exact phrases
        dates
        short expressions
    """

    rows = search_messages(
        question,
        limit=limit,
    )

    documents = []

    for row in rows:

        timestamp = row["timestamp"]

        sender = row["sender"] or ""

        message = row["message"] or ""

        content = (
            f"[ID {row['id']}] "
            f"[{timestamp}] "
            f"{sender}: "
            f"{message}"
        )

        documents.append(
            Document(
                page_content=content,

                metadata={
                    "source": "keyword_search",

                    "message_id": row["id"],

                    "timestamp": timestamp,

                    "sender": sender,

                    "fts_rank": row["fts_rank"],
                },
            )
        )

    return documents


# =====================================================================
# Hybrid retrieval
# =====================================================================
def retrieve(
    state: GraphState,
) -> GraphState:
    """
    Retrieve evidence using the deterministic hybrid retriever.

    The hybrid retriever combines:
        - semantic message retrieval
        - lexical SQLite retrieval
        - query expansion
        - evidence scoring
        - conversation chunk retrieval

    The LLM is not involved in retrieval.
    """

    question = state["question"]

    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID RETRIEVAL"
    )

    print(
        "=" * 70
    )

    print(
        f"Question: {question}"
    )

    result = hybrid_search(
        question,
        message_top_k=20,
        lexical_top_k=20,
        chunk_top_k=5,
    )

    candidates = result["candidates"]
    chunks = result["chunks"]

    print(
        f"Message candidates: {len(candidates)}"
    )

    print(
        f"Conversation chunks: {len(chunks)}"
    )

    print(
        "\nTop hybrid candidates:"
    )

    for rank, candidate in enumerate(
        candidates[:10],
        start=1,
    ):
        print(
            f"  #{rank} "
            f"ID={candidate.message_id} "
            f"score={candidate.score:.4f} "
            f"evidence={candidate.evidence_score:.4f} "
            f"sources={candidate.sources}"
        )

        if candidate.message:
            print(
                f"      {candidate.sender}: "
                f"{candidate.message}"
            )

    documents = []

    # -------------------------------------------------------------
    # Convert ranked message candidates into Documents
    # -------------------------------------------------------------

    for candidate in candidates[:MAX_HYBRID_CANDIDATES]:

        clean_message = candidate.message or ""
        while True:
            stripped = re.sub(
                r"^\[(?:message_id|ID)\s*=?\s*\d+\]\s*\[[^\]]+\]\s*[^:]*:\s*",
                "",
                clean_message,
                count=1,
            )
            if stripped == clean_message:
                break
            clean_message = stripped

        content = (
            f"[ID {candidate.message_id}] "
            f"[{candidate.timestamp}] "
            f"{candidate.sender or ''}: "
            f"{clean_message}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": "hybrid_search",
                    "message_id": candidate.message_id,
                    "timestamp": candidate.timestamp,
                    "sender": candidate.sender,
                    "hybrid_score": candidate.score,
                    "evidence_score": candidate.evidence_score,
                    "semantic_rank": candidate.semantic_rank,
                    "lexical_rank": candidate.lexical_rank,
                    "semantic_distance": candidate.semantic_distance,
                    "matched_terms": candidate.matched_terms,
                    "sources": candidate.sources,
                },
            )
        )

    # -------------------------------------------------------------
    # Add conversation chunks as supporting evidence
    # -------------------------------------------------------------

    for chunk in chunks:

        documents.append(
            Document(
                page_content=chunk["document"],
                metadata={
                    "source": "conversation_chunk",
                    "start_message_id":
                        chunk["start_message_id"],
                    "end_message_id":
                        chunk["end_message_id"],
                    "episode_id":
                        chunk["episode_id"],
                    "chunk_id":
                        chunk["chunk_id"],
                    "participants":
                        chunk["participants"],
                    "chunk_rank":
                        chunk["rank"],
                    "semantic_distance":
                        chunk["distance"],
                },
            )
        )

    print(
        f"\nTotal retrieval documents: "
        f"{len(documents)}"
    )

    return {
        **state,
        "documents": documents,
    }

# =====================================================================
# Relevance Grading Helpers
# =====================================================================

GRADE_PROMPT_TEMPLATE = (
    "You are a conservative relevance grader for a WhatsApp chat archive retrieval system.\n"
    "Assess whether each retrieved candidate contains information relevant to answering the user's question, "
    "or could plausibly serve as helpful conversational context.\n\n"
    "User Question:\n"
    "{question}\n\n"
    "Retrieved Candidates:\n"
    "{candidates_text}\n\n"
    "Evaluation Guidelines:\n"
    "1. Evaluate each candidate individually.\n"
    "2. Candidates may be single messages or conversation transcript chunks.\n"
    "3. Conversational chats are informal: short replies (e.g., 'yes', '9pm', 'sure'), typos, emojis, "
    "slang, and indirect statements may be relevant context.\n"
    "4. Do NOT reject a candidate solely because of zero lexical keyword overlap, short length, or emojis.\n"
    "5. If a candidate is clearly and unambiguously about a completely unrelated topic, mark 'irrelevant'.\n"
    "6. If a candidate is helpful or directly answers the question, mark 'relevant'.\n"
    "7. If you are uncertain or the context is ambiguous, mark 'uncertain'.\n"
    "8. Do not invent probabilities or numeric scores.\n\n"
    "Output MUST be a valid JSON array of objects with keys:\n"
    '- "candidate": integer candidate number (1, 2, ...)\n'
    '- "decision": exactly "relevant", "irrelevant", or "uncertain"\n'
    '- "reason": brief explanation (1 sentence)\n\n'
    "JSON Response:\n"
)


def format_candidate_for_grader(idx: int, document: Document) -> str:
    """Format a candidate document for the grader prompt preserving evidence type."""
    source = document.metadata.get("source", "unknown")
    if source == "conversation_chunk":
        chunk_id = document.metadata.get("chunk_id", "unknown")
        header = f"[Candidate {idx}] (Conversation Transcript Chunk ID: {chunk_id})"
    elif source == "hybrid_search":
        msg_id = document.metadata.get("message_id", "unknown")
        header = f"[Candidate {idx}] (Single Message ID: {msg_id})"
    else:
        header = f"[Candidate {idx}] (Evidence Source: {source})"

    return f"{header}\n{document.page_content.strip()}"


def parse_grader_response(
    raw_text: str,
    candidate_indices: List[int],
) -> Dict[int, Tuple[str, str]]:
    """
    Parse and validate structured grader JSON output.
    Fails open by returning ('uncertain', ...) for missing, malformed, or invalid entries.
    """
    results: Dict[int, Tuple[str, str]] = {
        idx: ("uncertain", "parsing_fallback") for idx in candidate_indices
    }

    if not raw_text or not raw_text.strip():
        return results

    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed_list = None
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed_list = json.loads(text[start : end + 1])
        except Exception:
            parsed_list = None
    else:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            try:
                single_obj = json.loads(text[obj_start : obj_end + 1])
                if isinstance(single_obj, dict):
                    parsed_list = [single_obj]
            except Exception:
                parsed_list = None

    if isinstance(parsed_list, list):
        for item in parsed_list:
            if not isinstance(item, dict):
                continue
            cand_raw = item.get("candidate")
            try:
                cand_idx = int(cand_raw)
            except (ValueError, TypeError):
                cand_idx = None

            decision_raw = str(item.get("decision", "")).strip().lower()
            if decision_raw in ("relevant", "irrelevant", "uncertain"):
                decision = decision_raw
            else:
                decision = "uncertain"

            reason = str(item.get("reason", "")).strip() or "No reason provided"

            if cand_idx in results:
                results[cand_idx] = (decision, reason)
    else:
        # Regex fallback for malformed JSON containing candidate objects
        pattern = re.compile(
            r'\{[^{}]*?"candidate"\s*:\s*(\d+)[^{}]*?"decision"\s*:\s*"([a-zA-Z_]+)"[^{}]*\}',
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            try:
                cand_idx = int(match.group(1))
                dec_val = match.group(2).strip().lower()
                decision = dec_val if dec_val in ("relevant", "irrelevant", "uncertain") else "uncertain"
                if cand_idx in results:
                    results[cand_idx] = (decision, "regex_extracted")
            except Exception:
                pass

    return results


def grade_candidate_batch(
    question: str,
    batch_docs: List[Document],
    grader_llm: Optional[Any] = None,
) -> List[Tuple[Document, str, str]]:
    """
    Grade a bounded batch of candidates using LLM in a single structured prompt.
    Returns list of tuples: (document, decision, reason).
    """
    if not batch_docs:
        return []

    active_llm = grader_llm if grader_llm is not None else llm
    indices = list(range(1, len(batch_docs) + 1))
    formatted_chunks = [
        format_candidate_for_grader(idx, doc)
        for idx, doc in zip(indices, batch_docs)
    ]
    candidates_text = "\n\n---\n\n".join(formatted_chunks)
    prompt = GRADE_PROMPT_TEMPLATE.format(
        question=question,
        candidates_text=candidates_text,
    )

    try:
        response = active_llm.invoke(prompt)
        raw_output = response.content if hasattr(response, "content") else str(response)
        parsed_decisions = parse_grader_response(raw_output, indices)
    except Exception as e:
        parsed_decisions = {
            idx: ("uncertain", f"grader_exception: {e}")
            for idx in indices
        }

    graded_batch = []
    for idx, doc in zip(indices, batch_docs):
        decision, reason = parsed_decisions.get(idx, ("uncertain", "missing_index"))
        graded_batch.append((doc, decision, reason))

    return graded_batch


# =====================================================================
# Grade retrieved documents
# =====================================================================

def grade_documents(
    state: GraphState,
    grader_llm: Optional[Any] = None,
) -> GraphState:
    """
    Conservatively grade retrieved documents using bounded LLM batch evaluation.

    Rules:
    - Retains relevant and uncertain candidates.
    - Removes only candidates confidently determined to be irrelevant.
    - Fails open on parsing errors, timeouts, or LLM exceptions.
    - Preserves message candidates and transcript chunks as distinct evidence types.
    """
    documents = state.get("documents", [])
    if not documents:
        print("\nNo retrieved candidates to grade.")
        return {
            **state,
            "documents": [],
        }

    question = state.get("original_question") or state.get("question", "")
    print(
        f"\nConservative relevance grading: evaluating {len(documents)} "
        f"candidates in batches of up to {GRADE_BATCH_SIZE}..."
    )

    kept_documents: List[Document] = []
    rejected_count = 0

    # Process in bounded batches
    for batch_start in range(0, len(documents), GRADE_BATCH_SIZE):
        batch = documents[batch_start : batch_start + GRADE_BATCH_SIZE]
        graded_results = grade_candidate_batch(question, batch, grader_llm=grader_llm)

        for doc, decision, reason in graded_results:
            new_metadata = dict(doc.metadata)
            new_metadata["relevance_decision"] = decision
            new_metadata["relevance_reason"] = reason

            updated_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata,
            )

            if decision == "irrelevant":
                rejected_count += 1
                source = new_metadata.get("source", "unknown")
                item_id = new_metadata.get("message_id") or new_metadata.get("chunk_id") or "?"
                print(
                    f"  [REJECTED] {source} ID={item_id}: {reason}"
                )
            else:
                kept_documents.append(updated_doc)
                source = new_metadata.get("source", "unknown")
                item_id = new_metadata.get("message_id") or new_metadata.get("chunk_id") or "?"
                print(
                    f"  [{decision.upper()}] {source} ID={item_id}: {reason}"
                )

    print(
        f"Relevance grading summary: kept {len(kept_documents)}/"
        f"{len(documents)} candidates ({rejected_count} rejected as irrelevant)."
    )

    return {
        **state,
        "documents": kept_documents,
    }


# =====================================================================
# Decide whether to generate or rewrite
# =====================================================================

def decide_to_generate(
    state: GraphState,
) -> str:
    """
    Decide what happens after document grading.

    Routing rules:
    - If rewrite limit has been reached, proceed to generation (handles
      insufficient context gracefully).
    - If at least one relevant or uncertain candidate remains, proceed
      to context expansion and generation.
    - If all candidates were rejected as irrelevant (or retrieval was empty)
      and rewrites remain, route to query rewriting.
    """
    documents = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count >= MAX_REWRITES:
        return "generate"

    if documents:
        return "generate"

    return "rewrite_query"


# =====================================================================
# Rewrite query
# =====================================================================

def rewrite_query(
    state: GraphState,
) -> GraphState:
    """
    Rewrite a failed retrieval query into language more likely to
    match the WhatsApp archive.
    """

    original_question = (
        state.get("original_question") or state["question"]
    )

    prompt = (
        "Rewrite the following question so that it is more likely "
        "to retrieve useful messages from a WhatsApp chat archive.\n\n"

        "Preserve the user's original intent.\n"

        "Use natural conversational terms that people might actually "
        "use in WhatsApp messages.\n\n"

        f"Original question:\n"
        f"{original_question}\n\n"

        "Return only the rewritten question."
    )

    new_question = llm.invoke(
        prompt
    ).content.strip()

    if not new_question:

        new_question = original_question

    print(
        f"\nQuery rewrite "
        f"{state['rewrite_count'] + 1}:"
    )

    print(
        f"  {new_question}"
    )

    return {
        **state,

        "question": new_question,

        "rewrite_count": (
            state["rewrite_count"] + 1
        ),
    }


# =====================================================================
# Expand relevant documents
# =====================================================================

def expand_context(
    state: GraphState,
) -> GraphState:
    """
    Expand retrieved message anchors using chronological neighbors
    when appropriate.

    Topic-synthesis semantic queries should preserve the retrieved
    anchors without expanding every anchor, because broad topic
    questions benefit from evidence across multiple retrieved points
    rather than large amounts of surrounding conversational noise.
    """

    expanded_documents = []

    primary_intent = state.get("primary_intent")

    # -------------------------------------------------------------
    # Semantic topic questions
    # -------------------------------------------------------------

    if primary_intent == "semantic_rag":

        for document in state["documents"][:20]:

            new_metadata = dict(
                document.metadata
            )

            new_metadata["context_expanded"] = False
            new_metadata["context_method"] = "retrieved_anchor"

            expanded_documents.append(
                Document(
                    page_content=(
                        "[RETRIEVED ANCHOR]\n"
                        f"{document.page_content}"
                    ),
                    metadata=new_metadata,
                )
            )

        print(
            f"\nContext expansion skipped for semantic topic query: "
            f"{len(expanded_documents)} retrieved documents preserved."
        )

        return {
            **state,
            "documents": expanded_documents,
        }

    # -------------------------------------------------------------
    # Other query types
    # -------------------------------------------------------------

    for document in state["documents"][:10]:

        message_id = document.metadata.get(
            "message_id"
        )

        # ---------------------------------------------------------
        # Conversation chunk
        # ---------------------------------------------------------

        if document.metadata.get("source") == "conversation_chunk":

            start_id = document.metadata.get(
                "start_message_id"
            )

            end_id = document.metadata.get(
                "end_message_id"
            )

            if start_id is None or end_id is None:
                expanded_documents.append(document)
                continue

            rows = get_chronological_context(
                int(start_id),
                before=CONTEXT_BEFORE,
                after=CONTEXT_AFTER,
            )

        # ---------------------------------------------------------
        # Hybrid message result
        # ---------------------------------------------------------

        elif message_id is not None:

            rows = get_chronological_context(
                int(message_id),
                before=CONTEXT_BEFORE,
                after=CONTEXT_AFTER,
            )

        else:

            expanded_documents.append(document)
            continue

        if not rows:
            expanded_documents.append(document)
            continue

        # ---------------------------------------------------------
        # Format chronological transcript
        # ---------------------------------------------------------

        lines = []

        for row in rows:

            timestamp = row["timestamp"]

            try:
                dt = datetime.fromisoformat(
                    timestamp
                )

                timestamp_text = dt.strftime(
                    "%d/%m/%Y %I:%M %p"
                )

            except Exception:

                timestamp_text = (
                    timestamp or ""
                )

            sender = row["sender"] or ""
            message = row["message"] or ""

            lines.append(
                f"[ID {row['id']}] "
                f"[{timestamp_text}] "
                f"{sender}: "
                f"{message}"
            )

        anchor_text = document.page_content

        expanded_transcript = (
            "[RETRIEVED ANCHOR]\n"
            f"{anchor_text}\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            + "\n".join(lines)
        )

        # ---------------------------------------------------------
        # Preserve retrieval metadata
        # ---------------------------------------------------------

        new_metadata = dict(
            document.metadata
        )

        new_metadata[
            "context_expanded"
        ] = True

        new_metadata[
            "context_messages_before"
        ] = CONTEXT_BEFORE

        new_metadata[
            "context_messages_after"
        ] = CONTEXT_AFTER

        new_metadata[
            "context_method"
        ] = "chronological"

        new_metadata[
            "context_start_message_id"
        ] = rows[0]["id"]

        new_metadata[
            "context_end_message_id"
        ] = rows[-1]["id"]

        expanded_documents.append(
            Document(
                page_content=expanded_transcript,
                metadata=new_metadata,
            )
        )

    print(
        f"\nContext expansion: "
        f"{len(expanded_documents)} "
        f"documents expanded chronologically."
    )

    return {
        **state,
        "documents": expanded_documents,
    }


# =====================================================================
# Generate answer
# =====================================================================

def generate(
    state: GraphState,
) -> GraphState:
    """
    Generate the final answer using the original question and the
    retrieved/expanded context.
    """

    documents = state["documents"]

    context = "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )

    print("\n" + "=" * 70)
    print("CONTEXT SENT TO LLM")
    print("=" * 70)
    print(context)
    print("=" * 70)

    is_profile = any(
        document.metadata.get("source")
        == "whole_conversation_profile"
        for document in documents
    )

    # -------------------------------------------------------------
    # Context description
    # -------------------------------------------------------------

    if is_profile:

        context_note = (
            "The context below is a whole-conversation statistical "
            "profile covering the participants and the conversation "
            "rather than a small retrieved excerpt."
        )

    else:

        context_note = (
            "The context below contains conversation messages retrieved "
            "because they may be relevant to the user's question. Some "
            "messages are exact retrieved anchors and some are nearby "
            "chronological context. Nearby messages may be unrelated."
        )

    # -------------------------------------------------------------
    # No context
    # -------------------------------------------------------------

    if not context.strip():

        prompt = (
            "You are answering a question about a WhatsApp conversation.\n\n"

            "There is no reliable retrieved context available.\n\n"

            f"Question:\n"
            f"{state['original_question']}\n\n"

            "Do not invent an answer. Clearly say that the "
            "available chat context is insufficient."
        )

    else:

        prompt = (
            "You are an evidence-grounded AI assistant answering a user's question "
            "about an archived WhatsApp conversation. You are summarizing the archive, "
            "NOT continuing or participating in the chat.\n\n"

            f"{context_note}\n\n"

            "YOUR TASK: Provide a concise, natural-language summary in the assistant's "
            "voice (typically 1–3 sentences) that directly answers the question using ONLY "
            "explicit evidence from the context messages. Synthesize only relevant messages "
            "whose relevance is directly supported. Never output a raw transcript line or message dump "
            "as the entire answer, and never prefix the answer with a list of IDs.\n\n"

            "STRICT GROUNDING RULES:\n\n"

            "1. Use ONLY information explicitly present in the context. "
            "Do NOT add any knowledge or assumptions from outside the context.\n\n"

            "2. Distinguish message types carefully:\n"
            "   - STATEMENT: Direct assertion of fact\n"
            "   - QUESTION: An inquiry or asking about something\n"
            "   - REQUEST / SUGGESTION: Asking or suggesting an action\n"
            "   - LOCATION / VENUE: Referencing a place or location\n"
            "   - POSSIBILITY: Speculation or something conditional\n"
            "   Only direct STATEMENTS can support factual claims. Questions, requests, "
            "locations, and possibilities must never be reported as confirmed events, activities, "
            "or completed actions.\n\n"

            "3. Never convert a question, request, or suggestion into evidence that the event or "
            "activity actually happened. Do not claim someone played an activity merely because "
            "a message mentions a court, ball, venue, or playing.\n\n"

            "4. Never interpret a location containing a topic word as proof that the topic itself was discussed "
            "or that an associated activity occurred. Mentioning a venue or location (e.g., a ground, court, store, "
            "or office) only proves that the location was named, not that an associated event, activity, or plan took place.\n\n"

            "5. Treat cryptic, contradictory, slang-heavy, or ambiguous messages conservatively. Preserve their literal "
            "phrasing and ambiguity rather than inventing an interpretation. If a message is an unexplained equation, "
            "shorthand, or unclear slang, report it neutrally without asserting an unverified meaning.\n\n"

            "6. Avoid broad, overreaching introductory claims such as 'they discussed various topics including X, Y, Z' "
            "unless each item was genuinely discussed. Do not feel obliged to mention every retrieved message; "
            "summarize only the specific points directly supported and relevant to the question.\n\n"

            "7. Use INLINE citations placed immediately after the specific claim they support (e.g., 'An inquiry was made "
            "about X [ID 12], and Y was requested [ID 34]'). Cite ONLY the smallest relevant set of message IDs directly "
            "supporting each claim. NEVER list or clump message IDs at the beginning or end of the answer.\n\n"

            "8. If the retrieved evidence supports only a limited conclusion (such as inquiries or mentions rather than "
            "actual activities or plans), state that limitation concisely instead of filling gaps with assumptions.\n\n"

            "9. Never infer participants, plans, intentions, relationships, or future actions unless the context explicitly "
            "states them.\n\n"

            "CONTEXT:\n"
            f"{context}\n\n"

            f"USER QUESTION:\n"
            f"{state['original_question']}\n\n"

            "Remember: Write a natural 1–3 sentence assistant summary with INLINE [ID 123] citations placed immediately "
            "after each claim. Never list IDs at the beginning. Summarize only supported, relevant points.\n\n"

            "ANSWER:"
        )

    try:

        answer = llm.invoke(
            prompt
        ).content.strip()

    except Exception as e:

        answer = (
            "I was unable to generate the answer "
            f"because the local language model returned "
            f"an error: {e}"
        )

    return {
        **state,
        "generation": answer,
    }


# =====================================================================
# Citation Integrity Validation
# =====================================================================

@dataclass
class CitationValidationResult:
    has_citations: bool
    all_cited_valid: bool
    cited_ids: List[int]
    valid_cited_ids: List[int]
    invalid_cited_ids: List[int]
    available_ids: List[int]
    malformed_citations: List[str]
    missing_citations: bool
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_citations": self.has_citations,
            "all_cited_valid": self.all_cited_valid,
            "cited_ids": list(self.cited_ids),
            "valid_cited_ids": list(self.valid_cited_ids),
            "invalid_cited_ids": list(self.invalid_cited_ids),
            "available_ids": list(self.available_ids),
            "malformed_citations": list(self.malformed_citations),
            "missing_citations": self.missing_citations,
            "details": self.details,
        }


def extract_available_message_ids(documents: List[Document]) -> Set[int]:
    """
    Extract all message IDs actually present in the evidence documents supplied to generation.

    Derives IDs from the formatted text of individual messages and chronological context lines,
    cross-checked with individual message metadata.
    Does NOT infer message IDs from metadata ranges or chunk IDs.
    """
    available_ids: Set[int] = set()
    if not documents:
        return available_ids

    # Regex patterns for message prefixes in candidate documents:
    # 1. "[ID 123]" used in hybrid search and chronological context
    # 2. "[message_id=123]" used in conversation chunks
    id_pattern = re.compile(r"\[(?:ID|message_id)\s*=?\s*(\d+)\]", re.IGNORECASE)

    for doc in documents:
        # Extract from actual document text
        if doc and hasattr(doc, "page_content") and doc.page_content:
            for match in id_pattern.finditer(doc.page_content):
                try:
                    available_ids.add(int(match.group(1)))
                except (ValueError, TypeError):
                    pass

        # Metadata cross-check for single message candidates only
        # Strictly exclude conversation chunks to avoid treating chunk_id as message_id
        if doc and hasattr(doc, "metadata") and doc.metadata:
            if doc.metadata.get("source") != "conversation_chunk":
                msg_id = doc.metadata.get("message_id")
                if msg_id is not None:
                    try:
                        available_ids.add(int(msg_id))
                    except (ValueError, TypeError):
                        pass

    return available_ids


def extract_answer_citations(answer: str) -> Tuple[List[int], List[str]]:
    """
    Extract citation IDs and identify malformed citations from the generated answer.
    Returns:
        (cited_ids, malformed_citations)
    """
    if not answer or not isinstance(answer, str):
        return [], []

    cited_ids: List[int] = []
    malformed_citations: List[str] = []

    # 1. Match all brackets that start with ID or id: e.g. [ID ...]
    id_bracket_pattern = re.compile(r"\[ID\s*([^\]]*)\]", re.IGNORECASE)
    for match in id_bracket_pattern.finditer(answer):
        raw_inside = match.group(1).strip()
        full_bracket = match.group(0)

        if not raw_inside:
            malformed_citations.append(full_bracket)
            continue

        # Split by comma or whitespace, ignoring optional "ID" repetition (e.g. [ID 10, ID 12])
        tokens = [t.strip() for t in re.split(r"[\s,]+", raw_inside) if t.strip()]
        has_valid_num = False
        is_malformed = False

        for token in tokens:
            if token.upper() == "ID":
                continue
            if token.isdigit():
                val = int(token)
                if val not in cited_ids:
                    cited_ids.append(val)
                has_valid_num = True
            else:
                is_malformed = True

        if is_malformed or not has_valid_num:
            malformed_citations.append(full_bracket)

    # 2. Check for other non-message citation formats (e.g. [chunk_123], [message_id=45])
    non_id_pattern = re.compile(r"\[(?:chunk|episode|message_id)[^\]]*\]", re.IGNORECASE)
    for match in non_id_pattern.finditer(answer):
        full_token = match.group(0)
        if full_token not in malformed_citations:
            malformed_citations.append(full_token)

    return cited_ids, malformed_citations


def validate_citations(
    answer: str,
    documents: List[Document],
    scope: str = "specific",
) -> CitationValidationResult:
    """
    Deterministically validate citation integrity against evidence documents.

    - Extracts supported message IDs from actual evidence text and metadata.
    - Extracts [ID <integer>] citations from the generated answer.
    - Identifies any cited IDs absent from the active context.
    - Distinguishes message IDs from chunk IDs.
    - Handles missing metadata and malformed syntax without crashing.
    """
    if not isinstance(documents, list):
        documents = []

    available_ids_set = extract_available_message_ids(documents)
    available_ids = sorted(list(available_ids_set))

    cited_ids, malformed_citations = extract_answer_citations(answer)

    has_citations = len(cited_ids) > 0 or len(malformed_citations) > 0

    valid_cited_ids = [cid for cid in cited_ids if cid in available_ids_set]
    invalid_cited_ids = [cid for cid in cited_ids if cid not in available_ids_set]

    # For specific retrieval queries, citations are expected; report missing citations
    missing_citations = (scope != "broad" and not has_citations and len(available_ids) > 0)

    all_cited_valid = (
        has_citations
        and len(invalid_cited_ids) == 0
        and len(malformed_citations) == 0
    )

    if not has_citations:
        if missing_citations:
            details = "No citations provided in answer for specific retrieval query."
        else:
            details = "No citations provided."
    elif len(invalid_cited_ids) > 0 or len(malformed_citations) > 0:
        details = (
            f"Citation validation failed: {len(invalid_cited_ids)} invalid IDs "
            f"({invalid_cited_ids}), {len(malformed_citations)} malformed citations ({malformed_citations})."
        )
    else:
        details = f"All {len(valid_cited_ids)} citations are valid ({valid_cited_ids})."

    return CitationValidationResult(
        has_citations=has_citations,
        all_cited_valid=all_cited_valid,
        cited_ids=cited_ids,
        valid_cited_ids=valid_cited_ids,
        invalid_cited_ids=invalid_cited_ids,
        available_ids=available_ids,
        malformed_citations=malformed_citations,
        missing_citations=missing_citations,
        details=details,
    )


# =====================================================================
# Groundedness check
# =====================================================================

def check_grounded(
    state: GraphState,
    grounding_llm: Optional[Any] = None,
) -> GraphState:
    """
    Check whether the generated answer is supported by the retrieved
    context, integrating deterministic mechanical citation validation
    with semantic claim support.
    """

    documents = state.get("documents", [])
    answer = state.get("generation", "")
    scope = state.get("scope", "specific")

    # 1. Deterministic mechanical citation validation
    citation_res = validate_citations(
        answer=answer,
        documents=documents,
        scope=scope,
    )

    print("\n" + "=" * 70)
    print("CITATION INTEGRITY VALIDATION")
    print("=" * 70)
    print(f"Status: {citation_res.details}")
    print(f"Cited IDs: {citation_res.cited_ids}")
    print(f"Valid cited IDs: {citation_res.valid_cited_ids}")
    print(f"Invalid cited IDs: {citation_res.invalid_cited_ids}")
    print(f"Available message IDs in context: {citation_res.available_ids}")
    if citation_res.malformed_citations:
        print(f"Malformed citations: {citation_res.malformed_citations}")
    print("=" * 70)

    if not documents:
        return {
            **state,
            "grounded": False,
            "semantic_grounded": False,
            "citation_validation": citation_res.to_dict(),
            "valid_citation_ids": citation_res.valid_cited_ids,
            "invalid_citation_ids": citation_res.invalid_cited_ids,
            "available_citation_ids": citation_res.available_ids,
        }

    context = "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = (
        "You are a strict factual groundedness checker for a WhatsApp "
        "conversation analysis system.\n\n"

        "The ANSWER below is an AI assistant's factual summary of the archived "
        "conversation (not a reply to a WhatsApp message). Your job is to determine "
        "whether EVERY substantive claim in the ANSWER is directly supported by the "
        "supplied CONTEXT.\n\n"

        "Rules:\n"
        "1. Check the answer claim-by-claim, not just whether the general "
        "topic appears in the context.\n\n"

        "2. A claim is supported only if the context explicitly states it "
        "or it is an extremely direct paraphrase of what the context says.\n\n"

        "3. Do NOT treat semantic similarity as factual evidence.\n\n"

        "4. Do NOT infer actions, intentions, plans, relationships, "
        "participants, or events that are not explicitly stated.\n\n"

        "5. A message mentioning a sport does not prove that someone "
        "played that sport.\n\n"

        "6. A message mentioning a person does not prove that the person "
        "participated in the event being discussed.\n\n"

        "7. A message about a possibility or question does not prove that "
        "the event actually happened.\n\n"

        "8. If even ONE substantive claim in the answer is unsupported, "
        "return 'no'.\n\n"

        "9. Ignore minor wording differences, grammar mistakes, and "
        "harmless paraphrasing when checking factual support.\n\n"

        "Examples:\n"
        "- Context: 'Tennis ni hai?' -> 'They played tennis' = no.\n"
        "- Context: 'Football ground mein tha' -> 'They played football' = no.\n"
        "- Context: 'Tum konsi sports loge?' -> 'Sports were discussed' = yes.\n"
        "- Context: 'Tennis ball leke aaiyo' -> 'They discussed tennis' = yes.\n"
        "- Context: 'Khelo to bata dena' -> 'They played together' = no.\n\n"

        "Answer with exactly one word:\n"
        "'yes' = EVERY substantive claim in the answer is supported\n"
        "'no' = AT LEAST ONE substantive claim is unsupported\n\n"

        f"CONTEXT:\n"
        f"{context}\n\n"

        f"ANSWER:\n"
        f"{answer}\n\n"

        "Decision:"
    )

    active_llm = grounding_llm if grounding_llm is not None else llm

    try:
        response = active_llm.invoke(
            prompt
        ).content.strip().lower()

        print(
            f"[GROUNDEDNESS CHECK] Raw response: {response!r}"
        )

        semantic_grounded = response.startswith(
            "yes"
        )

    except Exception:
        semantic_grounded = False

    # 3. Final grounded calculation
    has_invalid_citations = (
        len(citation_res.invalid_cited_ids) > 0
        or len(citation_res.malformed_citations) > 0
    )

    if has_invalid_citations:
        if semantic_grounded:
            print(
                "[GROUNDEDNESS CHECK] Overriding semantic 'yes' to ungrounded "
                f"due to invalid or malformed citations: {citation_res.invalid_cited_ids} {citation_res.malformed_citations}"
            )
        final_grounded = False
    else:
        final_grounded = semantic_grounded

    return {
        **state,
        "grounded": final_grounded,
        "semantic_grounded": semantic_grounded,
        "citation_validation": citation_res.to_dict(),
        "valid_citation_ids": citation_res.valid_cited_ids,
        "invalid_citation_ids": citation_res.invalid_cited_ids,
        "available_citation_ids": citation_res.available_ids,
    }


# =====================================================================
# Scope routing
# =====================================================================

def route_by_scope(
    state: GraphState,
) -> str:

    if state["scope"] == "broad":

        return "profile"

    return "retrieve"


# =====================================================================
# Build graph
# =====================================================================

def build_graph():

    workflow = StateGraph(
        GraphState
    )

    # -------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------

    workflow.add_node(
        "route_query",
        route_query_node,
    )

    workflow.add_node(
        "classify_scope",
        classify_scope,
    )

    workflow.add_node(
        "build_profile_context",
        build_profile_context,
    )

    workflow.add_node(
        "retrieve",
        retrieve,
    )

    workflow.add_node(
        "grade_documents",
        grade_documents,
    )

    workflow.add_node(
        "expand_context",
        expand_context,
    )

    workflow.add_node(
        "rewrite_query",
        rewrite_query,
    )

    workflow.add_node(
        "generate",
        generate,
    )

    workflow.add_node(
        "check_grounded",
        check_grounded,
    )

    # -------------------------------------------------------------
    # Entry
    # -------------------------------------------------------------

    workflow.set_entry_point(
        "route_query"
    )

    workflow.add_edge(
        "route_query",
        "classify_scope",
    )

    # -------------------------------------------------------------
    # Scope routing
    # -------------------------------------------------------------

    workflow.add_conditional_edges(
        "classify_scope",
        route_by_scope,
        {
            "profile":
                "build_profile_context",

            "retrieve":
                "retrieve",
        },
    )

    # -------------------------------------------------------------
    # Broad path
    # -------------------------------------------------------------

    workflow.add_edge(
        "build_profile_context",
        "generate",
    )

    # -------------------------------------------------------------
    # Retrieval path
    # -------------------------------------------------------------

    workflow.add_edge(
        "retrieve",
        "grade_documents",
    )

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate":
                "expand_context",

            "rewrite_query":
                "rewrite_query",
        },
    )

    # -------------------------------------------------------------
    # Context expansion
    # -------------------------------------------------------------

    workflow.add_edge(
        "expand_context",
        "generate",
    )

    # -------------------------------------------------------------
    # Query rewrite loop
    # -------------------------------------------------------------

    workflow.add_edge(
        "rewrite_query",
        "retrieve",
    )

    # -------------------------------------------------------------
    # Groundedness
    # -------------------------------------------------------------

    workflow.add_edge(
        "generate",
        "check_grounded",
    )

    workflow.add_edge(
        "check_grounded",
        END,
    )

    return workflow.compile()


# =====================================================================
# Public ask function
# =====================================================================

def ask(
    question: str,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run the complete ChatLens-RAG pipeline.

    Parameters
    ----------
    question:
        User's question.

    df:
        Enriched messages DataFrame.

        Required for broad/whole-conversation analysis.

    Returns
    -------
    dict
        answer
        grounded
        sources
        rewrites_used
        scope
        query_spec
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    graph = build_graph()

    initial_state: GraphState = {
        "question": question.strip(),
        "original_question": question.strip(),
        "documents": [],
        "generation": "",
        "rewrite_count": 0,
        "grounded": False,
        "semantic_grounded": False,
        "df": df,
        "scope": "",
        "primary_intent": "",
        "secondary_intents": [],
        "query_spec": None,
        "citation_validation": None,
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "available_citation_ids": [],
    }

    # Run the LangGraph pipeline
    result = graph.invoke(
        initial_state
    )

    return {
        "answer": result["generation"],

        "grounded": result["grounded"],

        "sources": [
            document.metadata
            for document in result["documents"]
        ],

        "rewrites_used": result["rewrite_count"],

        "scope": result["scope"],

        "query_spec": result.get("query_spec"),

        "citation_validation": result.get("citation_validation"),

        "valid_citation_ids": result.get("valid_citation_ids", []),

        "invalid_citation_ids": result.get("invalid_citation_ids", []),

        "available_citation_ids": result.get("available_citation_ids", []),
    }


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    import sys

    from parser import (
        parse_whatsapp_export,
    )

    from enrich import (
        enrich_messages,
        to_dataframe,
    )

    print(
        "=" * 70
    )

    print(
        "ChatLens-RAG Test"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Question
    # -------------------------------------------------------------

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What did we discuss about sports?"
    )

    print(
        f"\nQuestion:\n{question}"
    )

    # -------------------------------------------------------------
    # Parse
    # -------------------------------------------------------------

    print(
        "\nParsing WhatsApp export..."
    )

    msgs = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"Parsed {len(msgs)} messages."
    )

    # -------------------------------------------------------------
    # Enrich
    # -------------------------------------------------------------

    print(
        "\nEnriching messages..."
    )

    enriched = enrich_messages(
        msgs
    )

    print(
        f"Enriched {len(enriched)} messages."
    )

    # -------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------

    df = to_dataframe(
        enriched
    )

    # -------------------------------------------------------------
    # Run
    # -------------------------------------------------------------

    print(
        "\nRunning RAG pipeline..."
    )

    try:

        result = ask(
            question,
            df=df,
        )

    except Exception as e:

        print(
            "\n❌ RAG pipeline failed:"
        )

        print(
            e
        )

        raise

    # -------------------------------------------------------------
    # Result
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAnswer:\n{result['answer']}"
    )

    print(
        f"\nScope: "
        f"{result['scope']}"
    )

    print(
        f"Grounded: "
        f"{result['grounded']}"
    )

    print(
        f"Rewrites used: "
        f"{result['rewrites_used']}"
    )

    print(
        "\nSources:"
    )

    for source in result["sources"]:

        print(
            source
        )

    print(
        "\n✓ RAG test completed."
    )

