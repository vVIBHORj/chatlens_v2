
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
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, TypedDict, Dict, Tuple, Any, Set, Callable
from datetime import datetime

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from hybrid_retriever import hybrid_search
from analytics import build_style_profile
from context_expander import assemble_dialogue_evidence_units


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

SEMANTIC_CONTEXT_BEFORE = 3

SEMANTIC_CONTEXT_AFTER = 1

SEMANTIC_EXPANSION_MAX_WORDS = 6

SEMANTIC_EXPANSION_MAX_CHARS = 35

# Opt-in dialogue evidence experiment configuration
DIALOGUE_EVIDENCE_ENV_VAR = "CHATLENS_DIALOGUE_EVIDENCE_EXPERIMENT"
CONTEXTUAL_GRADING_ENV_VAR = "CHATLENS_CONTEXTUAL_GRADING_EXPERIMENT"
RETRIEVAL_STRATEGY_ENV_VAR = "CHATLENS_RETRIEVAL_STRATEGY"

STRATEGY_BASELINE = "baseline"
STRATEGY_DIALOGUE_EVIDENCE = "dialogue_evidence"
STRATEGY_CONTEXTUAL_GRADING = "contextual_grading"
VALID_RETRIEVAL_STRATEGIES = {
    STRATEGY_BASELINE,
    STRATEGY_DIALOGUE_EVIDENCE,
    STRATEGY_CONTEXTUAL_GRADING,
}

TOP_K_EVIDENCE_UNITS = 5
RRF_K_CONTEXTUAL = 60
WEIGHT_RETRIEVAL = 0.5
WEIGHT_CONTEXTUAL = 0.5


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
    grader_status: Optional[str]

    # Opt-in dialogue evidence & contextual grading experiment fields
    evidence_mode: Optional[str]
    dialogue_evidence_mode: Optional[bool]
    contextual_grading_mode: Optional[bool]
    retrieval_strategy: Optional[str]
    grader_llm: Optional[Any]
    database_path: Optional[str]
    embedding_function: Optional[Any]
    contextual_grading_decisions: Optional[List[Dict[str, Any]]]

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

def extract_message_body(text: str) -> str:
    """
    Extract the clean message text from a formatted transcript line or content.
    Strips leading [ID ...] or [message_id=...], [timestamp], and 'Sender: ' prefixes.
    """
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(
        r"^\[(?:ID|message_id)\s*=?\s*\d+\]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\[[^\]]+\]\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"^[^:\n]+:\s*",
        "",
        cleaned,
    )
    return cleaned.strip()


def should_expand_semantic_anchor(document: Document) -> bool:
    """
    Determine whether a retrieved anchor message in semantic RAG needs
    chronological context expansion.

    Single message candidates that are short conversational replies or
    elliptical fragments (e.g. 'Sub-level 4.', '9pm', 'Yes') lack standalone
    topic context and require preceding messages to be interpreted.

    Conversation chunks already contain multi-message dialogue and are not expanded.
    Self-contained factual messages with complete sentence structure are preserved
    as standalone anchors to avoid introducing conversational distractor noise.
    """
    metadata = document.metadata or {}
    if metadata.get("source") == "conversation_chunk":
        return False

    message_id = metadata.get("message_id")
    if message_id is None:
        return False

    body = extract_message_body(document.page_content)
    if not body:
        return False

    words = body.split()
    if len(words) <= SEMANTIC_EXPANSION_MAX_WORDS or len(body) <= SEMANTIC_EXPANSION_MAX_CHARS:
        return True

    return False


def get_grading_preceding_context(
    document: Document,
    database_path: Optional[str] = None,
) -> Optional[str]:
    """
    Fetch bounded preceding conversation context (up to 2 messages) for short
    single-message candidates to help the relevance grader evaluate them in context.
    Does not mutate the candidate document.
    """
    if not should_expand_semantic_anchor(document):
        return None

    message_id = document.metadata.get("message_id")
    if message_id is None:
        return None

    try:
        kwargs = {
            "before": 2,
            "after": 0,
        }
        if database_path is not None:
            kwargs["database_path"] = database_path

        rows = get_chronological_context(int(message_id), **kwargs)
    except Exception as e:
        print(
            f"[GRADER CONTEXT LOOKUP WARNING] Could not fetch preceding context for message_id={message_id}: {e}"
        )
        return None

    if not rows:
        return None

    preceding_rows = [
        r for r in rows
        if (r.get("id") if isinstance(r, dict) else r["id"]) != int(message_id)
    ]
    if not preceding_rows:
        return None

    lines = []
    for row in preceding_rows:
        if isinstance(row, dict) or hasattr(row, "__getitem__"):
            row_id = row["id"]
            timestamp = row["timestamp"]
            sender = row["sender"] or ""
            message = row["message"] or ""
        else:
            row_id = getattr(row, "id", None)
            timestamp = getattr(row, "timestamp", None)
            sender = getattr(row, "sender", "") or ""
            message = getattr(row, "message", "") or ""
        try:
            dt = datetime.fromisoformat(str(timestamp))
            timestamp_text = dt.strftime("%d/%m/%Y %I:%M %p")
        except Exception:
            timestamp_text = str(timestamp or "")
        lines.append(f"[ID {row_id}] [{timestamp_text}] {sender}: {message}")

    return "\n".join(lines)


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
    "2. Candidates may be single messages (which may include preceding conversational context for short replies) or conversation transcript chunks.\n"
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


def format_candidate_for_grader(
    idx: int,
    document: Document,
    preceding_context: Optional[str] = None,
) -> str:
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

    if preceding_context:
        return (
            f"{header}\n"
            f"Preceding Conversational Context:\n"
            f"{preceding_context.strip()}\n\n"
            f"Candidate Message to Evaluate:\n"
            f"{document.page_content.strip()}"
        )

    return f"{header}\n{document.page_content.strip()}"


def parse_grader_response(
    raw_text: str,
    candidate_indices: List[int],
) -> Dict[int, Tuple[str, str, str]]:
    """
    Parse and validate structured grader JSON output.
    Returns mapping from candidate index to (decision, reason, grader_status)
    where grader_status is 'success' or 'failure'.
    Fails open by returning ('uncertain', 'parsing_fallback', 'failure') for missing or unparseable entries.
    """
    results: Dict[int, Tuple[str, str, str]] = {
        idx: ("uncertain", "parsing_fallback", "failure") for idx in candidate_indices
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
                results[cand_idx] = (decision, reason, "success")
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
                    results[cand_idx] = (decision, "regex_extracted", "success")
            except Exception:
                pass

    return results


def grade_candidate_batch(
    question: str,
    batch_docs: List[Document],
    grader_llm: Optional[Any] = None,
    database_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    candidate_formatter: Optional[Callable[..., str]] = None,
    preceding_contexts: Optional[List[Optional[str]]] = None,
) -> List[Tuple[Document, str, str, str]]:
    """
    Grade a bounded batch of candidates using LLM in a single structured prompt.
    Returns list of tuples: (document, decision, reason, grader_status).
    """
    if not batch_docs:
        return []

    active_llm = grader_llm if grader_llm is not None else llm
    indices = list(range(1, len(batch_docs) + 1))
    active_formatter = candidate_formatter if candidate_formatter is not None else format_candidate_for_grader
    if preceding_contexts is not None:
        contexts = preceding_contexts
    else:
        contexts = [
            get_grading_preceding_context(doc, database_path=database_path)
            for doc in batch_docs
        ]
    formatted_chunks = []
    for idx, doc, ctx in zip(indices, batch_docs, contexts):
        try:
            chunk = active_formatter(idx, doc, preceding_context=ctx)
        except TypeError as te:
            if "unexpected keyword argument" in str(te) or "preceding_context" in str(te):
                try:
                    chunk = active_formatter(idx, doc, ctx)
                except TypeError:
                    chunk = active_formatter(idx, doc)
            else:
                raise
        formatted_chunks.append(chunk)
    candidates_text = "\n\n---\n\n".join(formatted_chunks)
    active_template = prompt_template if prompt_template is not None else GRADE_PROMPT_TEMPLATE
    prompt = active_template.format(
        question=question,
        candidates_text=candidates_text,
    )

    try:
        response = active_llm.invoke(prompt)
        raw_output = response.content if hasattr(response, "content") else str(response)
        parsed_decisions = parse_grader_response(raw_output, indices)
    except Exception as e:
        parsed_decisions = {
            idx: ("uncertain", f"grader_exception: {e}", "failure")
            for idx in indices
        }

    graded_batch = []
    for idx, doc in zip(indices, batch_docs):
        decision, reason, grader_status = parsed_decisions.get(
            idx, ("uncertain", "missing_index", "failure")
        )
        graded_batch.append((doc, decision, reason, grader_status))

    return graded_batch


# =====================================================================
# Grade retrieved documents
# =====================================================================

def grade_documents(
    state: GraphState,
    grader_llm: Optional[Any] = None,
    database_path: Optional[str] = None,
) -> GraphState:
    """
    Conservatively grade retrieved documents using bounded LLM batch evaluation.

    Rules:
    - Retains relevant and uncertain candidates.
    - Removes only candidates confidently determined to be irrelevant.
    - Fails open on parsing errors, timeouts, or LLM exceptions.
    - Distinguishes grader failure from successful uncertain decisions in metadata and state.
    - Preserves message candidates and transcript chunks as distinct evidence types.
    """
    documents = state.get("documents", [])
    if not documents:
        print("\nNo retrieved candidates to grade.")
        return {
            **state,
            "documents": [],
            "grader_status": "success",
        }

    question = state.get("original_question") or state.get("question", "")
    db_path = database_path or state.get("database_path")
    print(
        f"\nConservative relevance grading: evaluating {len(documents)} "
        f"candidates in batches of up to {GRADE_BATCH_SIZE}..."
    )

    kept_documents: List[Document] = []
    rejected_count = 0
    any_failure = False
    any_success = False

    active_grader = grader_llm or state.get("grader_llm")

    # Process in bounded batches
    for batch_start in range(0, len(documents), GRADE_BATCH_SIZE):
        batch = documents[batch_start : batch_start + GRADE_BATCH_SIZE]
        graded_results = grade_candidate_batch(
            question,
            batch,
            grader_llm=active_grader,
            database_path=db_path,
        )

        for doc, decision, reason, grader_status in graded_results:
            if grader_status == "failure":
                any_failure = True
            else:
                any_success = True

            new_metadata = dict(doc.metadata)
            new_metadata["relevance_decision"] = decision
            new_metadata["relevance_reason"] = reason
            new_metadata["grader_status"] = grader_status
            new_metadata["grader_failure"] = (grader_status == "failure")

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
                status_note = f" (grader_{grader_status})" if grader_status == "failure" else ""
                print(
                    f"  [{decision.upper()}{status_note}] {source} ID={item_id}: {reason}"
                )

    if any_failure and any_success:
        overall_grader_status = "degraded"
    elif any_failure:
        overall_grader_status = "failed"
    else:
        overall_grader_status = "success"

    print(
        f"Relevance grading summary: kept {len(kept_documents)}/"
        f"{len(documents)} candidates ({rejected_count} rejected as irrelevant, "
        f"status={overall_grader_status})."
    )

    return {
        **state,
        "documents": kept_documents,
        "grader_status": overall_grader_status,
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
    database_path: Optional[str] = None,
) -> GraphState:
    """
    Expand retrieved message anchors using chronological neighbors
    when appropriate.

    For semantic topic queries (primary_intent == 'semantic_rag'):
    - Short conversational replies and elliptical fragments (e.g. 'Sub-level 4.', '9pm')
      are selectively expanded with a bounded chronological window to recover
      their conversational antecedents and enable accurate citation.
    - Self-contained factual messages and multi-message conversation chunks
      are preserved as standalone anchors without unnecessary surrounding noise.

    For contextual queries (e.g. primary_intent == 'context'):
    - Retrieved anchors are expanded with the standard CONTEXT_BEFORE / CONTEXT_AFTER window.
    """

    expanded_documents = []
    primary_intent = state.get("primary_intent")
    db_path = database_path or state.get("database_path")

    def format_rows(rows: List[Any]) -> List[str]:
        lines = []
        for row in rows:
            timestamp = row["timestamp"]
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp_text = dt.strftime("%d/%m/%Y %I:%M %p")
            except Exception:
                timestamp_text = timestamp or ""
            sender = row["sender"] or ""
            message = row["message"] or ""
            lines.append(
                f"[ID {row['id']}] [{timestamp_text}] {sender}: {message}"
            )
        return lines

    def make_expanded_doc(document: Document, rows: List[Any], before: int, after: int) -> Document:
        lines = format_rows(rows)
        anchor_text = document.page_content
        expanded_transcript = (
            "[RETRIEVED ANCHOR]\n"
            f"{anchor_text}\n\n"
            "[CHRONOLOGICAL CONTEXT]\n"
            + "\n".join(lines)
        )
        new_metadata = dict(document.metadata)
        new_metadata["context_expanded"] = True
        new_metadata["context_messages_before"] = before
        new_metadata["context_messages_after"] = after
        new_metadata["context_method"] = "chronological"
        new_metadata["context_start_message_id"] = rows[0]["id"]
        new_metadata["context_end_message_id"] = rows[-1]["id"]
        return Document(
            page_content=expanded_transcript,
            metadata=new_metadata,
        )

    def make_unexpanded_doc(document: Document) -> Document:
        new_metadata = dict(document.metadata)
        new_metadata["context_expanded"] = False
        new_metadata["context_method"] = "retrieved_anchor"
        anchor_text = document.page_content
        if not anchor_text.startswith("[RETRIEVED ANCHOR]"):
            page_content = f"[RETRIEVED ANCHOR]\n{anchor_text}"
        else:
            page_content = anchor_text
        return Document(
            page_content=page_content,
            metadata=new_metadata,
        )

    # -------------------------------------------------------------
    # Semantic topic questions
    # -------------------------------------------------------------

    if primary_intent == "semantic_rag":
        expanded_count = 0
        unexpanded_count = 0

        for document in state.get("documents", [])[:20]:
            if should_expand_semantic_anchor(document):
                message_id = document.metadata.get("message_id")
                try:
                    kwargs = {
                        "before": SEMANTIC_CONTEXT_BEFORE,
                        "after": SEMANTIC_CONTEXT_AFTER,
                    }
                    if db_path is not None:
                        kwargs["database_path"] = db_path
                    rows = get_chronological_context(int(message_id), **kwargs)
                except Exception as e:
                    print(
                        f"[CONTEXT EXPANSION ERROR] Failed to fetch chronological context for message_id={message_id}: {e}"
                    )
                    rows = []

                if rows:
                    expanded_documents.append(
                        make_expanded_doc(document, rows, SEMANTIC_CONTEXT_BEFORE, SEMANTIC_CONTEXT_AFTER)
                    )
                    expanded_count += 1
                else:
                    expanded_documents.append(make_unexpanded_doc(document))
                    unexpanded_count += 1
            else:
                expanded_documents.append(make_unexpanded_doc(document))
                unexpanded_count += 1

        print(
            f"\nContext expansion for semantic query: {expanded_count} expanded, "
            f"{unexpanded_count} preserved as standalone anchors."
        )

        return {
            **state,
            "documents": expanded_documents,
        }

    # -------------------------------------------------------------
    # Other query types (e.g. CONTEXT)
    # -------------------------------------------------------------

    for document in state.get("documents", [])[:10]:
        message_id = document.metadata.get("message_id")
        source = document.metadata.get("source")

        rows = []
        try:
            kwargs = {
                "before": CONTEXT_BEFORE,
                "after": CONTEXT_AFTER,
            }
            if db_path is not None:
                kwargs["database_path"] = db_path

            if source == "conversation_chunk":
                start_id = document.metadata.get("start_message_id")
                end_id = document.metadata.get("end_message_id")
                if start_id is not None and end_id is not None:
                    rows = get_chronological_context(int(start_id), **kwargs)
            elif message_id is not None:
                rows = get_chronological_context(int(message_id), **kwargs)
        except Exception as e:
            print(
                f"[CONTEXT EXPANSION ERROR] Failed to fetch context for document: {e}"
            )
            rows = []

        if rows:
            expanded_documents.append(
                make_expanded_doc(document, rows, CONTEXT_BEFORE, CONTEXT_AFTER)
            )
        else:
            expanded_documents.append(make_unexpanded_doc(document))

    print(
        f"\nContext expansion: {len(expanded_documents)} documents processed."
    )

    return {
        **state,
        "documents": expanded_documents,
    }



# =====================================================================
# Citation ID Extraction for Evidence Context
# =====================================================================

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
    # 1. "[ID 123]" canonical format used in hybrid search, chunks, and chronological context
    # 2. "[message_id=123]" legacy format retained for backwards compatibility
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


# =====================================================================
# Build generation prompt
# =====================================================================

def build_generation_prompt(
    state: GraphState,
    documents: Optional[List[Document]] = None,
) -> str:
    """
    Construct the generation prompt for LLM answer synthesis, explicitly restricting
    permitted citation IDs to those present in the provided evidence.
    """
    docs = documents if documents is not None else state.get("documents", [])

    context = "\n\n---\n\n".join(
        doc.page_content
        for doc in docs
        if doc and hasattr(doc, "page_content") and doc.page_content
    )

    is_profile = any(
        doc.metadata.get("source") == "whole_conversation_profile"
        for doc in docs
        if doc and hasattr(doc, "metadata") and doc.metadata
    )

    question = state.get("original_question") or state.get("question", "")

    # -------------------------------------------------------------
    # No context
    # -------------------------------------------------------------
    if not context.strip():
        return (
            "You are answering a question about a WhatsApp conversation.\n\n"
            "There is no reliable retrieved context available.\n\n"
            f"Question:\n"
            f"{question}\n\n"
            "Do not invent an answer. Clearly say that the "
            "available chat context is insufficient."
        )

    # -------------------------------------------------------------
    # Context note and citation constraints
    # -------------------------------------------------------------
    if is_profile:
        context_note = (
            "The context below is a whole-conversation statistical "
            "profile covering the participants and the conversation "
            "rather than a small retrieved excerpt."
        )
        citation_rule = (
            "7. STATISTICAL PROFILE CONTEXT:\n"
            "   - The context below is a statistical profile without individual message IDs. "
            "Do not output message ID citations or bracketed numbers."
        )
        citation_reminder = "Do not output message ID citations."
    else:
        context_note = (
            "The context below contains conversation messages retrieved "
            "because they may be relevant to the user's question. Some "
            "messages are exact retrieved anchors and some are nearby "
            "chronological context. Nearby messages may be unrelated."
        )
        permitted_ids = sorted(extract_available_message_ids(docs))
        if permitted_ids:
            ids_str = ", ".join(str(mid) for mid in permitted_ids)
            citation_rule = (
                "7. CITATION INTEGRITY & PRECISE EVIDENCE ATTRIBUTION:\n"
                "   - Strict Claim-to-Message Attribution: Cite a message ID ONLY if that specific message's own text "
                "directly contains and supports the factual claim. Every factual claim must be supported by the specific message containing the relevant fact.\n"
                "   - Reply vs. Preceding Question: When a short reply answers a preceding question, cite ONLY the reply message that supplies the answer, "
                "NOT the preceding question, request, or suggestion. Do not cite a preceding question merely because it mentions the same topic or entity, "
                "and do not add extra citations simply to provide conversational background.\n"
                "   - Multi-Message Claims: When a claim genuinely requires multiple messages to establish the fact, cite only the messages strictly necessary to support that claim.\n"
                "   - Permitted IDs Only: Cite ONLY message IDs explicitly present in the supplied evidence. "
                "Never invent, infer, or extrapolate a message ID. "
                f"Permitted citation IDs in this context: [{ids_str}]. Any citation outside this permitted list is strictly invalid.\n"
                "   - Inline Placement & Syntax: Place each citation in inline brackets immediately after the specific factual claim it supports. "
                "Every citation MUST be enclosed in square brackets formatted as [ID <number>], for example [ID 0]. "
                "Always include the letters 'ID ' before the number inside brackets (do not write [0]; always write [ID 0]). "
                "Cite ONLY the smallest relevant set of message IDs directly supporting each claim. "
                "NEVER list, clump, or prefix message IDs at the beginning of the answer or before a sentence.\n"
                "   - Explicit Abstention Without Citations: If the supplied evidence does not contain information to answer the question, "
                "respond clearly in natural language and state that the information was not found or not mentioned in the conversation. "
                "In an abstention, do NOT output any citation brackets, message IDs, or placeholder citations. "
                "Never attach an unrelated message ID to an abstention merely because the message exists in context."
            )
            citation_reminder = (
                f"Cite ONLY the exact message ID whose text directly supports each claim, chosen strictly from [{ids_str}]. "
                "When a reply answers a question, cite the reply containing the answer, NOT the preceding question. "
                "Format every citation inside brackets as [ID <number>] placed directly after the claim it supports. "
                "Verify the message ID belongs to the actual supporting statement, not an adjacent message. "
                "Never start your answer with an ID. "
                "If the evidence does not contain the answer, state that the information was not found in a natural sentence and include no citation brackets."
            )
        else:
            citation_rule = (
                "7. CITATION INTEGRITY & PRECISE EVIDENCE ATTRIBUTION:\n"
                "   - No individual message IDs are present in the supplied evidence. Do not output any message citations.\n"
                "   - If the evidence does not support an answer, state that the information was not found "
                "without adding any citation brackets or placeholders."
            )
            citation_reminder = (
                "No message IDs are present in the evidence; do not cite any message IDs. "
                "If the evidence does not support an answer, state that the information was not found without any citation brackets."
            )

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
        "Do NOT add any knowledge or assumptions from outside the context. "
        "Do NOT infer an absent fact from unrelated or merely similar messages (e.g., do not confuse dev credentials with production secrets, or car keys with API keys). "
        "When the supplied evidence does not establish the requested fact, clearly state that the available conversation does not establish or mention it.\n\n"

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

        "6. Preserve Exact Facts in Short Replies: Short, numerical, temporal, location, and confirmation replies "
        "must retain the exact relevant fact (such as exact times, IP addresses, locations, and confirmation words like 'clean' or 'yes'). "
        "For example, when a reply states 'Yes, completely clean', preserve the confirmation that it was clean. "
        "Do not omit critical qualifiers or add names, events, causes, statuses, or conclusions unsupported by the cited message.\n\n"

        f"{citation_rule}\n\n"

        "8. If the retrieved evidence supports only a limited conclusion (such as inquiries or mentions rather than "
        "actual activities or plans), state that limitation concisely instead of filling gaps with assumptions.\n\n"

        "9. Never infer participants, plans, intentions, relationships, or future actions unless the context explicitly "
        "states them.\n\n"

        "CONTEXT:\n"
        f"{context}\n\n"

        "USER QUESTION:\n"
        f"{question}\n\n"

        f"Remember: Write a natural 1–3 sentence assistant summary. Place inline citations formatted as [ID <number>] directly "
        f"after each supported factual claim (using the actual supporting message ID). Never list IDs at the beginning. "
        f"Summarize only supported, relevant points. {citation_reminder}\n\n"

        "ANSWER:"
    )

    return prompt


# =====================================================================
# Generate answer
# =====================================================================

def generate(
    state: GraphState,
    generation_llm: Optional[Any] = None,
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

    prompt = build_generation_prompt(state, documents=documents)
    active_llm = generation_llm if generation_llm is not None else llm

    try:
        response = active_llm.invoke(prompt)
        raw_answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        raw_answer = (
            "I was unable to generate the answer "
            f"because the local language model returned "
            f"an error: {e}"
        )

    answer = raw_answer
    raw_citations_stripped = False

    # Clean abstention representation enforcement:
    # If the generated response is purely an abstention stating that information is not found/mentioned,
    # strip any extraneous citation brackets that the model mistakenly attached.
    if is_pure_abstention(raw_answer):
        cleaned_answer = re.sub(r"\s*\[(?:ID|message_id)\s*=?\s*\d+\]", "", raw_answer, flags=re.IGNORECASE).strip()
        if cleaned_answer != raw_answer:
            raw_citations_stripped = True
            answer = cleaned_answer

    return {
        **state,
        "raw_generation": raw_answer,
        "generation": answer,
        "abstention_citations_stripped": raw_citations_stripped,
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


def is_pure_abstention(answer: str) -> bool:
    """
    Check if the answer is purely an explicit abstention stating that the requested
    information was not found, not mentioned, or not discussed in the conversation.
    Ensures:
      - Does not match if any sentence contains a substantive positive claim.
      - Does not match mixed answers (substantive claim + absent claim).
      - Does not match questions.
      - Matches only when all substantive content indicates lack of information in the context.
    """
    if not answer or not isinstance(answer, str):
        return False

    raw_text = answer.strip()
    if not raw_text or "?" in raw_text:
        return False

    # Extract clauses and sentences: split on sentence ends, semicolons, and contrastive conjunctions
    # First remove bracketed citations to evaluate the prose
    prose = re.sub(r"\[(?:ID|message_id)\s*=?\s*\d+\]", "", raw_text, flags=re.IGNORECASE).strip()
    clauses = [
        c.strip()
        for c in re.split(r"(?:[.!?]+|\s*;\s*|,\s*(?:but|however|although|while|though|yet)\s+|\b(?:but|however|although|while|though)\b)", prose)
        if c.strip()
    ]
    if not clauses:
        return False

    abstention_patterns = [
        r"\b(?:was|is|were)\s+not\s+(?:found|mentioned|discussed|stated|provided|recorded)\b",
        r"\bdoes\s+not\s+(?:mention|state|contain|discuss|provide|record|indicate)\b",
        r"\b(?:no|insufficient)\s+(?:information|evidence|mention|data|record)\b",
        r"\bnot\s+found\s+in\s+the\s+(?:conversation|provided|archive|context|messages)\b",
        r"\bunable\s+to\s+(?:find|determine|answer)\b",
        r"\bcannot\s+(?:find|determine|be\s+determined|answer)\b",
        r"\bthere\s+is\s+no\s+(?:mention|information|evidence)\b",
        r"\bno\s+mention\s+of\b",
    ]

    def clause_is_abstention(s: str) -> bool:
        s_lower = s.lower()
        return any(re.search(pat, s_lower) for pat in abstention_patterns)

    # In a pure abstention, every non-trivial clause must express lack of information
    # or be an explanatory clause of the abstention (e.g. "so its storage location cannot be determined")
    for s in clauses:
        if not clause_is_abstention(s):
            if re.search(r"\b(?:so|therefore|hence|as\s+a\s+result)\b.*?\b(?:cannot|unknown|unclear|undetermined)\b", s, re.IGNORECASE):
                continue
            return False

    return True


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

    # For specific retrieval queries, citations are expected; report missing citations unless the answer is an explicit abstention
    is_abstention = is_pure_abstention(answer)
    missing_citations = (scope != "broad" and not has_citations and len(available_ids) > 0 and not is_abstention)

    all_cited_valid = (
        has_citations
        and len(invalid_cited_ids) == 0
        and len(malformed_citations) == 0
        and not is_abstention
    )

    if not has_citations:
        if is_abstention:
            details = "Valid abstention: No citations provided for absent information."
        elif missing_citations:
            details = "No citations provided in answer for specific retrieval query."
        else:
            details = "No citations provided."
    elif is_abstention:
        details = (
            f"Citation validation warning: citations attached to an explicit abstention ({cited_ids})."
        )
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
        "conversation analysis system.\n"
        "Determine whether the substantive claims and citations in the ANSWER are supported by the supplied CONTEXT.\n\n"

        "Evaluation Guidelines:\n"
        "1. Supported Claims: A claim is supported if the context explicitly states it, "
        "or if it is a direct paraphrase of facts stated in the context (e.g., 'vessel departure from lunar orbit' "
        "supports 'spaceship leaves the moon').\n"
        "2. Chronological Context: Questions and adjacent messages in context establish what short replies refer to.\n"
        "3. Unsupported Claims: If the answer asserts events, numbers, participants, or actions that are contradicted "
        "by context or not mentioned, mark 'no'. Do not infer completed actions from mere venue or topic mentions.\n"
        "4. Citation Attribution: When the answer cites [ID X], verify that the message labelled [ID X] in context "
        "actually contains or directly supports that claim. If the fact is in a different message, or if [ID X] is cited "
        "for a preceding question rather than the reply supplying the answer, mark 'no'.\n"
        "5. Explicit Abstentions: If the answer states that the requested information was not found or not mentioned in the conversation, "
        "mark 'yes' if the context indeed lacks that information and no irrelevant citations are attached. If the answer states that information was not found but attaches an unrelated citation, mark 'no'.\n\n"

        f"CONTEXT:\n"
        f"{context}\n\n"

        f"ANSWER:\n"
        f"{answer}\n\n"

        "Assess each substantive claim and citation against the context in 1-2 sentences, then state your final decision as exactly 'Decision: yes' or 'Decision: no'.\n"
        "Assessment:"
    )

    active_llm = grounding_llm if grounding_llm is not None else llm

    try:
        raw_response = active_llm.invoke(prompt)
        text = raw_response.content.strip() if hasattr(raw_response, "content") else str(raw_response).strip()

        print(
            f"[GROUNDEDNESS CHECK] Raw response:\n{text}"
        )

        decision_match = re.search(r"decision\s*:\s*(yes|no)", text, re.IGNORECASE)
        if decision_match:
            semantic_grounded = (decision_match.group(1).lower() == "yes")
        else:
            cleaned_first = text.strip().lower()
            semantic_grounded = cleaned_first.startswith("yes") or "decision: yes" in cleaned_first

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
# Opt-In Dialogue Evidence Pipeline Helpers
# =====================================================================

def resolve_retrieval_strategy(state: Optional[GraphState] = None) -> str:
    """
    Resolve which retrieval strategy to execute:
      1. 'baseline' (default production route: grade_documents -> expand_context)
      2. 'dialogue_evidence' (opt-in rank-only treatment: assemble -> rank)
      3. 'contextual_grading' (opt-in prototype: assemble -> grade -> rank)

    Precedence:
      1. Explicit state keys: 'retrieval_strategy', 'contextual_grading_mode', 'dialogue_evidence_mode'.
      2. Environment variables: CHATLENS_RETRIEVAL_STRATEGY, CHATLENS_CONTEXTUAL_GRADING_EXPERIMENT, CHATLENS_DIALOGUE_EVIDENCE_EXPERIMENT.
      3. Default fallback: 'baseline'.

    Conflict policy:
      - If multiple conflicting modes are explicitly asserted simultaneously, raises ValueError.
    """
    if state is not None:
        ctx_flag = state.get("contextual_grading_mode")
        diag_flag = state.get("dialogue_evidence_mode")
        legacy_diag_flag = state.get("use_dialogue_evidence")
        if diag_flag is None and legacy_diag_flag is not None:
            diag_flag = legacy_diag_flag

        strat_val = state.get("retrieval_strategy")
        if strat_val is not None:
            norm_strat = str(strat_val).strip().lower()
            if norm_strat not in VALID_RETRIEVAL_STRATEGIES:
                raise ValueError(
                    f"Unknown retrieval_strategy: {strat_val!r}. "
                    f"Must be one of: {sorted(VALID_RETRIEVAL_STRATEGIES)}"
                )
            if norm_strat == STRATEGY_BASELINE and (bool(ctx_flag) or bool(diag_flag)):
                raise ValueError("Conflicting configuration: retrieval_strategy='baseline' but experimental mode flag is True.")
            if norm_strat == STRATEGY_DIALOGUE_EVIDENCE and bool(ctx_flag):
                raise ValueError("Conflicting configuration: retrieval_strategy='dialogue_evidence' but contextual_grading_mode is True.")
            if norm_strat == STRATEGY_CONTEXTUAL_GRADING and bool(diag_flag):
                raise ValueError("Conflicting configuration: retrieval_strategy='contextual_grading' but dialogue_evidence_mode is True.")
            return norm_strat

        if bool(ctx_flag) and bool(diag_flag):
            raise ValueError(
                "Conflicting retrieval experiment flags: both dialogue_evidence_mode and "
                "contextual_grading_mode are enabled. Please specify only one strategy."
            )
        if bool(ctx_flag):
            return STRATEGY_CONTEXTUAL_GRADING
        if bool(diag_flag):
            return STRATEGY_DIALOGUE_EVIDENCE
        if ctx_flag is False and diag_flag is False:
            return STRATEGY_BASELINE

    # Check environment variables
    env_strat = os.environ.get(RETRIEVAL_STRATEGY_ENV_VAR, "").strip().lower()
    if env_strat:
        if env_strat not in VALID_RETRIEVAL_STRATEGIES:
            raise ValueError(
                f"Unknown CHATLENS_RETRIEVAL_STRATEGY env var: {env_strat!r}. "
                f"Must be one of: {sorted(VALID_RETRIEVAL_STRATEGIES)}"
            )
        return env_strat

    env_ctx = os.environ.get(CONTEXTUAL_GRADING_ENV_VAR, "0").strip().lower() in ("1", "true", "yes", "on")
    env_diag = os.environ.get(DIALOGUE_EVIDENCE_ENV_VAR, "0").strip().lower() in ("1", "true", "yes", "on")

    if env_ctx and env_diag:
        raise ValueError(
            "Conflicting environment variables: both CHATLENS_DIALOGUE_EVIDENCE_EXPERIMENT and "
            "CHATLENS_CONTEXTUAL_GRADING_EXPERIMENT are active. Please activate only one experiment."
        )
    if env_ctx:
        return STRATEGY_CONTEXTUAL_GRADING
    if env_diag:
        return STRATEGY_DIALOGUE_EVIDENCE

    return STRATEGY_BASELINE


def is_dialogue_evidence_enabled(state: Optional[GraphState] = None) -> bool:
    """Check whether the opt-in rank-only dialogue-evidence retrieval pipeline is active."""
    try:
        return resolve_retrieval_strategy(state) == STRATEGY_DIALOGUE_EVIDENCE
    except ValueError:
        return False


def is_contextual_grading_enabled(state: Optional[GraphState] = None) -> bool:
    """Check whether the opt-in contextual-grading retrieval pipeline is active."""
    try:
        return resolve_retrieval_strategy(state) == STRATEGY_CONTEXTUAL_GRADING
    except ValueError:
        return False


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two floating-point vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def rank_dialogue_evidence_units(
    question: str,
    evidence_units: List[Document],
    embedding_function: Optional[Any] = None,
) -> List[Document]:
    """
    Ranks assembled dialogue evidence units using Reciprocal Rank Fusion (RRF)
    combining:
      Signal A: Original retrieval ranking from hybrid search (protects lexical/dense search work).
      Signal B: Contextual evidence semantic similarity (scores query against complete dialogue transcript).

    Formula:
      fused_score(d) = (WEIGHT_RETRIEVAL / (RRF_K + rank_retrieval(d))) +
                       (WEIGHT_CONTEXTUAL / (RRF_K + rank_contextual(d)))

    Tie-breaking:
      Deterministic ordering by (-fused_score, original_rank, anchor_id).

    Fallback:
      If contextual embeddings cannot be calculated (e.g., service unavailable or error),
      falls back safely to Signal A (retrieval order) without discarding candidates.
    """
    if not evidence_units:
        return []

    # 1. Establish Signal A: original retrieval rank (1-indexed based on input order)
    docs_with_rank = []
    for idx, doc in enumerate(evidence_units, start=1):
        new_doc = Document(
            page_content=doc.page_content,
            metadata=dict(doc.metadata) if doc.metadata else {},
        )
        new_doc.metadata["retrieval_rank"] = idx
        docs_with_rank.append(new_doc)

    # 2. Establish Signal B: contextual similarity using embedding interface
    embedder = embedding_function
    if embedder is None:
        try:
            from vectorstore import get_embeddings
            embedder = get_embeddings()
        except Exception:
            embedder = None

    contextual_success = False
    if embedder is not None and question and question.strip():
        try:
            q_emb = embedder.embed_query(question.strip())
            texts = [d.page_content for d in docs_with_rank]
            doc_embs = embedder.embed_documents(texts)

            if len(doc_embs) == len(docs_with_rank):
                for doc, d_emb in zip(docs_with_rank, doc_embs):
                    sim = compute_cosine_similarity(q_emb, d_emb)
                    doc.metadata["contextual_similarity"] = round(sim, 4)
                contextual_success = True
        except Exception as e:
            print(f"[CONTEXTUAL RANKING WARNING] Could not compute contextual embeddings: {e}")
            contextual_success = False

    # 3. Calculate Contextual Ranks and RRF Fusion
    if contextual_success:
        # Sort by contextual_similarity DESC to assign contextual ranks
        def context_sort_key(d: Document):
            sim = d.metadata.get("contextual_similarity", -1.0)
            ret_rank = d.metadata.get("retrieval_rank", 999999)
            anchor_id = d.metadata.get("anchor_message_id") or 999999
            try:
                aid = int(anchor_id)
            except (ValueError, TypeError):
                aid = 999999
            return (-sim, ret_rank, aid)

        docs_by_context = sorted(docs_with_rank, key=context_sort_key)
        for c_rank, doc in enumerate(docs_by_context, start=1):
            doc.metadata["contextual_rank"] = c_rank
            doc.metadata["contextual_ranking_status"] = "success"
            doc.metadata["contextual_ranking_fallback"] = False

        # Calculate Fused Score
        for doc in docs_with_rank:
            r_ret = doc.metadata.get("retrieval_rank", len(docs_with_rank))
            r_ctx = doc.metadata.get("contextual_rank", len(docs_with_rank))
            fused = (WEIGHT_RETRIEVAL / (RRF_K_CONTEXTUAL + r_ret)) + (
                WEIGHT_CONTEXTUAL / (RRF_K_CONTEXTUAL + r_ctx)
            )
            doc.metadata["fused_score"] = round(fused, 6)
    else:
        # Fallback to pure retrieval ranking
        for doc in docs_with_rank:
            r_ret = doc.metadata.get("retrieval_rank", len(docs_with_rank))
            fused = 1.0 / (RRF_K_CONTEXTUAL + r_ret)
            doc.metadata["fused_score"] = round(fused, 6)
            doc.metadata["contextual_rank"] = None
            doc.metadata["contextual_similarity"] = None
            doc.metadata["contextual_ranking_status"] = "fallback_retrieval_order"
            doc.metadata["contextual_ranking_fallback"] = True

    # 4. Final Deterministic Ordering
    def final_sort_key(d: Document):
        fused = d.metadata.get("fused_score", 0.0)
        sim = d.metadata.get("contextual_similarity")
        sim_val = float(sim) if sim is not None else -1.0
        ret_rank = d.metadata.get("retrieval_rank", 999999)
        anchor_id = d.metadata.get("anchor_message_id")
        try:
            aid = int(anchor_id) if anchor_id is not None else 999999
        except (ValueError, TypeError):
            aid = 999999
        return (-fused, -sim_val, ret_rank, aid)

    ranked_docs = sorted(docs_with_rank, key=final_sort_key)
    return ranked_docs


def assemble_and_rank_evidence(
    state: GraphState,
    embedding_function: Optional[Any] = None,
    database_path: Optional[str] = None,
) -> GraphState:
    """
    Opt-in LangGraph node for dialogue-evidence pipeline:
      1. Assembles retrieved message anchors into chronological dialogue units.
      2. Ranks complete evidence units using hybrid retrieval + contextual transcript similarity.
      3. Bounds evidence to TOP_K_EVIDENCE_UNITS.
    """
    documents = state.get("documents", [])
    question = state.get("original_question") or state.get("question", "")
    db_path = database_path or state.get("database_path")

    print("\n" + "=" * 70)
    print("EXPERIMENTAL DIALOGUE EVIDENCE UNIT ASSEMBLY & RERANKING")
    print("=" * 70)
    print(f"Retrieved candidate documents: {len(documents)}")

    if not documents:
        print("No candidates retrieved to assemble.")
        return {
            **state,
            "documents": [],
            "evidence_mode": "dialogue_evidence_unit",
        }

    # 1. Assemble dialogue units using Task 36 helper
    evidence_units = assemble_dialogue_evidence_units(
        documents,
        before=2,
        after=1,
        database_path=db_path,
    )
    print(f"Assembled dialogue evidence units: {len(evidence_units)}")

    # 2. Contextually rank evidence units
    embedder = embedding_function or state.get("embedding_function")
    ranked_units = rank_dialogue_evidence_units(
        question=question,
        evidence_units=evidence_units,
        embedding_function=embedder,
    )

    # 3. Select bounded top-K evidence units
    bounded_units = ranked_units[:TOP_K_EVIDENCE_UNITS]
    print(f"Selected top-{len(bounded_units)} bounded evidence units for generation:")
    for idx, unit in enumerate(bounded_units, start=1):
        anchor_id = unit.metadata.get("anchor_message_id") or unit.metadata.get("message_id")
        fused = unit.metadata.get("fused_score", 0.0)
        ret_rank = unit.metadata.get("retrieval_rank")
        ctx_rank = unit.metadata.get("contextual_rank")
        print(f"  #{idx} Anchor ID={anchor_id} | fused_score={fused:.6f} (ret_rank={ret_rank}, ctx_rank={ctx_rank})")

    return {
        **state,
        "documents": bounded_units,
        "evidence_mode": "dialogue_evidence_unit",
    }


def format_candidate_for_contextual_grader(
    idx: int,
    document: Document,
    preceding_context: Optional[str] = None,
    *args: Any,
    **kwargs: Any,
) -> str:
    """Format an assembled dialogue evidence unit for the relevance grader."""
    anchor_id = (
        document.metadata.get("anchor_message_id")
        or document.metadata.get("message_id")
        or "unknown"
    )
    header = f"[Candidate {idx}] (Dialogue Evidence Unit | Anchor ID: {anchor_id})"
    return f"{header}\n{document.page_content.strip()}"


def assemble_grade_rank_evidence(
    state: GraphState,
    grader_llm: Optional[Any] = None,
    embedding_function: Optional[Any] = None,
    database_path: Optional[str] = None,
) -> GraphState:
    """
    Opt-in LangGraph node for third retrieval strategy: Contextual Grading After Evidence Assembly.
    Pipeline:
      1. Assembles retrieved message anchors into chronological dialogue units.
      2. Relevance grades the complete assembled evidence units in bounded batches.
      3. Retains units marked 'relevant' or 'uncertain' (fails open on grader failure).
      4. Ranks retained units using hybrid retrieval + contextual transcript similarity (RRF).
      5. Bounds evidence to TOP_K_EVIDENCE_UNITS.
    """
    documents = state.get("documents", [])
    question = state.get("original_question") or state.get("question", "")
    db_path = database_path or state.get("database_path")

    print("\n" + "=" * 70)
    print("EXPERIMENTAL CONTEXTUAL RELEVANCE GRADING AFTER EVIDENCE ASSEMBLY")
    print("=" * 70)
    print(f"Retrieved candidate documents: {len(documents)}")

    if not documents:
        print("No candidates retrieved to assemble and grade.")
        return {
            **state,
            "documents": [],
            "evidence_mode": "contextual_grading",
            "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING,
            "grader_status": "success",
        }

    # 1. Assemble complete dialogue units using context_expander helper
    evidence_units = assemble_dialogue_evidence_units(
        documents,
        before=2,
        after=1,
        database_path=db_path,
    )
    print(f"Assembled dialogue evidence units: {len(evidence_units)}")

    if not evidence_units:
        return {
            **state,
            "documents": [],
            "evidence_mode": "contextual_grading",
            "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING,
            "grader_status": "success",
        }

    # 2. Grade assembled units conservatively using LLM relevance grader
    active_llm = grader_llm or state.get("grader_llm") or llm
    kept_units: List[Document] = []
    grading_decisions: List[Dict[str, Any]] = []
    rejected_count = 0
    any_failure = False
    any_success = False

    for batch_start in range(0, len(evidence_units), GRADE_BATCH_SIZE):
        batch = evidence_units[batch_start : batch_start + GRADE_BATCH_SIZE]
        graded_results = grade_candidate_batch(
            question=question,
            batch_docs=batch,
            grader_llm=active_llm,
            database_path=db_path,
            candidate_formatter=format_candidate_for_contextual_grader,
            preceding_contexts=[None] * len(batch),
        )

        for doc, decision, reason, grader_status in graded_results:
            if grader_status == "failure":
                any_failure = True
            else:
                any_success = True

            new_metadata = dict(doc.metadata)
            new_metadata["relevance_decision"] = decision
            new_metadata["relevance_reason"] = reason
            new_metadata["grader_status"] = grader_status
            new_metadata["grader_failure"] = (grader_status == "failure")

            updated_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata,
            )

            anchor_id = new_metadata.get("anchor_message_id") or new_metadata.get("message_id") or "?"
            grading_decisions.append({
                "anchor_message_id": anchor_id,
                "included_message_ids": new_metadata.get("included_message_ids", []),
                "decision": decision,
                "reason": reason,
                "grader_status": grader_status,
                "grader_failure": (grader_status == "failure"),
            })
            if decision == "irrelevant":
                rejected_count += 1
                print(f"  [REJECTED] DialogueUnit Anchor={anchor_id}: {reason}")
            else:
                kept_units.append(updated_doc)
                status_note = f" (grader_{grader_status})" if grader_status == "failure" else ""
                print(f"  [{decision.upper()}{status_note}] DialogueUnit Anchor={anchor_id}: {reason}")

    if any_failure and any_success:
        overall_grader_status = "degraded"
    elif any_failure:
        overall_grader_status = "failed"
    else:
        overall_grader_status = "success"

    print(
        f"Contextual grading summary: kept {len(kept_units)}/{len(evidence_units)} "
        f"evidence units ({rejected_count} rejected as irrelevant, status={overall_grader_status})."
    )

    # 3. Contextual RRF ranking ONLY on retained units (rejected candidates can never re-enter)
    if kept_units:
        embedder = embedding_function or state.get("embedding_function")
        ranked_units = rank_dialogue_evidence_units(
            question=question,
            evidence_units=kept_units,
            embedding_function=embedder,
        )
        bounded_units = ranked_units[:TOP_K_EVIDENCE_UNITS]
    else:
        bounded_units = []

    print(f"Selected top-{len(bounded_units)} bounded evidence units for generation:")
    for idx, unit in enumerate(bounded_units, start=1):
        anchor_id = unit.metadata.get("anchor_message_id") or unit.metadata.get("message_id")
        fused = unit.metadata.get("fused_score", 0.0)
        ret_rank = unit.metadata.get("retrieval_rank")
        ctx_rank = unit.metadata.get("contextual_rank")
        print(f"  #{idx} Anchor ID={anchor_id} | fused_score={fused:.6f} (ret_rank={ret_rank}, ctx_rank={ctx_rank})")

    return {
        **state,
        "documents": bounded_units,
        "evidence_mode": "contextual_grading",
        "retrieval_strategy": STRATEGY_CONTEXTUAL_GRADING,
        "grader_status": overall_grader_status,
        "contextual_grading_decisions": grading_decisions,
    }


def route_post_retrieval(state: GraphState) -> str:
    """
    Routing decision immediately after retrieval:
    - If contextual grading experiment is enabled: routes to assemble_grade_rank_evidence.
    - Else if dialogue evidence experiment is enabled: routes to assemble_and_rank_evidence.
    - Otherwise (default production behavior): routes to grade_documents.
    """
    strategy = resolve_retrieval_strategy(state)
    if strategy == STRATEGY_CONTEXTUAL_GRADING:
        return "assemble_grade_rank_evidence"
    if strategy == STRATEGY_DIALOGUE_EVIDENCE:
        return "assemble_and_rank_evidence"
    return "grade_documents"


def decide_post_assembly(state: GraphState) -> str:
    """
    Routing decision immediately after dialogue evidence unit assembly & ranking:
    - If no documents remain and rewrites remain: routes to rewrite_query.
    - Otherwise: proceeds directly to grounded generation.
    """
    documents = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    if not documents and rewrite_count < MAX_REWRITES:
        return "rewrite_query"

    return "generate"


def decide_post_contextual_grading(state: GraphState) -> str:
    """
    Routing decision immediately after contextual grading & ranking:
    - If no documents remain and rewrites remain: routes to rewrite_query.
    - Otherwise: proceeds directly to grounded generation.
    """
    documents = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    if not documents and rewrite_count < MAX_REWRITES:
        return "rewrite_query"

    return "generate"


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
        "assemble_and_rank_evidence",
        assemble_and_rank_evidence,
    )

    workflow.add_node(
        "assemble_grade_rank_evidence",
        assemble_grade_rank_evidence,
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
    # Retrieval path (conditional post-retrieval routing)
    # -------------------------------------------------------------

    workflow.add_conditional_edges(
        "retrieve",
        route_post_retrieval,
        {
            "grade_documents":
                "grade_documents",

            "assemble_and_rank_evidence":
                "assemble_and_rank_evidence",

            "assemble_grade_rank_evidence":
                "assemble_grade_rank_evidence",
        },
    )

    # Default production path: relevance grading -> decision
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

    # Experimental dialogue evidence path: assembly & ranking -> decision
    workflow.add_conditional_edges(
        "assemble_and_rank_evidence",
        decide_post_assembly,
        {
            "generate":
                "generate",

            "rewrite_query":
                "rewrite_query",
        },
    )

    # Experimental contextual grading path: assembly -> grading -> ranking -> decision
    workflow.add_conditional_edges(
        "assemble_grade_rank_evidence",
        decide_post_contextual_grading,
        {
            "generate":
                "generate",

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
    dialogue_evidence_mode: Optional[bool] = None,
    contextual_grading_mode: Optional[bool] = None,
    retrieval_strategy: Optional[str] = None,
    grader_llm: Optional[Any] = None,
    embedding_function: Optional[Any] = None,
    database_path: Optional[str] = None,
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

    dialogue_evidence_mode:
        Optional boolean override for opt-in dialogue evidence experiment.
        Defaults to None (falls back to CHATLENS_DIALOGUE_EVIDENCE_EXPERIMENT env var).

    contextual_grading_mode:
        Optional boolean override for opt-in contextual grading experiment.
        Defaults to None (falls back to CHATLENS_CONTEXTUAL_GRADING_EXPERIMENT env var).

    retrieval_strategy:
        Optional explicit strategy name ('baseline', 'dialogue_evidence', 'contextual_grading').
        Defaults to None (falls back to CHATLENS_RETRIEVAL_STRATEGY env var).

    grader_llm:
        Optional custom grader LLM instance (e.g. for offline testing).

    embedding_function:
        Optional custom embeddings instance for contextual ranking.

    database_path:
        Optional explicit database path for isolated environments.

    Returns
    -------
    dict
        answer
        grounded
        sources
        rewrites_used
        scope
        query_spec
        evidence_mode
        retrieval_strategy
        grader_status
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    # Validate strategy configuration early to fail closed on conflicting inputs
    test_state: GraphState = {
        "dialogue_evidence_mode": dialogue_evidence_mode,
        "contextual_grading_mode": contextual_grading_mode,
        "retrieval_strategy": retrieval_strategy,
    }
    resolved_strategy = resolve_retrieval_strategy(test_state)

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
        "grader_status": None,
        "evidence_mode": "standard",
        "dialogue_evidence_mode": dialogue_evidence_mode,
        "contextual_grading_mode": contextual_grading_mode,
        "retrieval_strategy": resolved_strategy,
        "grader_llm": grader_llm,
        "embedding_function": embedding_function,
        "database_path": database_path,
    }

    # Run the LangGraph pipeline
    result = graph.invoke(
        initial_state
    )

    return {
        "answer": result["generation"],

        "raw_answer": result.get("raw_generation", result["generation"]),

        "abstention_citations_stripped": result.get("abstention_citations_stripped", False),

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

        "grader_status": result.get("grader_status"),

        "evidence_mode": result.get("evidence_mode", "standard"),

        "retrieval_strategy": result.get("retrieval_strategy", resolved_strategy),

        "contextual_grading_decisions": result.get("contextual_grading_decisions", []),
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

