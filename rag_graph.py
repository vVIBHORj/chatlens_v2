
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

from typing import List, Optional, TypedDict
from datetime import datetime

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from hybrid_retriever import hybrid_search
from analytics import build_style_profile

from query_router import route_query

from chat_database import (
    search_messages,
    get_chronological_context,
)


# =====================================================================
# Configuration
# =====================================================================

MAX_REWRITES = 2

SEMANTIC_TOP_K = 10

KEYWORD_TOP_K = 10

MAX_HYBRID_CANDIDATES = 20

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

class GraphState(TypedDict):
    question: str

    original_question: str

    documents: List[Document]

    generation: str

    rewrite_count: int

    grounded: bool

    df: Optional[pd.DataFrame]

    scope: str

    primary_intent: str

    secondary_intents: List[str]


# =====================================================================
# Scope classification
# =====================================================================
# =====================================================================
# Query routing
# =====================================================================

def route_query_node(
    state: GraphState,
) -> GraphState:
    """
    Detect the primary and secondary analytical intents
    before the existing scope/retrieval pipeline runs.

    This step is observational for now.
    The detected intents do NOT control graph routing yet.
    """

    route = route_query(
        state["question"]
    )

    primary_intent = route.primary_intent.value

    secondary_intents = [
        intent.value
        for intent in route.secondary_intents
    ]

    print(
        f"[QUERY ROUTER] "
        f"Primary={primary_intent} "
        f"Secondary={secondary_intents}"
    )

    return {
        **state,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
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

        content = (
            f"[ID {candidate.message_id}] "
            f"[{candidate.timestamp}] "
            f"{candidate.sender or ''}: "
            f"{candidate.message or ''}"
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
# Grade retrieved documents
# =====================================================================

def grade_documents(
    state: GraphState,
) -> GraphState:
    """
    Keep the full hybrid retrieval candidate pool.

    Retrieval is recall-first:
    semantic and keyword retrieval identify candidates,
    while the LLM should reason over the retrieved context
    instead of acting as a hard recall filter.
    """

    documents = state["documents"]

    print(
        "\nSkipping LLM relevance grading."
    )

    print(
        f"Retrieved candidates kept: "
        f"{len(documents)}"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        print(
            f"  Candidate {index}: "
            f"[{source}]"
        )

    return {
        **state,
        "documents": documents,
    }


# =====================================================================
# Decide whether to generate or rewrite
# =====================================================================

def decide_to_generate(
    state: GraphState,
) -> str:
    """
    Decide what happens after document grading.
    """

    if state["documents"]:

        return "generate"

    if (
        state["rewrite_count"]
        >= MAX_REWRITES
    ):

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
        state["question"]
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
            "You are an evidence-grounded assistant analyzing a "
            "WhatsApp conversation.\n\n"

            f"{context_note}\n\n"

            "IMPORTANT RULES:\n\n"

            "1. Use ONLY information contained in the context.\n\n"

            "2. Treat messages that directly mention the subject of "
            "the user's question as strong evidence, even when the "
            "surrounding chronological messages are unrelated.\n\n"

            "3. Synthesize evidence across ALL retrieved sections. "
            "Do not judge relevance based only on one chronological "
            "section.\n\n"

            "4. Distinguish direct evidence from reasonable "
            "interpretation.\n\n"

            "5. If multiple messages refer to the same topic, combine "
            "them into a concise summary instead of discussing each "
            "message in isolation.\n\n"

            "6. Do not require a long or detailed conversation before "
            "acknowledging a topic. A direct message about a topic is "
            "valid evidence that the topic was discussed.\n\n"

            "7. Do not invent names, dates, events, motivations, "
            "or statements that are not supported by the context.\n\n"

            "8. Do not assume that every message in an expanded "
            "chronological section is relevant. Focus on messages "
            "that actually help answer the question.\n\n"

            "9. If the evidence supports only a limited conclusion, "
            "give that limited conclusion rather than saying there "
            "was no discussion.\n\n"

            "10. If the context genuinely contains no useful evidence "
            "for the question, say that the available context is "
            "insufficient.\n\n"

            "CONTEXT:\n"
            f"{context}\n\n"

            f"USER QUESTION:\n"
            f"{state['original_question']}\n\n"

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
# Groundedness check
# =====================================================================

def check_grounded(
    state: GraphState,
) -> GraphState:
    """
    Check whether the generated answer is supported by the retrieved
    context.
    """

    documents = state["documents"]

    if not documents:

        return {
            **state,
            "grounded": False,
        }

    context = "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )

    answer = state["generation"]

    prompt = (
        "You are a factual groundedness checker.\n\n"

        "Determine whether the answer is supported by the supplied "
        "conversation context.\n\n"

        "Answer with exactly one word:\n"
        "'yes' = the answer is supported by the context\n"
        "'no' = the answer contains unsupported or invented claims\n\n"

        f"CONTEXT:\n"
        f"{context}\n\n"

        f"ANSWER:\n"
        f"{answer}\n\n"

        "Decision:"
    )

    try:

        response = llm.invoke(
            prompt
        ).content.strip().lower()

        grounded = response.startswith(
            "yes"
        )

    except Exception:

        grounded = False

    return {
        **state,
        "grounded": grounded,
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
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    graph = build_graph()

    initial_state: GraphState = {

        "question":
            question.strip(),

        "original_question":
            question.strip(),

        "documents":
            [],

        "generation":
            "",

        "rewrite_count":
            0,

        "grounded":
            False,

        "df":
            df,

        "scope":
            "",

        "primary_intent":
            "",

        "secondary_intents":
            [],
    }

    result = graph.invoke(
        initial_state
    )

    return {

        "answer":
            result["generation"],

        "grounded":
            result["grounded"],

        "sources":
            [
                document.metadata
                for document
                in result["documents"]
            ],

        "rewrites_used":
            result["rewrite_count"],

        "scope":
            result["scope"],
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

