"""
vectorstore.py

Dual-level Chroma vector store for WhatsApp conversations.

Architecture:

    WhatsApp messages
            |
            +-----------------------------+
            |                             |
            v                             v
    Individual messages          Conversation episodes/chunks
            |                             |
            v                             v
    whatsapp_messages             whatsapp_chat
            |                             |
            +--------------+--------------+
                           |
                           v
                  Hybrid RAG retrieval

Embedding model:
    qwen3-embedding:0.6b via Ollama

Ollama:
    http://127.0.0.1:11434

Important:
    - Original message IDs are preserved.
    - Individual messages are indexed separately.
    - Conversation chunks are also indexed.
    - SQLite remains the source of truth.
"""

from pathlib import Path
from typing import List, Optional
from collections import defaultdict
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from enrich import EnrichedMessage


# ============================================================================
# CONFIGURATION
# ============================================================================

PERSIST_DIR = "./chroma_db"

# Conversation-level collection
COLLECTION_NAME = "whatsapp_chat"

# Individual-message collection
MESSAGE_COLLECTION_NAME = "whatsapp_messages"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# ---------------------------------------------------------------------------
# Conversation chunk configuration
# ---------------------------------------------------------------------------

MAX_CHARS_PER_CHUNK = 6000
MAX_MESSAGES_PER_CHUNK = 50

# Number of messages from the previous chunk that are repeated.
CHUNK_OVERLAP_MESSAGES = 10

# A gap larger than this starts a new conversation episode.
EPISODE_GAP_MINUTES = 120

# ---------------------------------------------------------------------------
# Embedding batch size
# ---------------------------------------------------------------------------

EMBED_BATCH_SIZE = 50


# ============================================================================
# EMBEDDINGS
# ============================================================================

def get_embeddings() -> OllamaEmbeddings:
    """
    Create the Ollama embedding model.
    """

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def _safe_timestamp(message: EnrichedMessage) -> str:
    """
    Convert timestamp into a readable string.
    """

    if message.timestamp is None:
        return "Unknown time"

    return message.timestamp.strftime("%Y-%m-%d %I:%M %p")


def _calculate_average_sentiment(
    messages: List[EnrichedMessage],
) -> float:
    """
    Calculate average sentiment for a group of messages.
    """

    values = [
        m.sentiment_compound
        for m in messages
        if m.sentiment_compound is not None
    ]

    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


def _get_participants(
    messages: List[EnrichedMessage],
) -> str:
    """
    Return sorted unique participant names.
    """

    participants = {
        m.sender
        for m in messages
        if m.sender
    }

    return ", ".join(sorted(participants))


def _format_message(
    message: EnrichedMessage,
) -> str:
    """
    Format one message for embedding.

    We intentionally include:
        - message ID
        - timestamp
        - sender
        - original message

    The original wording is preserved.
    """

    return (
        f"[message_id={message.id}] "
        f"[{_safe_timestamp(message)}] "
        f"{message.sender}: "
        f"{message.message}"
    )


# ============================================================================
# INDIVIDUAL MESSAGE DOCUMENTS
# ============================================================================

def build_message_documents(
    enriched: List[EnrichedMessage],
) -> List[Document]:
    """
    Create ONE vector document per WhatsApp message.

    This is the fine-grained retrieval layer.

    Example:

        [message_id=123] [2022-09-19 09:27 PM] Vibhor:
        Tum konsi sports loge ...?

    Metadata keeps the original message ID available so that
    SQLite can later expand the surrounding conversation.
    """

    documents = []

    for message in enriched:

        # Skip empty messages.
        if not message.message or not message.message.strip():
            continue

        documents.append(
            Document(
                page_content=_format_message(message),
                metadata={
                    "source": "message",
                    "message_id": int(message.id),
                    "sender": message.sender or "",
                    "timestamp": (
                        message.timestamp.isoformat()
                        if message.timestamp
                        else ""
                    ),
                    "date": (
                        message.timestamp.strftime("%Y-%m-%d")
                        if message.timestamp
                        else ""
                    ),
                    "message_type": message.message_type,
                    "is_question": bool(message.is_question),
                    "sentiment_compound": float(
                        message.sentiment_compound
                    ),
                },
            )
        )

    return documents


# ============================================================================
# CONVERSATION EPISODES
# ============================================================================

def build_conversation_episodes(
    enriched: List[EnrichedMessage],
) -> List[List[EnrichedMessage]]:
    """
    Group messages into conversation episodes.

    A new episode starts when the gap between two messages is greater
    than EPISODE_GAP_MINUTES.

    This is better than grouping strictly by calendar day because
    a conversation can cross midnight.
    """

    if not enriched:
        return []

    # Ensure chronological order.
    messages = sorted(
        enriched,
        key=lambda m: (
            m.timestamp or datetime.min,
            m.id,
        ),
    )

    episodes = []
    current_episode = [messages[0]]

    for message in messages[1:]:

        previous = current_episode[-1]

        should_start_new_episode = False

        if (
            previous.timestamp is not None
            and message.timestamp is not None
        ):
            gap_minutes = (
                message.timestamp - previous.timestamp
            ).total_seconds() / 60.0

            if gap_minutes > EPISODE_GAP_MINUTES:
                should_start_new_episode = True

        if should_start_new_episode:
            episodes.append(current_episode)
            current_episode = [message]
        else:
            current_episode.append(message)

    if current_episode:
        episodes.append(current_episode)

    return episodes


# ============================================================================
# CONVERSATION CHUNKS
# ============================================================================

def _create_chunk_document(
    messages: List[EnrichedMessage],
    episode_id: int,
    chunk_index: int,
    global_chunk_index: int,
) -> Document:
    """
    Convert a group of messages into one conversation-level Document.
    """

    transcript = "\n".join(
        _format_message(message)
        for message in messages
    )

    start_message = messages[0]
    end_message = messages[-1]

    start_timestamp = (
        start_message.timestamp.isoformat()
        if start_message.timestamp
        else ""
    )

    end_timestamp = (
        end_message.timestamp.isoformat()
        if end_message.timestamp
        else ""
    )

    date = (
        start_message.timestamp.strftime("%Y-%m-%d")
        if start_message.timestamp
        else ""
    )

    return Document(
        page_content=transcript,
        metadata={
            "source": "conversation_chunk",

            "episode_id": int(episode_id),

            "chunk_index": int(chunk_index),

            "global_chunk_index": int(
                global_chunk_index
            ),

            "start_message_id": int(
                start_message.id
            ),

            "end_message_id": int(
                end_message.id
            ),

            "start_timestamp": start_timestamp,

            "end_timestamp": end_timestamp,

            "date": date,

            "num_messages": len(messages),

            "participants": _get_participants(
                messages
            ),

            "avg_sentiment": _calculate_average_sentiment(
                messages
            ),

            "has_questions": any(
                bool(m.is_question)
                for m in messages
            ),
        },
    )


def build_conversation_chunks(
    enriched: List[EnrichedMessage],
) -> List[Document]:
    """
    Build conversation-level chunks.

    Chunks:
        - respect conversation episodes
        - have a character limit
        - have a message-count limit
        - overlap by CHUNK_OVERLAP_MESSAGES

    This preserves local conversational context.
    """

    episodes = build_conversation_episodes(enriched)

    documents = []

    global_chunk_index = 0

    for episode_id, episode in enumerate(episodes):

        if not episode:
            continue

        start = 0
        chunk_index = 0

        while start < len(episode):

            chunk_messages = []
            char_count = 0

            current_index = start

            while current_index < len(episode):

                message = episode[current_index]

                formatted = _format_message(message)

                additional_chars = len(formatted)

                # Stop when the chunk would become too large.
                if (
                    chunk_messages
                    and (
                        char_count
                        + additional_chars
                        + 1
                        > MAX_CHARS_PER_CHUNK
                    )
                ):
                    break

                # Stop at maximum number of messages.
                if (
                    len(chunk_messages)
                    >= MAX_MESSAGES_PER_CHUNK
                ):
                    break

                chunk_messages.append(message)

                char_count += (
                    additional_chars + 1
                )

                current_index += 1

            if not chunk_messages:
                # Safety fallback.
                chunk_messages = [
                    episode[current_index]
                ]
                current_index += 1

            document = _create_chunk_document(
                messages=chunk_messages,
                episode_id=episode_id,
                chunk_index=chunk_index,
                global_chunk_index=global_chunk_index,
            )

            documents.append(document)

            global_chunk_index += 1
            chunk_index += 1

            # ------------------------------------------------------------
            # Overlap
            # ------------------------------------------------------------

            next_start = (
                current_index
                - CHUNK_OVERLAP_MESSAGES
            )

            # Never move backwards.
            if next_start <= start:
                next_start = current_index

            start = next_start

    return documents


# ============================================================================
# CHROMA BUILDING
# ============================================================================

def _delete_existing_collection(
    persist_dir: str,
    collection_name: str,
) -> None:
    """
    Delete one Chroma collection if it exists.
    """

    try:

        embeddings = get_embeddings()

        existing = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

        existing.delete_collection()

        print(
            f"    ✓ Deleted old collection: "
            f"{collection_name}"
        )

    except Exception:
        # Collection may not exist.
        pass


def _add_documents_in_batches(
    vectorstore: Chroma,
    documents: List[Document],
    id_prefix: str,
) -> None:
    """
    Add documents to Chroma in batches.

    Chroma/LangChain generates embeddings through the
    configured embedding function.
    """

    total = len(documents)

    if total == 0:
        return

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE,
    ):

        batch = documents[
            start:start + EMBED_BATCH_SIZE
        ]

        ids = [
            f"{id_prefix}_{start + i}"
            for i in range(len(batch))
        ]

        batch_number = (
            start // EMBED_BATCH_SIZE
        ) + 1

        total_batches = (
            total + EMBED_BATCH_SIZE - 1
        ) // EMBED_BATCH_SIZE

        print(
            f"    Embedding batch "
            f"{batch_number}/{total_batches} "
            f"({len(batch)} documents)"
        )

        vectorstore.add_documents(
            documents=batch,
            ids=ids,
        )


# ============================================================================
# BUILD COMPLETE VECTOR STORE
# ============================================================================

def build_vectorstore(
    enriched: List[EnrichedMessage],
    persist_dir: str = PERSIST_DIR,
) -> Chroma:
    """
    Build BOTH vector collections.

    Collections:

        whatsapp_messages
            Fine-grained message retrieval

        whatsapp_chat
            Conversation-level retrieval
    """

    persist_path = Path(persist_dir)
    persist_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Building ChatLens-RAG Dual-Level Vector Store")
    print("=" * 70)

    print(
        f"\nEmbedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Messages received: {len(enriched):,}"
    )

    # ------------------------------------------------------------------
    # Build documents
    # ------------------------------------------------------------------

    print("\nBuilding individual message documents...")

    message_documents = build_message_documents(
        enriched
    )

    print(
        f"✓ Created "
        f"{len(message_documents):,} message documents"
    )

    print("\nBuilding conversation chunks...")

    conversation_documents = (
        build_conversation_chunks(enriched)
    )

    print(
        f"✓ Created "
        f"{len(conversation_documents):,} conversation chunks"
    )

    # ------------------------------------------------------------------
    # Remove previous collections
    # ------------------------------------------------------------------

    print("\nRemoving old Chroma collections...")

    _delete_existing_collection(
        persist_dir,
        MESSAGE_COLLECTION_NAME,
    )

    _delete_existing_collection(
        persist_dir,
        COLLECTION_NAME,
    )

    # ------------------------------------------------------------------
    # Embedding function
    # ------------------------------------------------------------------

    embeddings = get_embeddings()

    # ------------------------------------------------------------------
    # Message collection
    # ------------------------------------------------------------------

    print(
        "\nCreating individual-message collection..."
    )

    message_store = Chroma(
        collection_name=MESSAGE_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    _add_documents_in_batches(
        vectorstore=message_store,
        documents=message_documents,
        id_prefix="whatsapp_message",
    )

    # ------------------------------------------------------------------
    # Conversation collection
    # ------------------------------------------------------------------

    print(
        "\nCreating conversation-level collection..."
    )

    conversation_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    _add_documents_in_batches(
        vectorstore=conversation_store,
        documents=conversation_documents,
        id_prefix="whatsapp_chunk",
    )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    message_count = (
        message_store._collection.count()
    )

    chunk_count = (
        conversation_store._collection.count()
    )

    print("\n" + "=" * 70)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\n✓ Individual messages: "
        f"{message_count:,}"
    )

    print(
        f"✓ Conversation chunks: "
        f"{chunk_count:,}"
    )

    print(
        f"✓ Persist directory: "
        f"{persist_path.resolve()}"
    )

    print(
        "\nCollections:"
    )

    print(
        f"  • {MESSAGE_COLLECTION_NAME}"
    )

    print(
        f"  • {COLLECTION_NAME}"
    )

    return conversation_store


# ============================================================================
# LOADERS
# ============================================================================

def load_vectorstore(
    persist_dir: str = PERSIST_DIR,
) -> Chroma:
    """
    Load the conversation-level Chroma collection.

    This keeps backward compatibility with the existing
    rag_graph.py.
    """

    return load_chunk_vectorstore(
        persist_dir=persist_dir
    )


def load_chunk_vectorstore(
    persist_dir: str = PERSIST_DIR,
) -> Chroma:
    """
    Load conversation-level vector store.
    """

    persist_path = Path(persist_dir)

    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma database not found at: "
            f"{persist_path}"
        )

    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def load_message_vectorstore(
    persist_dir: str = PERSIST_DIR,
) -> Chroma:
    """
    Load individual-message vector store.
    """

    persist_path = Path(persist_dir)

    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma database not found at: "
            f"{persist_path}"
        )

    embeddings = get_embeddings()

    return Chroma(
        collection_name=MESSAGE_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


# ============================================================================
# STATS
# ============================================================================

def get_vectorstore_stats(
    persist_dir: str = PERSIST_DIR,
) -> dict:
    """
    Return counts for both collections.
    """

    chunk_store = load_chunk_vectorstore(
        persist_dir
    )

    message_store = load_message_vectorstore(
        persist_dir
    )

    return {
        "conversation_chunks": (
            chunk_store._collection.count()
        ),
        "individual_messages": (
            message_store._collection.count()
        ),
    }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":

    from parser import parse_whatsapp_export
    from enrich import enrich_messages

    print("=" * 70)
    print("ChatLens-RAG Vector Store Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    print("\nParsing WhatsApp export...")

    messages = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"Parsed {len(messages):,} messages."
    )

    # ------------------------------------------------------------------
    # Enrich
    # ------------------------------------------------------------------

    print("\nEnriching messages...")

    enriched = enrich_messages(
        messages
    )

    print(
        f"Enriched {len(enriched):,} messages."
    )

    # ------------------------------------------------------------------
    # Validate IDs
    # ------------------------------------------------------------------

    ids = [
        message.id
        for message in enriched
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate message IDs detected."
        )

    print(
        f"✓ Message IDs are unique "
        f"({len(ids):,})"
    )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    build_vectorstore(
        enriched=enriched
    )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    stats = get_vectorstore_stats()

    print("\nFinal statistics:")

    print(
        f"  Individual messages: "
        f"{stats['individual_messages']:,}"
    )

    print(
        f"  Conversation chunks: "
        f"{stats['conversation_chunks']:,}"
    )

    print("\n✓ Vector store test completed.")