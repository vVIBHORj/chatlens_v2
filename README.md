# 💬 ChatLens-RAG

**A local, self-correcting RAG pipeline that turns your WhatsApp chat export into a Q&A assistant and a behavioral analytics dashboard — 100% offline, zero API costs.**

> ⚠️ **This is a pattern-analysis tool, not a psychological, medical, or relationship diagnostic.** It surfaces descriptive statistics (e.g. "average reply time increased over time"), never judgments about character or trust. Only analyze conversations you have the right to — your own chats, or chats where every participant has given explicit consent.

---

## Overview

ChatLens-RAG parses a raw WhatsApp `.txt` export and turns it into two things:

1. **A semantic Q&A interface** — ask natural-language questions about the conversation ("How did our chats change after March?", "Who starts conversations more often?") and get answers grounded in retrieved excerpts rather than free-form hallucination.
2. **A behavioral pattern dashboard** — sentiment trends, reply-time trends, message-share balance, conversation-initiation counts, and per-participant word usage, all computed with deterministic statistics rather than an LLM-invented score.

Everything runs **locally via [Ollama](https://ollama.com)** — no data leaves your machine, no API keys, no per-token billing.

## Why this is more than a "chat with your data" demo

Most retrieval-augmented generation tutorials stop at naive retrieve-then-generate. ChatLens-RAG instead implements:

- **Query routing** — every question is classified into an intent (semantic Q&A, statistics, time analysis, comparison, behavior, sentiment, anomaly detection, context) before any retrieval happens, so broad analytical questions can be routed to whole-conversation statistics instead of a handful of retrieved snippets.
- **Hybrid retrieval** — semantic search over a Chroma vector store is combined with exact/keyword search over a SQLite FTS5 index. Candidates are scored using a blend of semantic similarity, lexical match, query-term coverage, and a multi-source agreement bonus (RRF is used as one signal, not the final ranking).
- **Corrective RAG (CRAG) via LangGraph** — retrieved documents are graded for relevance by the LLM; if nothing relevant comes back, the query is rewritten and retried (bounded, to avoid infinite loops).
- **Chronology-aware context expansion** — once a relevant message is found, its surrounding conversation is expanded using real timestamps (not just neighboring row IDs), respecting natural conversational gaps so unrelated messages from hours later aren't stitched in.
- **Groundedness checking** — after an answer is generated, a second LLM pass verifies the answer is actually supported by the retrieved context and flags it if not.
- **SQLite as the source of truth, Chroma as the semantic index** — the complete enriched conversation always lives in SQLite (with an FTS5 index for keyword search); Chroma only ever holds embeddings used to *locate* relevant regions. This keeps retrieval reproducible and makes exact lookups trivial.
- **Deterministic analytics kept separate from LLM reasoning** — sentiment trends, reply-time trends, and engagement balance are computed with plain pandas over VADER-scored messages, never guessed by an LLM. The LLM is only used for open-ended reasoning over retrieved text and for turning computed numbers into an optional plain-language summary.

## Architecture

```
WhatsApp .txt export
        │
        ▼
   parser.py            → structured (timestamp, sender, message) records
        │
        ▼
   enrich.py             → VADER sentiment, response time, message length,
        │                   question detection, stable message IDs
        ▼
   chat_database.py      → SQLite (source of truth) + FTS5 keyword index
        │
        ▼
   vectorstore.py        → qwen3-embedding:0.6b (via Ollama) → Chroma
                            (message-level + conversation-episode chunks)
        │
        ▼
   query_router.py        → classify question intent (semantic / statistics /
        │                    time / comparison / behavior / sentiment / anomaly)
        ▼
   rag_graph.py (LangGraph — Corrective RAG)

     classify_scope
        /        \
     broad       specific
       │             │
       ▼             ▼
   whole-chat    hybrid_retriever.py (semantic + lexical + RRF + term coverage)
   style profile        │
                        ▼
                   grade_documents
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          relevant            irrelevant
              │                   │
              ▼                   ▼
   context_expander.py       rewrite_query
   (chronology-aware,             │
    gap-bounded expansion)        │
              │                   │
              ▼◄──────────────────┘
           generate
              │
              ▼
        check_grounded
              │
              ▼
             END
        │
        ▼
   app.py (Streamlit UI)
   Tab 1: Ask AI   |   Tab 2: Conversation Insights
```

## Tech stack

| Layer | Tool |
|---|---|
| LLM (reasoning, grading, groundedness) | [Ollama](https://ollama.com) + `qwen2.5:3b` |
| Embeddings | `qwen3-embedding:0.6b` (via Ollama) |
| Orchestration | LangChain + LangGraph |
| Vector store | ChromaDB (local, persisted to disk) |
| Structured storage / keyword search | SQLite + FTS5 |
| Sentiment analysis | VADER (`vaderSentiment`) |
| Analytics | pandas |
| UI | Streamlit + Plotly |

## Getting started

### 1. Install Ollama and pull the models

```bash
ollama pull qwen2.5:3b
ollama pull qwen3-embedding:0.6b
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r Requirements.txt
```

### 3. Export your WhatsApp chat

WhatsApp → open chat → tap contact/group name → **Export Chat** → **Without Media** → save the `.txt` file.

### 4. Run the app

```bash
streamlit run app.py
```

Upload your `.txt` export from the sidebar in the browser tab that opens.

## Project structure

```
.
├── app.py                # Streamlit UI — Ask AI tab + Conversation Insights dashboard
├── parser.py              # WhatsApp .txt export → structured messages
├── enrich.py               # VADER sentiment, reply time, question detection, message IDs
├── chat_database.py         # SQLite storage (source of truth) + FTS5 keyword search
├── vectorstore.py            # Message- and episode-level chunking, embeddings, Chroma store
├── hybrid_retriever.py        # Semantic + lexical hybrid search with weighted scoring
├── context_expander.py         # Chronology-aware, gap-bounded context expansion
├── query_router.py              # Question intent classification
├── rag_graph.py                  # LangGraph Corrective-RAG pipeline
├── analytics.py                   # Deterministic pattern statistics (pandas)
├── sample_chat.txt                 # Synthetic example chat for testing
└── Requirements.txt
```

## Testing individual components

```bash
python parser.py sample_chat.txt        # test parsing
python enrich.py                        # test sentiment/behavioral enrichment
python vectorstore.py                   # test chunking + embedding
python hybrid_test.py                   # test hybrid retrieval
python context_test.py                  # test context expansion
python analytics.py                     # test pattern statistics
python rag_graph.py "your question"     # ask a question (build the vectorstore first)
```

## Privacy

All parsing, embedding, retrieval, and generation happen on your machine through a local Ollama instance — no conversation data is ever sent to a third-party API. Only analyze conversations you're authorized to analyze.

## License

Add your preferred license (e.g. MIT) here.
