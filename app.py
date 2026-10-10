import os
import tempfile
import time
import hashlib
from typing import Any, Optional, Tuple, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from parser import parse_whatsapp_export
from enrich import enrich_messages, to_dataframe
from chat_database import save_messages
from vectorstore import build_vectorstore
from analytics import (
    sender_summary,
    weekly_trend,
    engagement_balance,
    sender_word_frequencies,
)
from rag_graph import ask


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Chat Pattern Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(16,185,129,0.16),
                rgba(59,130,246,0.12)
            );
        border: 1px solid rgba(128,128,128,0.18);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.72;
        max-width: 900px;
        line-height: 1.6;
    }

    /* ---------- KPI CARDS ---------- */

    .metric-card {
        padding: 1.25rem 1.4rem;
        border-radius: 17px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.045);
        min-height: 125px;
    }

    .metric-label {
        font-size: 0.82rem;
        opacity: 0.65;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 750;
    }

    .metric-description {
        font-size: 0.75rem;
        opacity: 0.55;
        margin-top: 0.25rem;
    }

    /* ---------- ANSWER ---------- */

    .answer-card {
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.18);
        background: rgba(128,128,128,0.045);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .answer-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.55;
        margin-bottom: 0.6rem;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        padding: 1rem 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.15);
        background: rgba(128,128,128,0.035);
    }

    /* ---------- QUESTION CHIPS ---------- */

    .question-chip {
        display: inline-block;
        padding: 0.45rem 0.75rem;
        margin: 0.25rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,0.2);
        background: rgba(128,128,128,0.05);
        font-size: 0.82rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.12);
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💬 Chat Pattern Analyzer</div>
        <div class="hero-subtitle">
            Explore conversation topics, messaging behavior, engagement patterns,
            sentiment trends and semantic questions from your WhatsApp export —
            privately and locally using Ollama.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "🔒 Your chat stays on this machine. "
    "AI processing is performed locally through Ollama."
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "answer_time" not in st.session_state:
    st.session_state.answer_time = None

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Analyzer")

    uploaded = st.file_uploader(
        "Upload WhatsApp export",
        type=["txt"],
        help="WhatsApp → Chat → Export chat → Without media",
    )

    st.divider()

    st.markdown("### 🧠 Local AI")

    st.success("Ollama • Local")

    st.caption("LLM: llama3.2")
    st.caption("Embeddings: mxbai-embed-large")

    st.divider()

    st.markdown("### 🔐 Privacy")

    st.caption(
        "Your conversation is processed locally. "
        "No chat data needs to leave your machine."
    )

    st.divider()

    st.caption("Chat Pattern Analyzer")
    st.caption("Pattern analysis ≠ psychological diagnosis")


# ============================================================
# INGESTION HELPERS
# ============================================================

def ingest_chat_file(
    file_path: str,
    database_path: str = "./chat_data.db",
    persist_dir: str = "./chroma_db",
    save_fn=save_messages,
    vectorstore_fn=build_vectorstore,
):
    """
    Parse raw WhatsApp export, enrich records, persist to SQLite,
    and build Chroma vector store.
    """
    messages = parse_whatsapp_export(file_path)
    enriched = enrich_messages(messages)
    df = to_dataframe(enriched)

    if len(messages) > 0 and not df.empty:
        save_fn(enriched, database_path=database_path, reset=True)
        vectorstore_fn(enriched, persist_dir=persist_dir)

    return messages, enriched, df


def handle_upload(
    file_bytes: bytes,
    session_state: Any,
    database_path: str = "./chat_data.db",
    persist_dir: str = "./chroma_db",
    save_fn=save_messages,
    vectorstore_fn=build_vectorstore,
    status_callback=None,
) -> Tuple[bool, Optional[str], Optional[pd.DataFrame], int]:
    """
    Coordinate upload lifecycle with content-hash fingerprinting:
    - Avoids re-processing identical file contents on rerun.
    - Resets active session state before processing new file to prevent
      stale data leakage.
    - Only updates processed_file_hash when ingestion succeeds.
    - Keeps session_state blocked if ingestion fails, enabling retry.
    """
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # If already successfully processed this exact file content, do not rebuild
    if session_state.get("processed_file_hash") == file_hash and session_state.get("df") is not None:
        return True, "already_processed", session_state.get("df"), len(session_state.get("df"))

    # A new or retried file has arrived: clear active session data immediately
    # so stale data is never exposed while processing or on failure
    session_state["df"] = None
    session_state["vectorstore_ready"] = False
    session_state["last_result"] = None
    session_state["last_question"] = ""

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt"
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if status_callback:
            status_callback("Processing WhatsApp export and updating search indexes...")

        messages, enriched, df = ingest_chat_file(
            tmp_path,
            database_path=database_path,
            persist_dir=persist_dir,
            save_fn=save_fn,
            vectorstore_fn=vectorstore_fn,
        )

        if len(messages) == 0:
            return False, "No messages could be parsed from this file.", None, 0

        if df.empty or "sender" not in df.columns:
            return False, "No text messages were found in the export.", None, 0

        if df["sender"].nunique() < 1:
            return False, "No identifiable participants were found.", None, 0

        # Ingestion succeeded: update state and fingerprint
        session_state["df"] = df
        session_state["vectorstore_ready"] = True
        session_state["processed_file_hash"] = file_hash

        return True, None, df, len(df)

    except Exception as e:
        session_state["df"] = None
        session_state["vectorstore_ready"] = False
        # Note: session_state["processed_file_hash"] is NOT updated, enabling retry
        return False, str(e), None, 0

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# PROCESS FILE
# ============================================================

if uploaded is not None:

    file_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.processed_file_hash != file_hash:

        start_processing = time.perf_counter()

        with st.status(
            "Analyzing your conversation...",
            expanded=True
        ) as status:

            success, error_msg, df, msg_count = handle_upload(
                file_bytes=file_bytes,
                session_state=st.session_state,
                status_callback=lambda msg: st.write(f"📄 {msg}"),
            )

            processing_time = (
                time.perf_counter() - start_processing
            )

            if success:

                status.update(
                    label="Analysis ready",
                    state="complete",
                )

                st.success(
                    f"✨ Processed {msg_count:,} messages from "
                    f"{df['sender'].nunique()} participants "
                    f"in {processing_time:.2f}s."
                )

            else:

                status.update(
                    label="Analysis failed",
                    state="error",
                )

                st.error(
                    f"Failed to process and index conversation: {error_msg}"
                )

else:

    # If the user cleared the file uploader, reset the session state
    if st.session_state.processed_file_hash is not None:
        st.session_state.df = None
        st.session_state.vectorstore_ready = False
        st.session_state.processed_file_hash = None
        st.session_state.last_result = None
        st.session_state.last_question = ""


# ============================================================
# MAIN APPLICATION
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    total_messages = len(df)
    participants = df["sender"].nunique()

    start_date = None
    end_date = None

    if "timestamp" in df.columns:

        try:

            start_date = pd.to_datetime(
                df["timestamp"]
            ).min()

            end_date = pd.to_datetime(
                df["timestamp"]
            ).max()

        except Exception:
            pass

    duration_text = "Unknown"

    if start_date is not None and end_date is not None:

        days = max(
            1,
            (end_date - start_date).days
        )

        if days >= 365:

            duration_text = f"{days / 365:.1f} years"

        elif days >= 30:

            duration_text = f"{days / 30:.1f} months"

        else:

            duration_text = f"{days} days"

    # ========================================================
    # KPI ROW
    # ========================================================

    st.markdown("### 📌 Conversation Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MESSAGES</div>
                <div class="metric-value">{total_messages:,}</div>
                <div class="metric-description">
                    Total messages analyzed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">PARTICIPANTS</div>
                <div class="metric-value">{participants}</div>
                <div class="metric-description">
                    Unique senders
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TIME SPAN</div>
                <div class="metric-value">{duration_text}</div>
                <div class="metric-description">
                    Conversation history
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        avg_daily = (
            total_messages / max(
                1,
                (end_date - start_date).days
            )
            if start_date is not None
            and end_date is not None
            else 0
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AVG / DAY</div>
                <div class="metric-value">{avg_daily:.1f}</div>
                <div class="metric-description">
                    Messages per day
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">AI STATUS</div>
                <div class="metric-value">● Local</div>
                <div class="metric-description">
                    Ollama powered
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2 = st.tabs(
        [
            "💬 Ask AI",
            "📊 Conversation Insights",
        ]
    )

    # ========================================================
    # TAB 1 — Q&A
    # ========================================================

    with tab1:

        st.markdown("### Ask anything about the conversation")

        st.caption(
            "Questions are answered using retrieved conversation context "
            "and the local language model."
        )

        # Example questions

        st.markdown("#### 💡 Try asking")

        examples = [
            "What are the main topics we discuss?",
            "How has our conversation changed over time?",
            "Who starts conversations more often?",
            "What topics appear most frequently?",
            "How quickly does each person reply?",
            "When are we most active?",
        ]

        cols = st.columns(3)

        for i, example in enumerate(examples):

            with cols[i % 3]:

                if st.button(
                    example,
                    key=f"example_{i}",
                    use_container_width=True,
                ):

                    st.session_state.last_question = example

        question = st.text_area(
            "Question",
            value=st.session_state.last_question,
            placeholder=(
                "Example: How did our communication pattern "
                "change over the last month?"
            ),
            height=90,
        )

        ask_button = st.button(
            "✨ Analyze Conversation",
            type="primary",
            use_container_width=True,
        )

        if ask_button and question.strip():

            question = question.strip()

            st.session_state.last_question = question

            start_time = time.perf_counter()

            with st.status(
                "Thinking...",
                expanded=False
            ) as status:

                result = ask(
                    question,
                    df=df
                )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                status.update(
                    label=f"Completed in {elapsed:.2f}s",
                    state="complete",
                )

            st.session_state.last_result = result
            st.session_state.answer_time = elapsed

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = st.session_state.last_result

        if result is not None:

            st.markdown(
                """
                <div class="answer-card">
                    <div class="answer-title">
                        AI Analysis
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                result["answer"]
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            # Answer metadata

            m1, m2, m3 = st.columns(3)

            with m1:

                scope_label = (
                    "Whole conversation"
                    if result["scope"] == "broad"
                    else "Retrieved excerpts"
                )

                st.metric(
                    "Analysis scope",
                    scope_label,
                )

            with m2:

                grounded = result["grounded"]

                st.metric(
                    "Grounding",
                    "Verified" if grounded else "Uncertain",
                )

            with m3:

                elapsed = st.session_state.answer_time

                st.metric(
                    "⏱ Answer time",
                    f"{elapsed:.2f}s",
                )

            # Timing breakdown placeholder

            st.caption(
                f"⚡ The complete question-answer cycle took "
                f"**{st.session_state.answer_time:.2f} seconds** "
                f"including retrieval and local LLM generation."
            )

            # Sources

            with st.expander(
                f"🔎 Sources used ({len(result['sources'])})"
            ):

                for i, source in enumerate(
                    result["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Source {i}**"
                    )

                    st.write(source)

                    if i < len(result["sources"]):
                        st.divider()

        else:

            st.info(
                "Ask a question above to generate an AI-powered "
                "conversation analysis."
            )

    # ========================================================
    # TAB 2 — DASHBOARD
    # ========================================================

    with tab2:

        st.markdown("### 📊 Conversation Intelligence")

        st.warning(
            "These insights describe observable messaging patterns. "
            "They are not psychological, medical, relationship-trust, "
            "or personality diagnoses."
        )

        # ----------------------------------------------------
        # SENDER SUMMARY
        # ----------------------------------------------------

        summary = sender_summary(df)

        st.markdown("### 👥 Participant Overview")

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        st.write("")

        # ----------------------------------------------------
        # WEEKLY TRENDS
        # ----------------------------------------------------

        trend = weekly_trend(df)

        st.markdown("### 📈 Communication Over Time")

        c1, c2 = st.columns(2)

        with c1:

            fig1 = px.line(
                trend,
                x="week",
                y="avg_sentiment",
                markers=True,
                title="Average Sentiment by Week",
            )

            fig1.update_layout(
                height=380,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
                hovermode="x unified",
            )

            st.plotly_chart(
                fig1,
                use_container_width=True,
            )

        with c2:

            fig2 = px.line(
                trend,
                x="week",
                y="avg_reply_time_minutes",
                markers=True,
                title="Average Reply Time",
            )

            fig2.update_layout(
                height=380,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
                hovermode="x unified",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # ENGAGEMENT
        # ----------------------------------------------------

        balance = engagement_balance(df)

        st.markdown("### ⚖️ Engagement Balance")

        c3, c4 = st.columns(2)

        with c3:

            share_df = pd.DataFrame(
                balance["message_share"].items(),
                columns=[
                    "sender",
                    "share"
                ],
            )

            fig3 = px.pie(
                share_df,
                names="sender",
                values="share",
                hole=0.45,
                title="Message Volume Share",
            )

            fig3.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig3,
                use_container_width=True,
            )

        with c4:

            init_df = pd.DataFrame(
                balance[
                    "conversation_initiations"
                ].items(),
                columns=[
                    "sender",
                    "initiations"
                ],
            )

            fig4 = px.bar(
                init_df,
                x="sender",
                y="initiations",
                text="initiations",
                title="Who Starts Conversations?",
            )

            fig4.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig4,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # WORD ANALYSIS
        # ----------------------------------------------------

        st.markdown("### 🗣️ Language & Word Usage")

        st.caption(
            "Frequently used words by each participant after "
            "removing common stop words and WhatsApp artifacts."
        )

        word_freqs = sender_word_frequencies(
            df,
            top_n=15,
        )

        word_cols = st.columns(2)

        for i, (sender, freqs) in enumerate(
            word_freqs.items()
        ):

            with word_cols[
                i % 2
            ]:

                st.markdown(
                    f"#### {sender}"
                )

                if freqs:

                    words, counts = zip(*freqs)

                    word_df = pd.DataFrame(
                        {
                            "word": words,
                            "count": counts,
                        }
                    ).sort_values(
                        "count"
                    )

                    fig = px.bar(
                        word_df,
                        x="count",
                        y="word",
                        orientation="h",
                        title="Most used words",
                    )

                    fig.update_layout(
                        height=420,
                        margin=dict(
                            l=20,
                            r=20,
                            t=60,
                            b=20,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                else:

                    st.info(
                        "Not enough text data."
                    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown("")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="info-card">
                <h3>💬 Ask Questions</h3>
                <p>
                    Ask semantic questions about topics,
                    communication patterns and changes over time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">
                <h3>📊 Discover Patterns</h3>
                <p>
                    Explore message volume, reply times,
                    sentiment and conversation initiation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            """
            <div class="info-card">
                <h3>🔒 Stay Private</h3>
                <p>
                    Your WhatsApp export can be analyzed
                    locally using Ollama without sending
                    your conversation to a cloud AI.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.info(
        "👈 Upload a WhatsApp `.txt` export from the sidebar "
        "to begin. Use WhatsApp → Chat → Export chat → Without media."
    )
