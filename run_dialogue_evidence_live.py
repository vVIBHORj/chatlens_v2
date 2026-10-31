"""
run_dialogue_evidence_live.py
==============================
Live Evaluation Runner for ChatLens-RAG: Default Baseline vs Opt-In Dialogue-Evidence Pipeline.

Executes a controlled, paired comparison across the 12 canonical evaluation cases:
  - Baseline: Default production retrieval path with grade_documents filter.
  - Treatment: Opt-in dialogue-evidence pipeline (assemble_and_rank_evidence with RRF reranking).

Safety Guarantees:
  - Fail-closed call budget counter (planned nominal: 72, absolute hard ceiling: 84).
  - Strict storage isolation: zero access/modification to production chat_data.db or chroma_db.
  - Fail-closed isolation interceptor raising IsolationBreachError on production access.
  - Teardown guaranteed in finally blocks.
  - Structured output saved to eval_results/ (git-ignored).
"""

import argparse
import copy
import inspect
import json
import os
import re
import shutil
import sqlite3
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

import chat_database
import hybrid_retriever
import rag_graph
import vectorstore
from chat_database import create_database, get_connection, save_messages
from enrich import enrich_messages, to_dataframe
from parser import parse_whatsapp_export
from vectorstore import build_vectorstore, get_embeddings
from evaluation_scoring import (
    classify_case_outcome,
    compute_citation_validity,
    evaluate_citation_attribution,
    evaluate_exact_facts_and_qualifiers,
)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
NOMINAL_BUDGET = 72
HARD_CEILING_CALLS = 84
DEFAULT_TIMEOUT_SECONDS = 45.0


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class IsolationBreachError(RuntimeError):
    """Raised when an operation attempts to access storage outside the sandbox."""
    pass


class InferenceBudgetExceededError(RuntimeError):
    """Raised when planned or attempted model invocations exceed the call ceiling."""
    pass


class LiveConfirmationRequiredError(RuntimeError):
    """Raised when live execution is attempted without explicit user confirmation."""
    pass


# ============================================================================
# EVALUATION CASE DEFINITIONS
# ============================================================================

@dataclass
class EvalCase:
    case_id: str
    category: str
    query: str
    target_id: int
    target_text: str
    target_sender: str
    target_timestamp: str
    preceding_messages: List[Dict[str, Any]]
    expected_answerable: bool
    expected_supporting_ids: List[int]
    expected_answer_keywords: List[str]
    description: str
    support_semantics: str = "joint"


EVAL_CASES: List[EvalCase] = [
    # 1. Terse location reply depending on question
    EvalCase(
        case_id="CASE-01-Terse-Location",
        category="Terse Context-Dependent",
        query="Where was the backup SSL certificate placed?",
        target_id=102,
        target_text="In Vault 9.",
        target_sender="Dev_Sam",
        target_timestamp="2026-10-14T09:11:00",
        preceding_messages=[
            {"id": 101, "timestamp": "2026-10-14T09:10:00", "sender": "Dev_Alex", "message": "Where did we store the backup SSL certificate?"}
        ],
        expected_answerable=True,
        expected_supporting_ids=[102],
        expected_answer_keywords=["Vault 9"],
        description="Terse location fragment whose entity depends on preceding question.",
    ),
    # 2. Short time reply
    EvalCase(
        case_id="CASE-02-Short-Time",
        category="Terse Numerical/Time",
        query="What time is the production deployment window scheduled for?",
        target_id=104,
        target_text="9pm.",
        target_sender="Dev_Lead",
        target_timestamp="2026-10-14T14:31:00",
        preceding_messages=[
            {"id": 103, "timestamp": "2026-10-14T14:30:00", "sender": "Dev_Ops", "message": "When are we kicking off the production deployment window tonight?"}
        ],
        expected_answerable=True,
        expected_supporting_ids=[104],
        expected_answer_keywords=["9pm", "9:00", "21:00"],
        description="Short time fragment answering deployment window start.",
    ),
    # 3. Pronoun-based reply
    EvalCase(
        case_id="CASE-03-Pronoun-Anaphora",
        category="Pronoun Resolution",
        query="Where can I find the incident root cause report?",
        target_id=106,
        target_text="It is in the second folder.",
        target_sender="Dev_Maya",
        target_timestamp="2026-10-14T11:05:00",
        preceding_messages=[
            {"id": 105, "timestamp": "2026-10-14T11:04:00", "sender": "Dev_Ken", "message": "Has anyone seen the incident root cause report?"}
        ],
        expected_answerable=True,
        expected_supporting_ids=[106],
        expected_answer_keywords=["second folder"],
        description="Candidate uses pronoun 'It' referring to incident root cause report.",
    ),
    # 4. Confirmation reply
    EvalCase(
        case_id="CASE-04-Yes-No-Confirmation",
        category="Terse Confirmation",
        query="Did the database migration script succeed on staging?",
        target_id=108,
        target_text="Yes, completely clean.",
        target_sender="Dev_Sarah",
        target_timestamp="2026-10-14T16:20:00",
        preceding_messages=[
            {"id": 107, "timestamp": "2026-10-14T16:19:00", "sender": "Dev_Chris", "message": "Did the database migration script finish successfully on staging?"}
        ],
        expected_answerable=True,
        expected_supporting_ids=[108],
        expected_answer_keywords=["yes", "clean", "succeed"],
        description="Yes/no confirmation whose meaning depends on the question.",
    ),
    # 5. Misleading context distractor (Unanswerable query)
    EvalCase(
        case_id="CASE-05-Misleading-Context",
        category="Misleading Context Distractor",
        query="Where is the staging API key stored?",
        target_id=110,
        target_text="On the kitchen counter.",
        target_sender="Dev_Sam",
        target_timestamp="2026-10-14T12:05:00",
        preceding_messages=[
            {"id": 109, "timestamp": "2026-10-14T12:04:00", "sender": "Dev_Alex", "message": "Where did you leave the office cafeteria keys?"}
        ],
        expected_answerable=False,
        expected_supporting_ids=[],
        expected_answer_keywords=[],
        description="Short reply answering about cafeteria keys, NOT the staging API key. Must abstain.",
    ),
    # 6. Self-contained relevant message
    EvalCase(
        case_id="CASE-06-Self-Contained-Relevant",
        category="Self-Contained Positive",
        query="What is the primary IP address for the Redis cache cluster?",
        target_id=112,
        target_text="The primary IP address for the Redis cache cluster is 10.0.4.15.",
        target_sender="Dev_Alex",
        target_timestamp="2026-10-14T08:30:00",
        preceding_messages=[
            {"id": 111, "timestamp": "2026-10-14T08:29:00", "sender": "Dev_Ops", "message": "Good morning team, checking infrastructure status."}
        ],
        expected_answerable=True,
        expected_supporting_ids=[112],
        expected_answer_keywords=["10.0.4.15"],
        description="Complete factual statement that is relevant standalone.",
    ),
    # 7. Self-contained irrelevant message
    EvalCase(
        case_id="CASE-07-Self-Contained-Irrelevant",
        category="Self-Contained Negative",
        query="What is the primary IP address for the Redis cache cluster?",
        target_id=114,
        target_text="I think ordering pizza sounds like a great choice for the entire team today.",
        target_sender="Dev_Dana",
        target_timestamp="2026-10-14T12:15:00",
        preceding_messages=[
            {"id": 113, "timestamp": "2026-10-14T12:14:00", "sender": "Dev_Chris", "message": "Anyone ordering lunch today?"}
        ],
        expected_answerable=False,
        expected_supporting_ids=[],
        expected_answer_keywords=[],
        description="Completely unrelated conversation about lunch. Must abstain.",
    ),
    # 8. Unrelated candidate with query-matching preceding message
    EvalCase(
        case_id="CASE-08-Unrelated-Candidate-Matching-Context",
        category="Context Distractor Invariance",
        query="Who approved the firewall port change?",
        target_id=116,
        target_text="I think the weather is going to be stormy this evening.",
        target_sender="Dev_Dana",
        target_timestamp="2026-10-14T17:05:00",
        preceding_messages=[
            {"id": 115, "timestamp": "2026-10-14T17:04:00", "sender": "Dev_Security", "message": "Who approved the firewall port change yesterday?"}
        ],
        expected_answerable=False,
        expected_supporting_ids=[],
        expected_answer_keywords=[],
        description="Preceding message matches query, but candidate discusses weather; no approval given. Must abstain.",
    ),
    # 9. Multi-message conversational exchange
    EvalCase(
        case_id="CASE-09-Multi-Message-Exchange",
        category="Multi-Message Conversation",
        query="Which server rack is experiencing power fluctuation?",
        target_id=119,
        target_text="Confirmed, heavy voltage drops.",
        target_sender="Dev_Facilities",
        target_timestamp="2026-10-14T13:42:00",
        preceding_messages=[
            {"id": 117, "timestamp": "2026-10-14T13:40:00", "sender": "Dev_Monitor", "message": "Rack C4 power sensor is throwing warnings."},
            {"id": 118, "timestamp": "2026-10-14T13:41:00", "sender": "Dev_Ops", "message": "Can anyone confirm on site?"}
        ],
        expected_answerable=True,
        expected_supporting_ids=[119, 117],
        expected_answer_keywords=["Rack C4", "C4"],
        description="Two preceding messages identify the entity (Rack C4) for candidate confirmation.",
        support_semantics="joint",
    ),
    # 10. Ambiguous / unresolvable context
    EvalCase(
        case_id="CASE-010-Ambiguous-Context-Abstention",
        category="Ambiguous / Abstention",
        query="Did the Kubernetes pod autoscaler trigger?",
        target_id=121,
        target_text="Maybe.",
        target_sender="Dev_Sam",
        target_timestamp="2026-10-14T15:10:00",
        preceding_messages=[
            {"id": 120, "timestamp": "2026-10-14T15:09:00", "sender": "Dev_Alex", "message": "Checking general system alerts."}
        ],
        expected_answerable=False,
        expected_supporting_ids=[],
        expected_answer_keywords=[],
        description="Candidate is 'Maybe.' and context does not resolve pod autoscaler query. Must abstain.",
    ),
    # 11. Similar but different entity
    EvalCase(
        case_id="CASE-011-Similar-Entity-Negative",
        category="Fine-Grained Entity Discrimination",
        query="Where is the production JWT secret stored?",
        target_id=123,
        target_text="In secrets.dev.json.",
        target_sender="Dev_Auth",
        target_timestamp="2026-10-14T11:22:00",
        preceding_messages=[
            {"id": 122, "timestamp": "2026-10-14T11:21:00", "sender": "Dev_Test", "message": "Where did we save the dev environment test JWT token?"}
        ],
        expected_answerable=False,
        expected_supporting_ids=[],
        expected_answer_keywords=[],
        description="Candidate discusses dev test token, NOT production JWT secret. Must abstain.",
    ),
    # 12. Ordinary length bypass
    EvalCase(
        case_id="CASE-012-Ordinary-Length-Bypass",
        category="Length Threshold Invariance",
        query="What is the backup retention policy for user data?",
        target_id=125,
        target_text="Our compliance policy requires retaining all daily user database snapshots for 90 days before archiving to cold storage.",
        target_sender="Dev_Compliance",
        target_timestamp="2026-10-14T10:15:00",
        preceding_messages=[
            {"id": 124, "timestamp": "2026-10-14T10:14:00", "sender": "Dev_Lead", "message": "Let's review the compliance docs."}
        ],
        expected_answerable=True,
        expected_supporting_ids=[125],
        expected_answer_keywords=["90 days", "cold storage"],
        description="18-word self-contained answer that exceeds the 6-word threshold and bypasses lookup.",
    ),
]


# ============================================================================
# BUDGET-ENFORCING CALL COUNTER & PROMPT CAPTURE
# ============================================================================

class GlobalBudgetEnforcingLLM:
    """
    Wraps an underlying LangChain ChatModel with a fail-closed hard call counter.
    Tracks operation breakdown (grading, rewrite, generate, grounding) and prompt text.
    """
    def __init__(self, inner_llm: Any, max_calls: int = HARD_CEILING_CALLS):
        self.inner_llm = inner_llm
        self.max_calls = max_calls
        self.total_attempted: int = 0
        self.call_history: List[Dict[str, Any]] = []
        self.operation_counts: Dict[str, int] = {
            "grading": 0,
            "rewrite": 0,
            "generate": 0,
            "grounding": 0,
            "unknown": 0,
        }

    def _classify_operation(self, prompt_text: str) -> str:
        if "relevance grader" in prompt_text.lower() or "retrieved candidates" in prompt_text.lower():
            return "grading"
        if "rewrite the following question" in prompt_text.lower():
            return "rewrite"
        if "groundedness checker" in prompt_text.lower() or "factual groundedness" in prompt_text.lower():
            return "grounding"
        if "answering a user's question" in prompt_text.lower() or "strict grounding rules" in prompt_text.lower():
            return "generate"
        return "unknown"

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        prompt_str = str(prompt)
        op = self._classify_operation(prompt_str)

        # 1. Enforce hard call ceiling BEFORE calling model
        if self.total_attempted >= self.max_calls:
            raise InferenceBudgetExceededError(
                f"GLOBAL INFERENCE BUDGET EXHAUSTED: Attempted {self.total_attempted}/{self.max_calls} calls. "
                f"Invocation ({op}) rejected before execution."
            )

        # 2. Increment counter
        self.total_attempted += 1
        self.operation_counts[op] = self.operation_counts.get(op, 0) + 1

        entry = {
            "index": self.total_attempted,
            "operation": op,
            "timestamp": datetime.now().isoformat(),
            "prompt_preview": prompt_str[:120].replace("\n", " "),
        }
        self.call_history.append(entry)

        # 3. Invoke underlying model
        t0 = time.time()
        try:
            res = self.inner_llm.invoke(prompt, *args, **kwargs)
            entry["latency_seconds"] = round(time.time() - t0, 3)
            return res
        except Exception as e:
            entry["latency_seconds"] = round(time.time() - t0, 3)
            entry["error"] = str(e)
            raise


# ============================================================================
# EMBEDDING INVOCATION TRACKER
# ============================================================================

class TrackingEmbeddingsWrapper:
    """Tracks embedding queries and documents separately from generation LLM calls."""
    def __init__(self, inner_embedder: Any):
        self.inner_embedder = inner_embedder
        self.query_invocations: int = 0
        self.document_batch_invocations: int = 0
        self.total_documents_embedded: int = 0

    def embed_query(self, text: str) -> List[float]:
        self.query_invocations += 1
        return self.inner_embedder.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_batch_invocations += 1
        self.total_documents_embedded += len(texts)
        return self.inner_embedder.embed_documents(texts)

    @property
    def doc_batch_invocations(self) -> int:
        """Alias for document_batch_invocations for backwards compatibility."""
        return self.document_batch_invocations

    @property
    def total_docs_embedded(self) -> int:
        """Alias for total_documents_embedded for backwards compatibility."""
        return self.total_documents_embedded


# ============================================================================
# FAIL-CLOSED STORAGE ISOLATION SANDBOX
# ============================================================================

class SafeEvaluationSandbox:
    """
    Manages an isolated temporary SQLite database and Chroma vector store.
    Guarantees that production chat_data.db and chroma_db are NEVER touched.
    """
    def __init__(self, case: EvalCase):
        self.case = case
        self.tmp_dir = tempfile.mkdtemp(prefix=f"chatlens_eval_{case.case_id}_")
        self.tmp_path = Path(self.tmp_dir).resolve()
        self.db_path = self.tmp_path / "case_chat.db"
        self.chroma_path = self.tmp_path / "case_chroma"
        self.chat_txt_path = self.tmp_path / "case_chat.txt"

        self.prod_db_path = Path("./chat_data.db").resolve()
        self.prod_chroma_path = Path("./chroma_db").resolve()

        # Save original bindings
        self.orig_db_path = chat_database.DATABASE_PATH
        self.orig_chroma_persist_dir = vectorstore.PERSIST_DIR
        self.orig_get_connection = chat_database.get_connection
        self.orig_load_msg = hybrid_retriever.load_message_vectorstore
        self.orig_load_chunk = hybrid_retriever.load_chunk_vectorstore
        self.orig_vec_load_msg = vectorstore.load_message_vectorstore
        self.orig_vec_load_chunk = vectorstore.load_chunk_vectorstore
        self.orig_defaults: Dict[Any, Tuple[Any, Any]] = {}

    def build_synthetic_transcript(self) -> str:
        """Constructs an authentic WhatsApp chat transcript for the case."""
        lines = []
        # Preceding boundary/distractor message
        lines.append("14/10/2026, 08:50 - Dev_Security: Checking port access and daily operational status.")

        # Context messages
        for m in self.case.preceding_messages:
            dt_str = "14/10/2026, 09:10"
            if "timestamp" in m and m["timestamp"]:
                try:
                    dt = datetime.fromisoformat(m["timestamp"])
                    dt_str = dt.strftime("%d/%m/%Y, %H:%M")
                except Exception:
                    pass
            lines.append(f"{dt_str} - {m['sender']}: {m['message']}")

        # Target message
        t_dt_str = "14/10/2026, 09:11"
        if self.case.target_timestamp:
            try:
                dt = datetime.fromisoformat(self.case.target_timestamp)
                t_dt_str = dt.strftime("%d/%m/%Y, %H:%M")
            except Exception:
                pass
        lines.append(f"{t_dt_str} - {self.case.target_sender}: {self.case.target_text}")

        # Post boundary message
        lines.append("14/10/2026, 09:20 - Dev_Ops: Routine status check complete.")
        return "\n".join(lines) + "\n"

    def setup(self):
        """Builds isolated SQLite and Chroma stores with synthetic chat."""
        # Write chat file
        transcript = self.build_synthetic_transcript()
        self.chat_txt_path.write_text(transcript, encoding="utf-8")

        # Parse & enrich
        msgs = parse_whatsapp_export(str(self.chat_txt_path))
        # Ensure our target and preceding message IDs match the case definition
        enriched = enrich_messages(msgs)

        # Re-map IDs to match canonical case definitions:
        id_map = {}
        for m in self.case.preceding_messages:
            for em in enriched:
                if em.message == m["message"]:
                    id_map[em.id] = m["id"]
                    em.id = m["id"]
        for em in enriched:
            if em.message == self.case.target_text:
                id_map[em.id] = self.case.target_id
                em.id = self.case.target_id

        # Save to isolated SQLite
        create_database(str(self.db_path))
        save_messages(enriched, database_path=str(self.db_path), reset=False)

        # Build isolated Chroma
        build_vectorstore(enriched, persist_dir=str(self.chroma_path))
        self.activate()

    def activate(self):
        """Installs fail-closed interceptors."""
        target_db = str(self.db_path)
        target_chroma = str(self.chroma_path)
        tmp_p = self.tmp_path

        chat_database.DATABASE_PATH = target_db
        vectorstore.PERSIST_DIR = target_chroma

        # Dynamically update defaults on functions in chat_database, hybrid_retriever, vectorstore
        for mod in (chat_database, hybrid_retriever, vectorstore):
            for name, fn in inspect.getmembers(mod, inspect.isfunction):
                if fn.__defaults__:
                    sig = inspect.signature(fn)
                    params = list(sig.parameters.values())
                    default_offset = len(params) - len(fn.__defaults__)
                    new_defaults = []
                    has_change = False
                    for idx, def_val in enumerate(fn.__defaults__):
                        p = params[default_offset + idx]
                        if p.name == "database_path" and (
                            def_val == self.orig_db_path or def_val in ("./chat_data.db", "chat_data.db")
                        ):
                            new_defaults.append(target_db)
                            has_change = True
                        elif p.name == "persist_dir" and (
                            def_val == self.orig_chroma_persist_dir or def_val in ("./chroma_db", "chroma_db")
                        ):
                            new_defaults.append(target_chroma)
                            has_change = True
                        else:
                            new_defaults.append(def_val)
                    if has_change and fn not in self.orig_defaults:
                        installed_tuple = tuple(new_defaults)
                        self.orig_defaults[fn] = (fn.__defaults__, installed_tuple)
                        fn.__defaults__ = installed_tuple

        def fail_closed_get_connection(db_path=None):
            chosen = str(db_path or target_db).replace("\\", "/")
            resolved = Path(chosen).resolve()
            if (
                resolved == self.prod_db_path
                or chosen in ("./chat_data.db", "chat_data.db")
                or resolved.name == "chat_data.db"
            ):
                raise IsolationBreachError(f"FAIL-CLOSED BREACH: Access attempted to production database: {db_path}")
            if not resolved.is_relative_to(tmp_p):
                raise IsolationBreachError(f"FAIL-CLOSED BREACH: Database access outside sandbox: {resolved}")
            return self.orig_get_connection(str(resolved))

        chat_database.get_connection = fail_closed_get_connection

        from langchain_chroma import Chroma
        def isolated_load_msg(persist_dir=None):
            return Chroma(
                collection_name=vectorstore.MESSAGE_COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=target_chroma,
            )
        def isolated_load_chunk(persist_dir=None):
            return Chroma(
                collection_name=vectorstore.COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=target_chroma,
            )

        hybrid_retriever.load_message_vectorstore = isolated_load_msg
        hybrid_retriever.load_chunk_vectorstore = isolated_load_chunk
        vectorstore.load_message_vectorstore = isolated_load_msg
        vectorstore.load_chunk_vectorstore = isolated_load_chunk

    def teardown(self):
        """Restores original globals and removes temporary sandbox."""
        for fn, (orig, installed) in self.orig_defaults.items():
            try:
                if fn.__defaults__ == installed:
                    fn.__defaults__ = orig
            except Exception:
                pass
        self.orig_defaults.clear()

        chat_database.DATABASE_PATH = self.orig_db_path
        vectorstore.PERSIST_DIR = self.orig_chroma_persist_dir
        chat_database.get_connection = self.orig_get_connection
        hybrid_retriever.load_message_vectorstore = self.orig_load_msg
        hybrid_retriever.load_chunk_vectorstore = self.orig_load_chunk
        vectorstore.load_message_vectorstore = self.orig_vec_load_msg
        vectorstore.load_chunk_vectorstore = self.orig_vec_load_chunk

        import gc
        gc.collect()

        try:
            import chromadb.api.client
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        gc.collect()

        if self.tmp_path.exists():
            for _ in range(5):
                try:
                    shutil.rmtree(self.tmp_dir, ignore_errors=False)
                    break
                except Exception:
                    gc.collect()
                    time.sleep(0.2)
            if self.tmp_path.exists():
                shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()


# ============================================================================
# EXECUTION RESULT DATA STRUCTURES
# ============================================================================

@dataclass
class CaseModeRunResult:
    case_id: str
    mode: str  # "baseline" or "treatment"
    query: str
    expected_answerable: bool
    expected_supporting_ids: List[int]
    expected_keywords: List[str]

    # Retrieval & Reranking
    retrieved_candidate_ids: List[int] = field(default_factory=list)
    retained_graded_ids: List[int] = field(default_factory=list)
    assembled_evidence_ids: List[int] = field(default_factory=list)
    target_in_retrieval: bool = False
    target_in_final_evidence: bool = False

    # Generation & Answer
    answer: str = ""
    raw_answer: str = ""
    abstention_citations_stripped: bool = False
    valid_cited_ids: List[int] = field(default_factory=list)
    invalid_cited_ids: List[int] = field(default_factory=list)
    all_citations_valid: bool = False
    abstention_detected: bool = False
    correct_abstention: bool = False

    # Groundedness & Attribution
    grounded: bool = False
    semantic_grounded: bool = False
    attribution_precision: Optional[float] = None
    attribution_recall: Optional[float] = None
    citation_validity: Optional[float] = None
    exact_fact_preserved: Optional[bool] = None
    semantic_answer_correct: Optional[bool] = None

    # Classification & Pass Status
    classification: str = "UNKNOWN"  # TP, FP, TN, FN
    passed: bool = False

    # Performance & Calls
    latency_seconds: float = 0.0
    llm_calls_made: int = 0
    rewrites_used: int = 0
    error: Optional[str] = None


@dataclass
class ModeSummaryMetrics:
    mode_name: str
    total_cases: int = 0
    answerable_cases: int = 0
    unanswerable_cases: int = 0

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    strict_accuracy: float = 0.0
    decisive_accuracy: float = 0.0
    coverage_rate: float = 0.0

    population_recall_terse: float = 0.0  # CASE-01, 02, 04
    overall_evidence_recall: float = 0.0
    citation_validity_rate: Optional[float] = None
    mean_attribution_precision: Optional[float] = None  # Macro
    citation_level_attribution_precision: Optional[float] = None  # Micro
    mean_attribution_recall: float = 0.0  # Macro recall
    groundedness_pass_rate: float = 0.0
    correct_abstention_rate: float = 0.0
    false_positives_unanswerable: int = 0
    case12_criterion_status: str = "NOT EVALUABLE (case is answerable)"

    total_llm_calls: int = 0
    mean_latency_seconds: float = 0.0


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def evaluate_run_metrics(res: CaseModeRunResult, case: EvalCase):
    """Evaluates deterministic metrics on one case run using canonical scoring contract."""
    ans_lower = res.answer.lower()

    # Detect explicit abstention
    abstention_cues = [
        "not found", "insufficient", "no information", "not mentioned",
        "does not mention", "cannot answer", "unable to find",
        "not state", "was not found", "no evidence"
    ]
    res.abstention_detected = any(cue in ans_lower for cue in abstention_cues) or not res.valid_cited_ids

    # Target in evidence
    if case.expected_answerable:
        res.target_in_final_evidence = (
            case.target_id in res.assembled_evidence_ids or case.target_id in res.retained_graded_ids
        )
    else:
        res.target_in_final_evidence = True  # N/A for unanswerable

    # Citation validity
    res.citation_validity = compute_citation_validity(res.valid_cited_ids, res.invalid_cited_ids)

    # Citation attribution precision & recall under canonical contract
    support_semantics = getattr(case, "support_semantics", "joint")
    res.attribution_precision, res.attribution_recall = evaluate_citation_attribution(
        is_answerable=case.expected_answerable,
        valid_cited_ids=res.valid_cited_ids,
        expected_supporting_ids=case.expected_supporting_ids,
        invalid_cited_ids=res.invalid_cited_ids,
        support_semantics=support_semantics,
    )

    # Exact facts and semantic correctness
    fact_eval = evaluate_exact_facts_and_qualifiers(
        case_id=case.case_id,
        answer=res.answer,
        expected_keywords=case.expected_answer_keywords,
    )
    res.semantic_answer_correct = fact_eval["semantic_answer_correct"]
    res.exact_fact_preserved = fact_eval["exact_fact_preserved"]

    # Abstention correctness
    has_citations = bool(res.valid_cited_ids or res.invalid_cited_ids)
    if not case.expected_answerable:
        res.correct_abstention = res.abstention_detected and not has_citations
    else:
        res.correct_abstention = False

    # Canonical classification
    is_joint = (support_semantics == "joint") and (len(case.expected_supporting_ids) > 1)
    res.classification = classify_case_outcome(
        expected_answerable=case.expected_answerable,
        target_in_final_evidence=res.target_in_final_evidence,
        semantic_answer_correct=bool(res.semantic_answer_correct),
        grounded=res.grounded,
        attribution_precision=res.attribution_precision,
        attribution_recall=res.attribution_recall,
        correct_abstention=res.correct_abstention,
        valid_cited_ids=res.valid_cited_ids,
        invalid_cited_ids=res.invalid_cited_ids,
        is_joint_supporting=is_joint,
        expected_supporting_ids=case.expected_supporting_ids,
    )
    res.passed = (res.classification in ("TP", "TN"))


def compute_mode_summary(results: List[CaseModeRunResult], mode_name: str) -> ModeSummaryMetrics:
    metrics = ModeSummaryMetrics(mode_name=mode_name, total_cases=len(results))
    if not results:
        return metrics

    metrics.answerable_cases = sum(1 for r in results if r.expected_answerable)
    metrics.unanswerable_cases = sum(1 for r in results if not r.expected_answerable)

    metrics.tp = sum(1 for r in results if r.classification == "TP")
    metrics.fp = sum(1 for r in results if r.classification == "FP")
    metrics.tn = sum(1 for r in results if r.classification == "TN")
    metrics.fn = sum(1 for r in results if r.classification == "FN")

    metrics.strict_accuracy = (metrics.tp + metrics.tn) / len(results)
    decisive = [r for r in results if r.classification in ("TP", "FP", "TN", "FN")]
    metrics.decisive_accuracy = (
        (metrics.tp + metrics.tn) / len(decisive) if decisive else 0.0
    )
    metrics.coverage_rate = len(decisive) / len(results)

    # Terse cases: CASE-01, CASE-02, CASE-04
    terse_runs = [r for r in results if r.case_id in ("CASE-01-Terse-Location", "CASE-02-Short-Time", "CASE-04-Yes-No-Confirmation")]
    if terse_runs:
        metrics.population_recall_terse = sum(1 for r in terse_runs if r.target_in_final_evidence) / len(terse_runs)

    ans_runs = [r for r in results if r.expected_answerable]
    if ans_runs:
        metrics.overall_evidence_recall = sum(1 for r in ans_runs if r.target_in_final_evidence) / len(ans_runs)
        metrics.groundedness_pass_rate = sum(1 for r in ans_runs if r.grounded) / len(ans_runs)
        rec_list = [r.attribution_recall for r in ans_runs if r.attribution_recall is not None]
        metrics.mean_attribution_recall = (sum(rec_list) / len(ans_runs)) if ans_runs else 0.0

    unans_runs = [r for r in results if not r.expected_answerable]
    if unans_runs:
        metrics.correct_abstention_rate = sum(1 for r in unans_runs if r.correct_abstention) / len(unans_runs)
        metrics.false_positives_unanswerable = sum(1 for r in unans_runs if r.classification == "FP")

    # Aggregate citation validity
    total_valid_cits = sum(len(r.valid_cited_ids) for r in results)
    total_invalid_cits = sum(len(r.invalid_cited_ids) for r in results)
    total_cits = total_valid_cits + total_invalid_cits
    if total_cits > 0:
        metrics.citation_validity_rate = total_valid_cits / total_cits
    else:
        metrics.citation_validity_rate = None

    # Macro mean attribution precision across cases with citations
    prec_list = [r.attribution_precision for r in results if r.attribution_precision is not None]
    if prec_list:
        metrics.mean_attribution_precision = sum(prec_list) / len(prec_list)
    else:
        metrics.mean_attribution_precision = None

    # Micro citation-level attribution precision
    supporting_citations = 0
    for r in ans_runs:
        exp_set = set(r.expected_supporting_ids)
        vc_set = set(r.valid_cited_ids)
        supporting_citations += len(vc_set.intersection(exp_set))
    if total_cits > 0:
        metrics.citation_level_attribution_precision = supporting_citations / total_cits
    else:
        metrics.citation_level_attribution_precision = None

    metrics.total_llm_calls = sum(r.llm_calls_made for r in results)
    metrics.mean_latency_seconds = sum(r.latency_seconds for r in results) / len(results)
    return metrics


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

def execute_paired_evaluation(
    live_model_name: str = DEFAULT_MODEL,
    max_budget: int = HARD_CEILING_CALLS,
    confirm_live: bool = False,
) -> Tuple[List[CaseModeRunResult], List[CaseModeRunResult], ModeSummaryMetrics, ModeSummaryMetrics, GlobalBudgetEnforcingLLM, TrackingEmbeddingsWrapper]:
    """Runs the paired comparison across all 12 cases."""
    if not confirm_live:
        raise LiveConfirmationRequiredError("Live execution requires explicit confirm_live=True.")

    # 1. Initialize local ChatOllama and global budget counter
    inner_llm = ChatOllama(
        model=live_model_name,
        temperature=0.0,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    budget_llm = GlobalBudgetEnforcingLLM(inner_llm=inner_llm, max_calls=max_budget)

    # Patch rag_graph's module-level llm with the budget counter
    orig_rag_llm = rag_graph.llm
    rag_graph.llm = budget_llm

    # Track embedding invocations separately
    orig_get_embeddings = vectorstore.get_embeddings
    real_embedder = orig_get_embeddings()
    tracking_embedder = TrackingEmbeddingsWrapper(real_embedder)
    vectorstore.get_embeddings = lambda: tracking_embedder

    baseline_results: List[CaseModeRunResult] = []
    treatment_results: List[CaseModeRunResult] = []

    print("\n" + "=" * 80)
    print("STARTING PAIRED EVALUATION: BASELINE VS DIALOGUE EVIDENCE PIPELINE")
    print(f"Model: {live_model_name} | Max Call Ceiling: {max_budget}")
    print("=" * 80)

    try:
        for idx, case in enumerate(EVAL_CASES, start=1):
            print(f"\n----------------------------------------------------------------------")
            print(f"CASE {idx:02d}/12: [{case.case_id}] {case.category}")
            print(f"Query: \"{case.query}\"")
            print(f"----------------------------------------------------------------------")

            # Setup isolated sandbox for this case
            with SafeEvaluationSandbox(case) as env:
                # -------------------------------------------------------------
                # 1. BASELINE RUN (grade_documents filter)
                # -------------------------------------------------------------
                print(f"  [1/2] Running Baseline Mode (Default Path)...")
                t0_base = time.time()
                calls_before_base = budget_llm.total_attempted
                base_run = CaseModeRunResult(
                    case_id=case.case_id,
                    mode="baseline",
                    query=case.query,
                    expected_answerable=case.expected_answerable,
                    expected_supporting_ids=case.expected_supporting_ids,
                    expected_keywords=case.expected_answer_keywords,
                )

                try:
                    # Explicitly set dialogue evidence flag to False
                    os.environ["CHATLENS_DIALOGUE_EVIDENCE_EXPERIMENT"] = "0"
                    ans_base = rag_graph.ask(
                        question=case.query,
                        dialogue_evidence_mode=False,
                        database_path=str(env.db_path),
                    )
                    base_run.latency_seconds = round(time.time() - t0_base, 3)
                    base_run.llm_calls_made = budget_llm.total_attempted - calls_before_base
                    base_run.answer = ans_base.get("answer", "")
                    base_run.raw_answer = ans_base.get("raw_answer", base_run.answer)
                    base_run.abstention_citations_stripped = ans_base.get("abstention_citations_stripped", False)
                    base_run.grounded = ans_base.get("grounded", False)
                    base_run.valid_cited_ids = ans_base.get("valid_citation_ids", [])
                    base_run.invalid_citation_ids = ans_base.get("invalid_citation_ids", [])
                    base_run.rewrites_used = ans_base.get("rewrites_used", 0)
                    val = ans_base.get("citation_validation")
                    base_run.all_citations_valid = val.get("all_cited_valid", False) if val else (len(base_run.invalid_citation_ids) == 0)

                    for s in ans_base.get("sources", []):
                        mid = s.get("message_id")
                        if mid is not None:
                            base_run.retained_graded_ids.append(mid)

                    base_run.target_in_retrieval = case.target_id in base_run.retained_graded_ids
                    evaluate_run_metrics(base_run, case)
                    print(f"        Result: {base_run.classification} | Grounded: {base_run.grounded} | Calls: {base_run.llm_calls_made} | Cited: {base_run.valid_cited_ids}")

                except Exception as e:
                    base_run.latency_seconds = round(time.time() - t0_base, 3)
                    base_run.llm_calls_made = budget_llm.total_attempted - calls_before_base
                    base_run.error = str(e)
                    print(f"        [BASELINE ERROR]: {e}")
                    if isinstance(e, (InferenceBudgetExceededError, IsolationBreachError)):
                        raise

                baseline_results.append(base_run)

                # -------------------------------------------------------------
                # 2. TREATMENT RUN (assemble_and_rank_evidence RRF reranker)
                # -------------------------------------------------------------
                print(f"  [2/2] Running Treatment Mode (Dialogue Evidence Pipeline)...")
                t0_treat = time.time()
                calls_before_treat = budget_llm.total_attempted
                treat_run = CaseModeRunResult(
                    case_id=case.case_id,
                    mode="treatment",
                    query=case.query,
                    expected_answerable=case.expected_answerable,
                    expected_supporting_ids=case.expected_supporting_ids,
                    expected_keywords=case.expected_answer_keywords,
                )

                try:
                    os.environ["CHATLENS_DIALOGUE_EVIDENCE_EXPERIMENT"] = "1"
                    ans_treat = rag_graph.ask(
                        question=case.query,
                        dialogue_evidence_mode=True,
                        database_path=str(env.db_path),
                    )
                    treat_run.latency_seconds = round(time.time() - t0_treat, 3)
                    treat_run.llm_calls_made = budget_llm.total_attempted - calls_before_treat
                    treat_run.answer = ans_treat.get("answer", "")
                    treat_run.raw_answer = ans_treat.get("raw_answer", treat_run.answer)
                    treat_run.abstention_citations_stripped = ans_treat.get("abstention_citations_stripped", False)
                    treat_run.grounded = ans_treat.get("grounded", False)
                    treat_run.valid_cited_ids = ans_treat.get("valid_citation_ids", [])
                    treat_run.invalid_citation_ids = ans_treat.get("invalid_citation_ids", [])
                    treat_run.rewrites_used = ans_treat.get("rewrites_used", 0)
                    val = ans_treat.get("citation_validation")
                    treat_run.all_citations_valid = val.get("all_cited_valid", False) if val else (len(treat_run.invalid_citation_ids) == 0)

                    for s in ans_treat.get("sources", []):
                        mid = s.get("anchor_message_id") or s.get("message_id")
                        if mid is not None:
                            treat_run.assembled_evidence_ids.append(mid)
                        inc = s.get("included_message_ids", [])
                        for inmid in inc:
                            if inmid not in treat_run.assembled_evidence_ids:
                                treat_run.assembled_evidence_ids.append(inmid)

                    treat_run.target_in_final_evidence = case.target_id in treat_run.assembled_evidence_ids
                    evaluate_run_metrics(treat_run, case)
                    print(f"        Result: {treat_run.classification} | Grounded: {treat_run.grounded} | Calls: {treat_run.llm_calls_made} | Cited: {treat_run.valid_cited_ids}")

                except Exception as e:
                    treat_run.latency_seconds = round(time.time() - t0_treat, 3)
                    treat_run.llm_calls_made = budget_llm.total_attempted - calls_before_treat
                    treat_run.error = str(e)
                    print(f"        [TREATMENT ERROR]: {e}")
                    if isinstance(e, (InferenceBudgetExceededError, IsolationBreachError)):
                        raise

                treatment_results.append(treat_run)

    finally:
        # Restore rag_graph's original llm and vectorstore get_embeddings
        rag_graph.llm = orig_rag_llm
        vectorstore.get_embeddings = orig_get_embeddings
        os.environ["CHATLENS_DIALOGUE_EVIDENCE_EXPERIMENT"] = "0"

    base_summary = compute_mode_summary(baseline_results, "Baseline (Production Path)")
    treat_summary = compute_mode_summary(treatment_results, "Treatment (Dialogue Evidence Pipeline)")

    return baseline_results, treatment_results, base_summary, treat_summary, budget_llm, tracking_embedder


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live Paired Evaluation for Dialogue-Evidence Pipeline")
    parser.add_argument("--confirm-live", action="store_true", help="Authorize real local Ollama inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to evaluate (default: qwen2.5:3b)")
    parser.add_argument("--max-calls", type=int, default=HARD_CEILING_CALLS, help="Hard call ceiling (default: 84)")
    args = parser.parse_args()

    if not args.confirm_live:
        print("\n[DRY RUN DEFAULT] Live inference was not authorized with --confirm-live.")
        print(f"Registered Cases: {len(EVAL_CASES)}")
        for idx, c in enumerate(EVAL_CASES, 1):
            print(f"  {idx:02d}. [{c.case_id}] {c.category} (Target ID: {c.target_id}, Answerable: {c.expected_answerable})")
        print(f"\nPlanned Nominal Calls: {NOMINAL_BUDGET} | Hard Ceiling: {args.max_calls}")
        print("To run live inference safely in the isolated sandbox, supply --confirm-live.")
        return 0

    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"dialogue_evidence_live_eval_{ts}.json"

    base_runs, treat_runs, base_sum, treat_sum, budget_tracker, embed_tracker = execute_paired_evaluation(
        live_model_name=args.model,
        max_budget=args.max_calls,
        confirm_live=True,
    )

    # Save results to json
    report_dict = {
        "evaluation_timestamp": ts,
        "model": args.model,
        "total_calls_attempted": budget_tracker.total_attempted,
        "max_budget": args.max_calls,
        "operation_breakdown": budget_tracker.operation_counts,
        "embedding_counts": {
            "query_invocations": embed_tracker.query_invocations,
            "document_batch_invocations": embed_tracker.document_batch_invocations,
            "total_documents_embedded": embed_tracker.total_documents_embedded,
        },
        "baseline_summary": asdict(base_sum),
        "treatment_summary": asdict(treat_sum),
        "baseline_runs": [asdict(r) for r in base_runs],
        "treatment_runs": [asdict(r) for r in treat_runs],
    }

    out_file.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"\n✓ Saved complete evaluation audit trail to: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
