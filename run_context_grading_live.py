"""
run_context_grading_live.py
===========================
Multi-Case Live Relevance Grading Evaluation Runner for ChatLens-RAG.

Evaluates real-model relevance grading behavior across the 12 deterministic
synthetic evaluation cases from test_context_grading_evaluation.py under two
strictly controlled conditions:
  - Condition A (Context-Aware): Preceding conversational context supplied.
  - Condition B (Standalone Baseline): Preceding context lookup omitted.

Safety & Execution Guard:
  - Defaults to DRY-RUN mode (zero model calls, zero network/Ollama access).
  - Explicit --confirm-live flag is REQUIRED to execute live inference.
  - Planned inference-call budget enforcement (default limit: 24 calls).
  - Pre-execution budget validation: aborts before running if planned calls exceed limit.
  - Strictly isolated sandbox: rejects production chat_data.db and chroma_db.
  - Guarantees teardown and restoration of function defaults and storage paths.
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
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import chat_database
import hybrid_retriever
import rag_graph
import vectorstore
from rag_graph import (
    format_candidate_for_grader,
    grade_candidate_batch,
    parse_grader_response,
)
from test_context_grading_evaluation import (
    EVALUATION_CASES,
    ContextGradingCase,
)


# ---------------------------------------------------------------------
# Isolated Experimental Prompt Variant (Treatment)
# ---------------------------------------------------------------------

EXPERIMENTAL_GRADE_PROMPT_TEMPLATE = (
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
    "3. When Preceding Conversational Context is provided for a candidate, interpret the candidate message in the light of that dialogue. A short reply, confirmation, numerical value, or pronoun is relevant if the preceding dialogue establishes its meaning and the candidate helps answer the user question.\n"
    "4. Preceding Conversational Context is supporting evidence for interpreting the candidate message, NOT an independent candidate to grade. An unrelated or off-topic candidate message does not become relevant merely because the preceding conversation mentions the user query.\n"
    "5. Conversational chats are informal: terse replies, brief confirmations, numerical answers, typos, emojis, slang, and indirect statements may be relevant context.\n"
    "6. Do NOT reject a candidate solely because of zero lexical keyword overlap, short length, or emojis.\n"
    "7. If a candidate is clearly and unambiguously about a completely unrelated topic, mark 'irrelevant'.\n"
    "8. If a candidate is helpful or directly answers the question (either on its own or through its preceding conversational dialogue), mark 'relevant'.\n"
    "9. If you are uncertain or the context is ambiguous, mark 'uncertain'.\n"
    "10. Do not invent probabilities or numeric scores.\n\n"
    "Output MUST be a valid JSON array of objects with keys:\n"
    '- "candidate": integer candidate number (1, 2, ...)\n'
    '- "decision": exactly "relevant", "irrelevant", or "uncertain"\n'
    '- "reason": brief explanation (1 sentence)\n\n'
    "JSON Response:\n"
)


# ---------------------------------------------------------------------
# Isolated Experimental Structure-Aware Prompt Variant (Offline Study)
# ---------------------------------------------------------------------

EXPERIMENTAL_STRUCTURE_GRADE_PROMPT_TEMPLATE = (
    "You are a conservative relevance grader for a WhatsApp chat archive retrieval system.\n"
    "Assess whether each retrieved candidate contains information relevant to answering the user's question, "
    "or could plausibly serve as helpful conversational context.\n\n"
    "User Question:\n"
    "{question}\n\n"
    "Retrieved Candidates:\n"
    "{candidates_text}\n\n"
    "Evaluation Guidelines:\n"
    "1. Evaluate each candidate individually.\n"
    "2. Candidates may be single messages, conversation transcript chunks, or chronological conversation transcripts with a marked target message.\n"
    "3. When a candidate is formatted as a Chronological Conversation Transcript with a marked target message (e.g., '[Turn N (TARGET MESSAGE TO EVALUATE)]'):\n"
    "   - Use the preceding turns to resolve references, antecedents, pronouns, entities, and confirmations in the marked target message.\n"
    "   - The target message is relevant if the preceding dialogue establishes its meaning and the target message helps answer or resolve the user question.\n"
    "4. Preceding turns are supporting context for interpreting the marked target message, NOT independent candidates to grade. An unrelated or off-topic target message does not become relevant merely because preceding turns discuss the user question.\n"
    "5. Conversational chats are informal: terse replies, brief confirmations, numerical answers, typos, emojis, slang, and indirect statements may be relevant context.\n"
    "6. Do NOT reject a candidate solely because of zero lexical keyword overlap, short length, or emojis.\n"
    "7. If a candidate is clearly and unambiguously about a completely unrelated topic, mark 'irrelevant'.\n"
    "8. If a candidate is helpful or directly answers the question (either on its own or through reference resolution with preceding turns), mark 'relevant'.\n"
    "9. If you are uncertain or the context is ambiguous, mark 'uncertain'.\n"
    "10. Do not invent probabilities or numeric scores.\n\n"
    "Output MUST be a valid JSON array of objects with keys:\n"
    '- "candidate": integer candidate number (1, 2, ...)\n'
    '- "decision": exactly "relevant", "irrelevant", or "uncertain"\n'
    '- "reason": brief explanation (1 sentence)\n\n'
    "JSON Response:\n"
)


DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_MAX_CALLS = 24
DEFAULT_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------

class IsolationBreachError(Exception):
    """Raised when an operation attempts to access storage outside the sandbox."""
    pass


class InferenceBudgetExceededError(Exception):
    """Raised when planned inference calls exceed the authorized call budget."""
    pass


class LiveConfirmationRequiredError(Exception):
    """Raised when live execution is attempted without explicit user confirmation."""
    pass


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class CaseExecutionResult:
    case_id: str
    category: str
    query: str
    candidate_id: int
    candidate_text: str
    expected_label: str  # "relevant" or "irrelevant"
    condition: str  # "context_aware" or "standalone"
    prompt: str
    has_preceding_context: bool
    raw_output: str
    parsed_decision: str  # "relevant", "irrelevant", or "uncertain"
    reasoning: str
    status: str  # "success" or "failure"
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class ConditionSummaryMetrics:
    condition_name: str
    total_cases: int
    decisive_cases: int
    abstention_cases: int
    coverage_rate: float
    abstention_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    decisive_accuracy: float
    false_positive_rate: float
    strict_accuracy: float
    infra_failures: int
    total_ground_truth_relevant: int = 0
    total_ground_truth_irrelevant: int = 0
    population_recall: float = 0.0
    uncertain_relevant: int = 0
    uncertain_irrelevant: int = 0


@dataclass
class MultiCaseEvaluationReport:
    model_name: str
    cases_evaluated: int
    condition_a_results: List[CaseExecutionResult]
    condition_b_results: List[CaseExecutionResult]
    metrics_a: ConditionSummaryMetrics
    metrics_b: ConditionSummaryMetrics
    recovered_cases: List[str]
    broken_decisions: List[str]
    invariant_cases: List[str]
    infra_failures_count: int
    attempted_calls: int = 0
    max_budget: int = DEFAULT_MAX_CALLS
    is_complete: bool = True
    termination_reason: Optional[str] = None


# ---------------------------------------------------------------------
# Budget-Enforcing & Prompt-Capturing Wrapper
# ---------------------------------------------------------------------

class BudgetEnforcingLLMWrapper:
    """
    Enforces a strict runtime call budget on a grader LLM and captures prompts.
    Immediately checks the budget before each invocation and increments attempted
    calls regardless of whether the call succeeds or raises an exception.
    """
    def __init__(self, inner_llm: Any, max_calls: int = DEFAULT_MAX_CALLS):
        self.inner_llm = inner_llm
        self.max_calls = max_calls
        self.attempted_calls: int = 0
        self.captured_prompts: List[str] = []
        self.exhausted: bool = False

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        # 1. Check whether call budget is exhausted
        if self.attempted_calls >= self.max_calls:
            self.exhausted = True
            raise InferenceBudgetExceededError(
                f"Inference budget exhausted: attempted {self.attempted_calls}/{self.max_calls} calls. "
                "Call rejected before invoking model."
            )
        # 2. Increment attempted-call count BEFORE invoking model
        self.attempted_calls += 1
        self.captured_prompts.append(str(prompt))

        # 3. Invoke underlying model exactly once
        return self.inner_llm.invoke(prompt, *args, **kwargs)


class PromptCapturingLLMWrapper(BudgetEnforcingLLMWrapper):
    """Backwards-compatible alias for BudgetEnforcingLLMWrapper."""
    pass


# ---------------------------------------------------------------------
# Fail-Closed Isolated Sandbox Environment
# ---------------------------------------------------------------------

class SafeContextGradingEnvironment:
    """
    Manages an isolated temporary SQLite database populated with the 12
    synthetic evaluation cases. Installs fail-closed guards blocking access
    to production chat_data.db and chroma_db.
    """
    def __init__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="chatlens_context_eval_safe_")
        self.tmp_path = Path(self.tmp_dir).resolve()
        self.db_path = self.tmp_path / "eval_sandbox.db"
        self.chroma_path = self.tmp_path / "eval_chroma"

        # Production path references for guard assertion
        self.prod_db_path = Path("./chat_data.db").resolve()
        self.prod_chroma_path = Path("./chroma_db").resolve()

        # Original bindings to restore
        self.orig_db_path = chat_database.DATABASE_PATH
        self.orig_chroma_persist_dir = vectorstore.PERSIST_DIR
        self.orig_get_connection = chat_database.get_connection
        self.orig_load_msg = hybrid_retriever.load_message_vectorstore
        self.orig_load_chunk = hybrid_retriever.load_chunk_vectorstore
        self.orig_vec_load_msg = vectorstore.load_message_vectorstore
        self.orig_vec_load_chunk = vectorstore.load_chunk_vectorstore
        self.orig_defaults: Dict[Any, Tuple[Any, Any]] = {}

        self.is_active = False

    def setup(self):
        """Populates the isolated SQLite database with synthetic test cases."""
        try:
            self.activate()
            # Initialize schema in the temporary database
            chat_database.create_database(str(self.db_path))

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            for case in EVALUATION_CASES:
                # Insert preceding messages for context expansion
                for m in case.preceding_messages:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO messages (
                            id, timestamp, sender, message, message_type, is_question, message_length
                        ) VALUES (?, ?, ?, ?, 'chat', 0, ?)
                        """,
                        (m["id"], m["timestamp"], m["sender"], m["message"], len(m["message"].split())),
                    )

                # Insert the candidate message itself
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO messages (
                        id, timestamp, sender, message, message_type, is_question, message_length
                    ) VALUES (?, ?, ?, ?, 'chat', 0, ?)
                    """,
                    (
                        case.candidate_id,
                        case.timestamp,
                        case.sender,
                        case.candidate_text,
                        len(case.candidate_text.split()),
                    ),
                )

            conn.commit()
            conn.close()
        except Exception:
            self.teardown()
            raise

    def activate(self):
        """Installs fail-closed interceptors for database and vectorstore access."""
        target_db = str(self.db_path)
        target_chroma = str(self.chroma_path)
        resolved_tmp = self.tmp_path

        chat_database.DATABASE_PATH = target_db
        vectorstore.PERSIST_DIR = target_chroma

        # Dynamically update defaults on chat_database functions
        for name, fn in inspect.getmembers(chat_database, inspect.isfunction):
            if fn.__module__ == "chat_database" and fn.__defaults__:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                default_offset = len(params) - len(fn.__defaults__)
                new_defaults = []
                has_change = False
                for idx, def_val in enumerate(fn.__defaults__):
                    p = params[default_offset + idx]
                    if p.name == "database_path" and (
                        def_val == self.orig_db_path or def_val == "./chat_data.db"
                    ):
                        new_defaults.append(target_db)
                        has_change = True
                    else:
                        new_defaults.append(def_val)
                if has_change and fn not in self.orig_defaults:
                    installed_tuple = tuple(new_defaults)
                    self.orig_defaults[fn] = (fn.__defaults__, installed_tuple)
                    fn.__defaults__ = installed_tuple

        # Fail-closed SQLite interceptor
        def fail_closed_get_connection(database_path=None):
            if database_path is None:
                chosen_str = target_db
            else:
                p_str = str(database_path).replace("\\", "/")
                resolved = Path(str(database_path)).resolve()
                if (
                    resolved == self.prod_db_path
                    or p_str in ("./chat_data.db", "chat_data.db")
                    or resolved.name == "chat_data.db"
                ):
                    raise IsolationBreachError(
                        f"FAIL-CLOSED BREACH: Explicit access attempted to production database: {database_path}"
                    )
                chosen_str = str(database_path)

            resolved_chosen = Path(chosen_str).resolve()
            if not resolved_chosen.is_relative_to(resolved_tmp):
                raise IsolationBreachError(
                    f"FAIL-CLOSED BREACH: Access attempted outside isolated sandbox: {resolved_chosen}"
                )
            return self.orig_get_connection(str(resolved_chosen))

        chat_database.get_connection = fail_closed_get_connection
        globals()["get_connection"] = fail_closed_get_connection

        # Fail-closed Chroma path validator and interceptors
        def _validate_chroma_path(persist_dir: Optional[str] = None) -> Path:
            if persist_dir is None:
                resolved = Path(target_chroma).resolve()
            else:
                p_str = str(persist_dir).replace("\\", "/")
                resolved = Path(persist_dir).resolve()
                if (
                    resolved == self.prod_chroma_path
                    or p_str in ("./chroma_db", "chroma_db")
                    or resolved.name == "chroma_db"
                ):
                    raise IsolationBreachError(
                        f"FAIL-CLOSED BREACH: Explicit access attempted to production Chroma: {persist_dir}"
                    )
            if not resolved.is_relative_to(resolved_tmp):
                raise IsolationBreachError(
                    f"FAIL-CLOSED BREACH: Chroma access attempted outside sandbox: {resolved}"
                )
            return resolved

        def fail_closed_mock_store(persist_dir=None):
            _validate_chroma_path(persist_dir)
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = []
            mock_store.similarity_search_with_relevance_scores.return_value = []
            mock_store.as_retriever.return_value = MagicMock()
            return mock_store

        hybrid_retriever.load_message_vectorstore = fail_closed_mock_store
        hybrid_retriever.load_chunk_vectorstore = fail_closed_mock_store
        vectorstore.load_message_vectorstore = fail_closed_mock_store
        vectorstore.load_chunk_vectorstore = fail_closed_mock_store

        self.is_active = True
        self.assert_fail_closed_isolation()

    def assert_fail_closed_isolation(self):
        """Verifies active storage configuration resolves strictly inside sandbox."""
        active_db = Path(chat_database.DATABASE_PATH).resolve()
        active_chroma = Path(vectorstore.PERSIST_DIR).resolve()

        if not active_db.is_relative_to(self.tmp_path):
            raise IsolationBreachError(
                f"Isolation check failed: database {active_db} is outside {self.tmp_path}"
            )
        if not active_chroma.is_relative_to(self.tmp_path):
            raise IsolationBreachError(
                f"Isolation check failed: chroma {active_chroma} is outside {self.tmp_path}"
            )
        if active_db == self.prod_db_path:
            raise IsolationBreachError("Isolation check failed: points to production chat_data.db")
        if active_chroma == self.prod_chroma_path:
            raise IsolationBreachError("Isolation check failed: points to production chroma_db")

    def teardown(self):
        """Restores original module bindings and cleans up temporary files."""
        chat_database.DATABASE_PATH = self.orig_db_path
        vectorstore.PERSIST_DIR = self.orig_chroma_persist_dir
        chat_database.get_connection = self.orig_get_connection
        globals()["get_connection"] = self.orig_get_connection
        hybrid_retriever.load_message_vectorstore = self.orig_load_msg
        hybrid_retriever.load_chunk_vectorstore = self.orig_load_chunk
        vectorstore.load_message_vectorstore = self.orig_vec_load_msg
        vectorstore.load_chunk_vectorstore = self.orig_vec_load_chunk

        for fn, (orig, installed) in self.orig_defaults.items():
            try:
                if fn.__defaults__ == installed:
                    fn.__defaults__ = orig
            except Exception:
                pass
        self.orig_defaults.clear()

        self.is_active = False

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()


# ---------------------------------------------------------------------
# Candidate Document Construction
# ---------------------------------------------------------------------

def build_candidate_document(case: ContextGradingCase) -> Document:
    """
    Builds the standardized candidate document for a synthetic case.
    Guarantees identical representation between Condition A and Condition B.
    """
    clean_text = f"[ID {case.candidate_id}] [{case.timestamp}] {case.sender}: {case.candidate_text}"
    return Document(
        page_content=clean_text,
        metadata={
            "source": "hybrid_search",
            "message_id": case.candidate_id,
            "timestamp": case.timestamp,
            "sender": case.sender,
            "hybrid_score": 0.85,
        },
    )


# ---------------------------------------------------------------------
# Structure-Aware Formatting Helpers
# ---------------------------------------------------------------------

def _normalize_iso_timestamp(text: str) -> str:
    """
    Safely parses an ISO timestamp string into '%d/%m/%Y %I:%M %p' if valid.
    If parsing fails, returns the original text untouched.
    Never invents timestamps.
    """
    if not text:
        return ""
    text_clean = str(text).strip()
    try:
        dt = datetime.fromisoformat(text_clean)
        return dt.strftime("%d/%m/%Y %I:%M %p")
    except Exception:
        return text_clean


def _parse_preceding_messages(preceding_context: Optional[str]) -> List[str]:
    """
    Parses preceding_context produced by get_grading_preceding_context() into
    individual message blocks, preserving message boundaries, indentation, and
    trailing blank lines. Message boundaries begin at index 0 with: [ID <id>] [<timestamp>]

    Limitation: Because the upstream context producer joins messages using newline
    separation without escaping literal bracketed headers in message bodies, any line
    starting unindented with '[ID <id>] [<timestamp>]' is identified as a message boundary.
    """
    if not preceding_context or not preceding_context.strip():
        return []

    messages: List[str] = []
    current_lines: List[str] = []
    header_pattern = re.compile(r"^\[ID\s+[^\]]+\]\s+\[[^\]]+\]")

    for raw_line in preceding_context.splitlines():
        if header_pattern.match(raw_line):
            if current_lines:
                messages.append("\n".join(current_lines))
                current_lines = []
            current_lines.append(raw_line)
        else:
            if current_lines:
                current_lines.append(raw_line)
            else:
                if raw_line.strip():
                    current_lines.append(raw_line)

    if current_lines:
        messages.append("\n".join(current_lines))

    return messages


def _normalize_message_line(line: str) -> str:
    """
    Parses and normalizes message line timestamps without modifying IDs,
    speaker labels, or message text. If timestamp cannot be parsed as ISO,
    preserves original timestamp string. Preserves multiline message bodies,
    indentation, blank lines, and trailing spaces.
    """
    if not line:
        return ""
    # Pattern 1: [ID <id>] [<timestamp>] <sender>: <body...>
    m1 = re.match(r"^\s*\[ID\s+([^\]]+)\]\s+\[([^\]]+)\]\s+([^:]+):\s?(.*)$", line, re.DOTALL)
    if m1:
        msg_id, raw_ts, sender, body = m1.groups()
        norm_ts = _normalize_iso_timestamp(raw_ts)
        return f"[ID {msg_id}] [{norm_ts}] {sender}: {body}"

    # Pattern 2: [ID <id>] [<timestamp>] <body...> (no speaker colon)
    m2 = re.match(r"^\s*\[ID\s+([^\]]+)\]\s+\[([^\]]+)\]\s?(.*)$", line, re.DOTALL)
    if m2:
        msg_id, raw_ts, rest = m2.groups()
        norm_ts = _normalize_iso_timestamp(raw_ts)
        return f"[ID {msg_id}] [{norm_ts}] {rest}"

    return line


def format_candidate_structure_aware(
    idx: int,
    document: Document,
    preceding_context: Optional[str] = None,
) -> str:
    """
    Experimental structure-aware formatter for relevance grading evaluation.
    Presents preceding messages and target candidate as a unified, chronological
    conversation transcript with explicit turn labels, while preserving exact
    message IDs, timestamps, speaker labels, and multiline text.
    Marks the candidate as the target message to evaluate.
    If preceding context is unavailable, formats the candidate directly without inventing context.
    """
    source = document.metadata.get("source", "unknown")
    msg_id = document.metadata.get("message_id")
    chunk_id = document.metadata.get("chunk_id")

    cand_line = _normalize_message_line(document.page_content)

    preceding_messages = _parse_preceding_messages(preceding_context)
    normalized_preceding = [
        _normalize_message_line(msg)
        for msg in preceding_messages
    ]

    if normalized_preceding:
        if source == "hybrid_search" and msg_id is not None:
            header = f"[Candidate {idx}] (Chronological Conversation Transcript | Target Message ID: {msg_id})"
        elif source == "conversation_chunk" and chunk_id is not None:
            header = f"[Candidate {idx}] (Chronological Conversation Transcript | Chunk ID: {chunk_id})"
        else:
            header = f"[Candidate {idx}] (Chronological Conversation Transcript)"

        turns = []
        for t_idx, p_msg in enumerate(normalized_preceding, 1):
            turns.append(f"[Turn {t_idx}] {p_msg}")

        target_turn = len(normalized_preceding) + 1
        turns.append(f"[Turn {target_turn} (TARGET MESSAGE TO EVALUATE)] {cand_line}")

        return f"{header}\n" + "\n".join(turns)
    else:
        if source == "hybrid_search" and msg_id is not None:
            header = f"[Candidate {idx}] (Single Message ID: {msg_id} | TARGET MESSAGE TO EVALUATE)"
        elif source == "conversation_chunk" and chunk_id is not None:
            header = f"[Candidate {idx}] (Conversation Transcript Chunk ID: {chunk_id} | TARGET MESSAGE TO EVALUATE)"
        else:
            header = f"[Candidate {idx}] (Evidence Source: {source} | TARGET MESSAGE TO EVALUATE)"

        return f"{header}\n{cand_line}"


# ---------------------------------------------------------------------
# Single Case Evaluation
# ---------------------------------------------------------------------

def evaluate_case_condition(
    case: ContextGradingCase,
    condition: str,  # "context_aware", "standalone", "control_prompt", "treatment_prompt", "control_structure", "treatment_structure"
    grader_llm: Any,
    database_path: str,
    prompt_template: Optional[str] = None,
    candidate_formatter: Optional[Callable[[int, Document, Optional[str]], str]] = None,
    preceding_context: Optional[str] = None,
) -> CaseExecutionResult:
    """
    Evaluates a single synthetic case under a given condition using
    the real rag_graph.grade_candidate_batch() pipeline.
    """
    doc = build_candidate_document(case)

    # Use existing BudgetEnforcingLLMWrapper if provided, else create a local instance
    if isinstance(grader_llm, BudgetEnforcingLLMWrapper):
        wrapper = grader_llm
    else:
        wrapper = BudgetEnforcingLLMWrapper(grader_llm, max_calls=DEFAULT_MAX_CALLS)

    prompt_count_before = len(wrapper.captured_prompts)
    t0 = time.time()
    raw_output = ""
    error_msg = None

    if condition in ("context_aware", "control_prompt", "treatment_prompt"):
        graded_batch = grade_candidate_batch(
            question=case.query,
            batch_docs=[doc],
            grader_llm=wrapper,
            database_path=database_path,
            prompt_template=prompt_template,
        )
    elif condition == "standalone":
        # Suppress preceding context lookup strictly for Condition B
        with patch("rag_graph.get_grading_preceding_context", return_value=None):
            graded_batch = grade_candidate_batch(
                question=case.query,
                batch_docs=[doc],
                grader_llm=wrapper,
                database_path=database_path,
                prompt_template=prompt_template,
            )
    elif condition in ("control_structure", "treatment_structure"):
        graded_batch = grade_candidate_batch(
            question=case.query,
            batch_docs=[doc],
            grader_llm=wrapper,
            database_path=database_path,
            prompt_template=prompt_template,
            candidate_formatter=candidate_formatter,
            preceding_contexts=[preceding_context],
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    latency = time.time() - t0

    # Intercept budget exhaustion immediately so it is not masked as normal parsing fallback
    if getattr(wrapper, "exhausted", False):
        raise InferenceBudgetExceededError(
            f"Inference budget exhausted ({wrapper.attempted_calls}/{wrapper.max_calls} calls attempted)."
        )

    # Capture prompt passed during this evaluation
    if len(wrapper.captured_prompts) > prompt_count_before:
        prompt = wrapper.captured_prompts[prompt_count_before]
    else:
        prompt = ""
    has_preceding_context = (
        "Preceding Conversational Context:" in prompt
        or "Chronological Conversation Transcript" in prompt
        or "[Turn 1]" in prompt
    )

    if graded_batch:
        _, decision, reason, status = graded_batch[0]
        if "InferenceBudgetExceededError" in str(reason) or "Inference budget exhausted" in str(reason):
            raise InferenceBudgetExceededError(f"Budget exhausted during grading: {reason}")
    else:
        decision = "uncertain"
        reason = "Empty batch result from grade_candidate_batch"
        status = "failure"

    return CaseExecutionResult(
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        candidate_id=case.candidate_id,
        candidate_text=case.candidate_text,
        expected_label=case.expected_label,
        condition=condition,
        prompt=prompt,
        has_preceding_context=has_preceding_context,
        raw_output=raw_output,
        parsed_decision=decision,
        reasoning=reason,
        status=status,
        latency_seconds=latency,
        error=error_msg,
    )


# ---------------------------------------------------------------------
# Metrics Calculation
# ---------------------------------------------------------------------

def compute_condition_metrics(
    condition_name: str,
    results: List[CaseExecutionResult],
) -> ConditionSummaryMetrics:
    """
    Computes binary metrics, abstention rates, population recall, and coverage with explicit denominators.
    """
    total = len(results)
    decisive_results = [r for r in results if r.parsed_decision in ("relevant", "irrelevant")]
    decisive_count = len(decisive_results)
    abstention_count = total - decisive_count

    coverage = decisive_count / total if total > 0 else 0.0
    abstention_rate = abstention_count / total if total > 0 else 0.0

    tp = sum(1 for r in decisive_results if r.parsed_decision == "relevant" and r.expected_label == "relevant")
    fp = sum(1 for r in decisive_results if r.parsed_decision == "relevant" and r.expected_label == "irrelevant")
    tn = sum(1 for r in decisive_results if r.parsed_decision == "irrelevant" and r.expected_label == "irrelevant")
    fn = sum(1 for r in decisive_results if r.parsed_decision == "irrelevant" and r.expected_label == "relevant")

    total_gt_relevant = sum(1 for r in results if r.expected_label == "relevant")
    total_gt_irrelevant = sum(1 for r in results if r.expected_label == "irrelevant")
    uncertain_relevant = sum(1 for r in results if r.parsed_decision == "uncertain" and r.expected_label == "relevant")
    uncertain_irrelevant = sum(1 for r in results if r.parsed_decision == "uncertain" and r.expected_label == "irrelevant")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    decisive_acc = (tp + tn) / decisive_count if decisive_count > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    strict_acc = (tp + tn) / total if total > 0 else 0.0
    population_recall = tp / total_gt_relevant if total_gt_relevant > 0 else 0.0
    infra_failures = sum(1 for r in results if r.status == "failure")

    return ConditionSummaryMetrics(
        condition_name=condition_name,
        total_cases=total,
        decisive_cases=decisive_count,
        abstention_cases=abstention_count,
        coverage_rate=coverage,
        abstention_rate=abstention_rate,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        decisive_accuracy=decisive_acc,
        false_positive_rate=fpr,
        strict_accuracy=strict_acc,
        infra_failures=infra_failures,
        total_ground_truth_relevant=total_gt_relevant,
        total_ground_truth_irrelevant=total_gt_irrelevant,
        population_recall=population_recall,
        uncertain_relevant=uncertain_relevant,
        uncertain_irrelevant=uncertain_irrelevant,
    )


# ---------------------------------------------------------------------
# Full Suite Execution (Live or Mocked)
# ---------------------------------------------------------------------

def run_context_grading_suite(
    grader_llm: Any,
    env: SafeContextGradingEnvironment,
    cases: Optional[List[ContextGradingCase]] = None,
    max_budget: Optional[int] = None,
    mode: str = "prompt_experiment",  # "prompt_experiment" or "context_comparison"
) -> MultiCaseEvaluationReport:
    """
    Executes the 12-case A/B evaluation suite against the provided LLM instance.
    Modes:
      - 'prompt_experiment': Compares Control Prompt vs Experimental Treatment Prompt,
        holding Preceding Conversational Context constant (enabled for both).
      - 'context_comparison': Compares Context-Aware vs Standalone Baseline,
        holding Prompt Template constant (production GRADE_PROMPT_TEMPLATE).
    Enforces a single shared BudgetEnforcingLLMWrapper across all cases and conditions.
    """
    eval_cases = cases or EVALUATION_CASES
    results_a: List[CaseExecutionResult] = []
    results_b: List[CaseExecutionResult] = []

    # Single shared wrapper enforcing the overall run budget across all cases and conditions
    if isinstance(grader_llm, BudgetEnforcingLLMWrapper):
        enforcing_wrapper = grader_llm
    else:
        budget = max_budget if max_budget is not None else DEFAULT_MAX_CALLS
        enforcing_wrapper = BudgetEnforcingLLMWrapper(inner_llm=grader_llm, max_calls=budget)

    is_complete = True
    termination_reason = None

    try:
        for case in eval_cases:
            if mode == "structure_experiment":
                # Single shared context lookup per candidate
                doc = build_candidate_document(case)
                resolved_context = rag_graph.get_grading_preceding_context(
                    doc, database_path=str(env.db_path)
                )

                # Condition A: Control Formatter (Standard format_candidate_for_grader) + Production GRADE_PROMPT_TEMPLATE
                res_a = evaluate_case_condition(
                    case=case,
                    condition="control_structure",
                    grader_llm=enforcing_wrapper,
                    database_path=str(env.db_path),
                    prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
                    candidate_formatter=rag_graph.format_candidate_for_grader,
                    preceding_context=resolved_context,
                )
                results_a.append(res_a)

                # Condition B: Treatment Formatter (format_candidate_structure_aware) + EXACT SAME Production GRADE_PROMPT_TEMPLATE
                res_b = evaluate_case_condition(
                    case=case,
                    condition="treatment_structure",
                    grader_llm=enforcing_wrapper,
                    database_path=str(env.db_path),
                    prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
                    candidate_formatter=format_candidate_structure_aware,
                    preceding_context=resolved_context,
                )
                results_b.append(res_b)
            elif mode == "prompt_experiment":
                # Condition A: Control Prompt (Production GRADE_PROMPT_TEMPLATE) with Context
                res_a = evaluate_case_condition(
                    case=case,
                    condition="control_prompt",
                    grader_llm=enforcing_wrapper,
                    database_path=str(env.db_path),
                    prompt_template=rag_graph.GRADE_PROMPT_TEMPLATE,
                )
                results_a.append(res_a)

                # Condition B: Treatment Prompt (EXPERIMENTAL_GRADE_PROMPT_TEMPLATE) with Context
                res_b = evaluate_case_condition(
                    case=case,
                    condition="treatment_prompt",
                    grader_llm=enforcing_wrapper,
                    database_path=str(env.db_path),
                    prompt_template=EXPERIMENTAL_GRADE_PROMPT_TEMPLATE,
                )
                results_b.append(res_b)
            else:
                # Mode: context_comparison
                res_a = evaluate_case_condition(
                    case=case,
                    condition="context_aware",
                    grader_llm=enforcing_wrapper,
                    database_path=str(env.db_path),
                )
                results_a.append(res_a)

                res_b = evaluate_case_condition(
                    case=case,
                    condition="standalone",
                    grader_llm=enforcing_wrapper,
                    database_path=str(env.db_path),
                )
                results_b.append(res_b)
    except InferenceBudgetExceededError as exc:
        is_complete = False
        termination_reason = str(exc)

    if mode == "structure_experiment":
        name_a = "Control Formatter (Standard)"
        name_b = "Treatment Formatter (Structure-Aware)"
    elif mode == "prompt_experiment":
        name_a = "Control Prompt (Original)"
        name_b = "Treatment Prompt (Experimental)"
    else:
        name_a = "Context-Aware (Condition A)"
        name_b = "Standalone Baseline (Condition B)"

    metrics_a = compute_condition_metrics(name_a, results_a)
    metrics_b = compute_condition_metrics(name_b, results_b)

    recovered = []
    broken = []
    invariant = []

    for ra, rb in zip(results_a, results_b):
        if mode in ("prompt_experiment", "structure_experiment"):
            if ra.parsed_decision != ra.expected_label and rb.parsed_decision == rb.expected_label:
                recovered.append(ra.case_id)
            elif ra.parsed_decision == ra.expected_label and rb.parsed_decision != rb.expected_label:
                broken.append(ra.case_id)
            else:
                invariant.append(ra.case_id)
        else:
            if rb.parsed_decision != rb.expected_label and ra.parsed_decision == ra.expected_label:
                recovered.append(ra.case_id)
            elif rb.parsed_decision == rb.expected_label and ra.parsed_decision != ra.expected_label:
                broken.append(ra.case_id)
            else:
                invariant.append(ra.case_id)

    total_infra_failures = metrics_a.infra_failures + metrics_b.infra_failures
    inner_model = getattr(enforcing_wrapper.inner_llm, "model", str(type(enforcing_wrapper.inner_llm).__name__))

    return MultiCaseEvaluationReport(
        model_name=inner_model,
        cases_evaluated=len(results_a),
        condition_a_results=results_a,
        condition_b_results=results_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        recovered_cases=recovered,
        broken_decisions=broken,
        invariant_cases=invariant,
        infra_failures_count=total_infra_failures,
        attempted_calls=enforcing_wrapper.attempted_calls,
        max_budget=enforcing_wrapper.max_calls,
        is_complete=is_complete,
        termination_reason=termination_reason,
    )


# ---------------------------------------------------------------------
# Formatting & Reporting
# ---------------------------------------------------------------------

def format_evaluation_summary(report: MultiCaseEvaluationReport) -> str:
    """Formats markdown results summary table and metrics."""
    ma = report.metrics_a
    mb = report.metrics_b

    status_str = (
        f"COMPLETE (All {report.cases_evaluated} cases evaluated | {report.attempted_calls}/{report.max_budget} calls used)"
        if report.is_complete
        else f"INCOMPLETE ({report.attempted_calls}/{report.max_budget} calls used)"
    )

    lines = [
        "## ChatLens-RAG Multi-Case Relevance Grading Evaluation",
        f"**Model:** `{report.model_name}` | **Cases Evaluated:** {report.cases_evaluated} | **Status:** {status_str}",
        "",
    ]

    if not report.is_complete:
        lines.extend([
            "> [!WARNING]",
            f"> **Inference Budget Exhausted:** {report.termination_reason}",
            f"> Evaluation aborted mid-suite. The metrics below reflect only the {report.cases_evaluated} cases completed prior to budget exhaustion.",
            "",
        ])

    header_b = mb.condition_name
    header_a = ma.condition_name

    lines.extend([
        "### Case-by-Case Results",
        f"| Case ID | Category | Expected | {header_a} | {header_b} | Preceding Context | Transition |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---|",
    ])

    for ra, rb in zip(report.condition_a_results, report.condition_b_results):
        if ra.case_id in report.recovered_cases:
            transition = "**Recovered (+1 TP)**"
        elif ra.case_id in report.broken_decisions:
            transition = "**BROKEN (Regression)**"
        else:
            transition = "Invariant"

        has_ctx = "Yes" if (ra.has_preceding_context or rb.has_preceding_context) else "No"
        lines.append(
            f"| `{ra.case_id}` | {ra.category} | `{ra.expected_label}` | `{ra.parsed_decision}` | `{rb.parsed_decision}` | {has_ctx} | {transition} |"
        )

    # In prompt_experiment: A is Control, B is Treatment. Delta = B - A.
    # In context_comparison: B is Standalone, A is Context-Aware. Delta = A - B.
    is_prompt_exp = "Control" in ma.condition_name
    base_m = ma if is_prompt_exp else mb
    target_m = mb if is_prompt_exp else ma
    col1_title = f"{ma.condition_name}"
    col2_title = f"{mb.condition_name}"
    delta_title = f"Delta ({target_m.condition_name} vs {base_m.condition_name})"

    lines.extend([
        "",
        "### Summary Metrics Comparison",
        f"| Metric | {col1_title} | {col2_title} | {delta_title} |",
        "|:---|:---:|:---:|:---:|",
        f"| **Decisive Coverage** | {ma.coverage_rate:.2%} ({ma.decisive_cases}/{ma.total_cases}) | {mb.coverage_rate:.2%} ({mb.decisive_cases}/{mb.total_cases}) | {target_m.coverage_rate - base_m.coverage_rate:+.2%} |",
        f"| **Abstention Rate** | {ma.abstention_rate:.2%} ({ma.abstention_cases}/{ma.total_cases}) | {mb.abstention_rate:.2%} ({mb.abstention_cases}/{mb.total_cases}) | {target_m.abstention_rate - base_m.abstention_rate:+.2%} |",
        f"| **True Positives (TP)** | {ma.tp} | {mb.tp} | {target_m.tp - base_m.tp:+d} |",
        f"| **False Negatives (FN)** | {ma.fn} | {mb.fn} | {target_m.fn - base_m.fn:+d} |",
        f"| **True Negatives (TN)** | {ma.tn} | {mb.tn} | {target_m.tn - base_m.tn:+d} |",
        f"| **False Positives (FP)** | {ma.fp} | {mb.fp} | {target_m.fp - base_m.fp:+d} |",
        f"| **Uncertain on Relevant** | {ma.uncertain_relevant} | {mb.uncertain_relevant} | {target_m.uncertain_relevant - base_m.uncertain_relevant:+d} |",
        f"| **Uncertain on Irrelevant** | {ma.uncertain_irrelevant} | {mb.uncertain_irrelevant} | {target_m.uncertain_irrelevant - base_m.uncertain_irrelevant:+d} |",
        f"| **Precision (Decisive)** | {ma.precision:.2%} ({ma.tp}/{ma.tp + ma.fp if ma.tp + ma.fp > 0 else 1}) | {mb.precision:.2%} ({mb.tp}/{mb.tp + mb.fp if mb.tp + mb.fp > 0 else 1}) | {target_m.precision - base_m.precision:+.2%} |",
        f"| **Recall (Decisive)** | {ma.recall:.2%} ({ma.tp}/{ma.tp + ma.fn if ma.tp + ma.fn > 0 else 1}) | {mb.recall:.2%} ({mb.tp}/{mb.tp + mb.fn if mb.tp + mb.fn > 0 else 1}) | {target_m.recall - base_m.recall:+.2%} |",
        f"| **Population Recall (All Relevant)** | {ma.population_recall:.2%} ({ma.tp}/{ma.total_ground_truth_relevant}) | {mb.population_recall:.2%} ({mb.tp}/{mb.total_ground_truth_relevant}) | {target_m.population_recall - base_m.population_recall:+.2%} |",
        f"| **F1 Score** | {ma.f1:.4f} | {mb.f1:.4f} | {target_m.f1 - base_m.f1:+.4f} |",
        f"| **Decisive Accuracy** | {ma.decisive_accuracy:.2%} ({ma.tp + ma.tn}/{ma.decisive_cases if ma.decisive_cases > 0 else 1}) | {mb.decisive_accuracy:.2%} ({mb.tp + mb.tn}/{mb.decisive_cases if mb.decisive_cases > 0 else 1}) | {target_m.decisive_accuracy - base_m.decisive_accuracy:+.2%} |",
        f"| **False Positive Rate** | {ma.false_positive_rate:.2%} ({ma.fp}/{ma.fp + ma.tn if ma.fp + ma.tn > 0 else 1}) | {mb.false_positive_rate:.2%} ({mb.fp}/{mb.fp + mb.tn if mb.fp + mb.tn > 0 else 1}) | {target_m.false_positive_rate - base_m.false_positive_rate:+.2%} |",
        f"| **Strict Accuracy (Abstentions as Errors)** | {ma.strict_accuracy:.2%} ({ma.tp + ma.tn}/{ma.total_cases}) | {mb.strict_accuracy:.2%} ({mb.tp + mb.tn}/{mb.total_cases}) | {target_m.strict_accuracy - base_m.strict_accuracy:+.2%} |",
        "",
        f"**Recovered Cases:** {len(report.recovered_cases)} | **Broken Decisions:** {len(report.broken_decisions)} | **Infrastructure Failures:** {report.infra_failures_count}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI Entry Point & Dry-Run Guard
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Case Live Relevance Grading Evaluation Runner (Dry-run default)."
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        default=False,
        help="Explicit authorization flag required to execute live model calls against Ollama.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicitly run in dry-run mode (default behavior).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Grader model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"Maximum inference-call budget limit (default: {DEFAULT_MAX_CALLS}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-call inference timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}s).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prompt_experiment", "structure_experiment", "context_comparison"],
        default="prompt_experiment",
        help="Evaluation mode: 'prompt_experiment' (Control vs Treatment Prompt with Context), 'structure_experiment' (Standard vs Structure-Aware Formatter with identical prompt), or 'context_comparison' (Context-Aware vs Standalone). Default: prompt_experiment.",
    )

    args = parser.parse_args()

    # Pre-execution budget calculation
    num_cases = len(EVALUATION_CASES)
    planned_calls = num_cases * 2  # Condition A + Condition B

    print("=====================================================================")
    print("ChatLens-RAG: Multi-Case Relevance Grading Evaluation Runner")
    print("=====================================================================")
    print(f"Target Grader Model:      {args.model}")
    print(f"Evaluation Mode:          {args.mode}")
    print(f"Evaluation Cases:         {num_cases}")
    print(f"Planned Inference Calls:  {planned_calls} (12 cases x 2 conditions)")
    print(f"Max Call Budget Limit:    {args.max_calls}")
    print("=====================================================================")

    # Enforce inference budget prior to any action
    if planned_calls > args.max_calls:
        err_msg = (
            f"INFERENCE BUDGET EXCEEDED: Planned inference calls ({planned_calls}) "
            f"exceeds authorized maximum budget ({args.max_calls}). Aborting."
        )
        print(f"\n[ERROR] {err_msg}", file=sys.stderr)
        raise InferenceBudgetExceededError(err_msg)

    # STRICT LIVE GUARD: Default to dry-run unless --confirm-live is supplied
    is_live_requested = args.confirm_live and not args.dry_run

    if not is_live_requested:
        print("\n[DRY-RUN MODE] Zero model calls will be executed.")
        print("Verifying isolated sandbox setup and dataset integrity...")

        with SafeContextGradingEnvironment() as env:
            env.assert_fail_closed_isolation()
            print(f"Temporary Sandbox Location: {env.tmp_path}")
            print(f"Isolated SQLite Store:      {env.db_path}")

            print("\nEvaluation Cases Registered:")
            for idx, c in enumerate(EVALUATION_CASES, 1):
                preceding_count = len(c.preceding_messages)
                print(f"  {idx:02d}. [{c.case_id}] {c.category} (preceding_messages={preceding_count}, expected={c.expected_label})")

        print("\n---------------------------------------------------------------------")
        print("DRY-RUN VERIFICATION SUCCESSFUL.")
        print("Zero live model calls or network endpoints were contacted.")
        print("To execute live inference in an authorized session, supply --confirm-live.")
        print("---------------------------------------------------------------------")
        return 0

    # Live Mode Execution (Requires explicit --confirm-live in authorized runs)
    print("\n[LIVE MODE AUTHORIZED] Initializing isolated evaluation environment...")
    with SafeContextGradingEnvironment() as env:
        env.assert_fail_closed_isolation()
        print(f"Temporary Sandbox Location: {env.tmp_path}")
        print(f"Isolated SQLite Store:      {env.db_path}")

        from langchain_ollama import ChatOllama
        live_llm = ChatOllama(
            model=args.model,
            temperature=0.0,
            timeout=args.timeout,
        )

        wrapped_live_llm = BudgetEnforcingLLMWrapper(
            inner_llm=live_llm,
            max_calls=args.max_calls,
        )

        print(f"\nExecuting live evaluation across 12 cases ({args.mode})...")
        report = run_context_grading_suite(
            grader_llm=wrapped_live_llm,
            env=env,
            cases=EVALUATION_CASES,
            max_budget=args.max_calls,
            mode=args.mode,
        )

        print("\n" + format_evaluation_summary(report))
        if not report.is_complete:
            print(f"\n[ABORTED] {report.termination_reason}", file=sys.stderr)
            return 2

        print("\nLive grading evaluation completed safely within isolated sandbox.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (InferenceBudgetExceededError, LiveConfirmationRequiredError, IsolationBreachError) as exc:
        print(f"\n[ABORTED] {exc}", file=sys.stderr)
        sys.exit(2)
