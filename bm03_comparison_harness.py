"""
bm03_comparison_harness.py
===========================
Safe, isolated A/B comparison harness for BM-03 (Short-Reply Context Resolution)
in ChatLens-RAG.

Evaluates relevance grading under two strictly controlled conditions:
  - Condition A (Context-Aware): Uses preceding conversational context for short replies.
  - Condition B (Standalone Baseline): Suppresses preceding context lookup for grading.

Safety Guarantees & Constraints:
--------------------------------
1. Fail-Closed Isolation: All SQLite and Chroma paths are validated to be strictly
   within a newly created temporary directory. Attempted access to production
   'chat_data.db' or 'chroma_db' is intercepted and blocked with IsolationBreachError.
2. Identical Inputs: Both conditions evaluate the EXACT SAME retrieved candidates
   from a single retrieval pass against the synthetic 'tech_ops' benchmark fixture.
3. Clean Teardown: Global module pointers and patched loaders are restored in try/finally
   blocks, and temporary directories are removed on exit.
4. Offline Testability: Can be executed with mock LLMs offline with zero network or
   Ollama calls.
"""

import inspect
import time
import copy
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import patch, MagicMock

import pandas as pd
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import chat_database
import vectorstore
import hybrid_retriever
import rag_graph
from parser import parse_whatsapp_export
from enrich import enrich_messages, to_dataframe
from chat_database import save_messages, get_connection
from vectorstore import build_vectorstore, get_embeddings
from rag_graph import (
    retrieve,
    grade_candidate_batch,
    grade_documents,
    expand_context,
    generate,
    check_grounded,
    GraphState,
)
from benchmark_suite import (
    TECH_OPS_CONVERSATION,
    BENCHMARK_CASES,
    BenchmarkCase,
    CaseExecutionResult,
    evaluate_case_metrics,
)

# Extract canonical BM-03 case definition
BM_03_CASE: BenchmarkCase = next(
    c for c in BENCHMARK_CASES if c.case_id == "BM-03-Short-Reply-Context"
)

DEFAULT_TIMEOUT_SECONDS = 60.0


class IsolationBreachError(RuntimeError):
    """Raised when an operation attempts to access storage outside the isolated temp directory."""
    pass


@dataclass
class GraderConditionResult:
    condition_name: str
    prompt: str
    model_decisions: List[Dict[str, Any]]
    retained_candidate_ids: List[int]
    rejected_candidate_ids: List[int]
    grader_status: str
    latency_seconds: float


@dataclass
class GraderComparisonResult:
    case_id: str
    question: str
    retrieved_candidate_ids: List[int]
    context_aware: GraderConditionResult
    standalone_baseline: GraderConditionResult
    prompt_has_preceding_context_A: bool
    prompt_has_preceding_context_B: bool
    decision_difference_detected: bool


@dataclass
class E2EConditionResult:
    condition_name: str
    answer: str
    valid_citation_ids: List[int]
    invalid_citation_ids: List[int]
    citations_valid: bool
    grounded: bool
    attribution_precision: float
    attribution_recall: float
    e2e_passed: bool
    stage_latencies: Dict[str, float]
    total_latency_seconds: float


@dataclass
class E2EComparisonResult:
    case_id: str
    question: str
    context_aware: E2EConditionResult
    standalone_baseline: E2EConditionResult


class SafeBM03Environment:
    """
    Manages a fail-closed, temporary isolated environment specifically for BM-03.
    """
    def __init__(self, build_real_chroma: bool = False):
        self.build_real_chroma = build_real_chroma
        self.tmp_dir = tempfile.mkdtemp(prefix="chatlens_bm03_safe_")
        self.tmp_path = Path(self.tmp_dir).resolve()
        self.chat_file = self.tmp_path / "tech_ops_chat.txt"
        self.db_path = self.tmp_path / "tech_ops_chat.db"
        self.chroma_path = self.tmp_path / "tech_ops_chroma"
        self.df: Optional[pd.DataFrame] = None
        self.ingestion_latency: float = 0.0

        # Production path references for guard assertion
        self.prod_db_path = Path("./chat_data.db").resolve()
        self.prod_chroma_path = Path("./chroma_db").resolve()

        # Original module bindings to restore on teardown
        self.orig_db_path = chat_database.DATABASE_PATH
        self.orig_chroma_persist_dir = vectorstore.PERSIST_DIR
        self.orig_get_connection = chat_database.get_connection
        self.orig_load_msg = hybrid_retriever.load_message_vectorstore
        self.orig_load_chunk = hybrid_retriever.load_chunk_vectorstore
        self.orig_vec_load_msg = vectorstore.load_message_vectorstore
        self.orig_vec_load_chunk = vectorstore.load_chunk_vectorstore
        self.orig_defaults: Dict[Any, Tuple] = {}

        self.is_active = False

    def setup(self):
        """Builds the synthetic tech_ops dataset in the isolated environment."""
        try:
            # Install isolation interceptors and bind globals before storage actions
            self.activate()

            try:
                import chromadb.api.client
                chromadb.api.client.SharedSystemClient.clear_system_cache()
            except Exception:
                pass

            t0 = time.time()
            # 1. Write synthetic chat file
            self.chat_file.write_text(TECH_OPS_CONVERSATION, encoding="utf-8")

            # 2. Parse & enrich
            messages = parse_whatsapp_export(str(self.chat_file))
            enriched = enrich_messages(messages)
            self.df = to_dataframe(enriched)

            # 3. Populate SQLite & FTS5
            save_messages(enriched, database_path=str(self.db_path), reset=True)

            # 4. Optional Chroma build
            if self.build_real_chroma:
                build_vectorstore(enriched, persist_dir=str(self.chroma_path))

            self.ingestion_latency = time.time() - t0
        except Exception:
            self.teardown()
            raise

    def activate(self):
        """Installs fail-closed interceptors for database and Chroma access."""
        target_db = str(self.db_path)
        target_chroma = str(self.chroma_path)
        resolved_tmp = self.tmp_path

        chat_database.DATABASE_PATH = target_db
        vectorstore.PERSIST_DIR = target_chroma

        # Dynamically update defaults on chat_database functions so no-arg calls use target_db
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
                # Strict rejection of explicit production database path
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

            # FAIL-CLOSED ASSERTION: Target MUST be within tmp_path
            if not resolved_chosen.is_relative_to(resolved_tmp):
                raise IsolationBreachError(
                    f"FAIL-CLOSED BREACH: Access attempted to database path outside isolated sandbox: {resolved_chosen}"
                )

            return self.orig_get_connection(str(resolved_chosen))

        chat_database.get_connection = fail_closed_get_connection
        globals()["get_connection"] = fail_closed_get_connection

        # Fail-closed Chroma path validator and interceptors (installed unconditionally in all modes)
        def _validate_chroma_path(persist_dir: Optional[str] = None) -> Path:
            if persist_dir is None:
                resolved = Path(target_chroma).resolve()
            else:
                p_str = str(persist_dir).replace("\\", "/")
                resolved = Path(persist_dir).resolve()
                # Strict rejection of explicit production Chroma path
                if (
                    resolved == self.prod_chroma_path
                    or p_str in ("./chroma_db", "chroma_db")
                    or resolved.name == "chroma_db"
                ):
                    raise IsolationBreachError(
                        f"FAIL-CLOSED BREACH: Explicit access attempted to production Chroma directory: {persist_dir}"
                    )
            if not resolved.is_relative_to(resolved_tmp):
                raise IsolationBreachError(
                    f"FAIL-CLOSED BREACH: Access attempted to Chroma directory outside isolated sandbox: {resolved}"
                )
            return resolved

        if self.build_real_chroma:
            from langchain_chroma import Chroma

            def fail_closed_load_msg(persist_dir=None):
                resolved_target = _validate_chroma_path(persist_dir)
                return Chroma(
                    collection_name=vectorstore.MESSAGE_COLLECTION_NAME,
                    embedding_function=get_embeddings(),
                    persist_directory=str(resolved_target),
                )

            def fail_closed_load_chunk(persist_dir=None):
                resolved_target = _validate_chroma_path(persist_dir)
                return Chroma(
                    collection_name=vectorstore.COLLECTION_NAME,
                    embedding_function=get_embeddings(),
                    persist_directory=str(resolved_target),
                )
        else:
            def fail_closed_load_msg(persist_dir=None):
                _validate_chroma_path(persist_dir)
                mock_store = MagicMock()
                mock_store.similarity_search.return_value = []
                mock_store.similarity_search_with_relevance_scores.return_value = []
                mock_store.as_retriever.return_value = MagicMock()
                return mock_store

            def fail_closed_load_chunk(persist_dir=None):
                _validate_chroma_path(persist_dir)
                mock_store = MagicMock()
                mock_store.similarity_search.return_value = []
                mock_store.similarity_search_with_relevance_scores.return_value = []
                mock_store.as_retriever.return_value = MagicMock()
                return mock_store

        hybrid_retriever.load_message_vectorstore = fail_closed_load_msg
        hybrid_retriever.load_chunk_vectorstore = fail_closed_load_chunk
        vectorstore.load_message_vectorstore = fail_closed_load_msg
        vectorstore.load_chunk_vectorstore = fail_closed_load_chunk

        self.is_active = True
        self.assert_fail_closed_isolation()

    def assert_fail_closed_isolation(self):
        """Verifies that all active storage configurations resolve inside the temporary directory."""
        active_db = Path(chat_database.DATABASE_PATH).resolve()
        active_chroma = Path(vectorstore.PERSIST_DIR).resolve()

        if not active_db.is_relative_to(self.tmp_path):
            raise IsolationBreachError(
                f"Isolation check failed: chat_database.DATABASE_PATH ({active_db}) is outside {self.tmp_path}"
            )
        if not active_chroma.is_relative_to(self.tmp_path):
            raise IsolationBreachError(
                f"Isolation check failed: vectorstore.PERSIST_DIR ({active_chroma}) is outside {self.tmp_path}"
            )
        if active_db == self.prod_db_path:
            raise IsolationBreachError("Isolation check failed: active database points to production chat_data.db")
        if active_chroma == self.prod_chroma_path:
            raise IsolationBreachError("Isolation check failed: active Chroma points to production chroma_db")

    def teardown(self):
        """Restores all bindings and removes the temporary directory."""
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
                # Restore original only if current defaults still match what the harness installed
                if fn.__defaults__ == installed:
                    fn.__defaults__ = orig
            except Exception:
                pass
        self.orig_defaults.clear()

        self.is_active = False

        try:
            import chromadb.api.client
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()


class PromptCapturingLLMWrapper:
    """Wraps an LLM to capture the exact prompt text passed to invoke."""
    def __init__(self, inner_llm):
        self.inner_llm = inner_llm
        self.captured_prompts: List[str] = []

    def invoke(self, prompt, *args, **kwargs):
        self.captured_prompts.append(str(prompt))
        return self.inner_llm.invoke(prompt, *args, **kwargs)


def retrieve_candidates_for_bm03(
    env: SafeBM03Environment,
    question: Optional[str] = None,
) -> List[Document]:
    """
    Performs a single retrieval pass for BM-03.
    """
    env.assert_fail_closed_isolation()
    q = question or BM_03_CASE.question
    state: GraphState = {
        "question": q,
        "original_question": q,
        "documents": [],
        "database_path": str(env.db_path),
    }
    retrieval_res = retrieve(state)
    return retrieval_res.get("documents", [])


def run_bm03_grader_comparison(
    env: SafeBM03Environment,
    retrieved_docs: List[Document],
    grader_llm: Optional[Any] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> GraderComparisonResult:
    """
    Runs the primary A/B comparison for BM-03 relevance grading:
      Condition A: Context-aware grading (with preceding context)
      Condition B: Standalone baseline (suppressing preceding context lookup)

    Guarantees:
      - Both conditions receive identical deep-copied candidate documents.
      - Isolation is asserted before each condition.
    """
    env.assert_fail_closed_isolation()
    question = BM_03_CASE.question

    active_base_llm = grader_llm if grader_llm is not None else rag_graph.llm
    retrieved_candidate_ids = [
        d.metadata.get("message_id")
        for d in retrieved_docs
        if d.metadata.get("message_id") is not None
    ]

    # -------------------------------------------------------------
    # Condition A: Context-Aware
    # -------------------------------------------------------------
    docs_a = copy.deepcopy(retrieved_docs)
    wrapper_a = PromptCapturingLLMWrapper(active_base_llm)

    t0_a = time.time()
    graded_a = grade_candidate_batch(
        question=question,
        batch_docs=docs_a,
        grader_llm=wrapper_a,
        database_path=str(env.db_path),
    )
    latency_a = time.time() - t0_a

    prompt_a = wrapper_a.captured_prompts[0] if wrapper_a.captured_prompts else ""
    decisions_a = [
        {"candidate": idx + 1, "doc_id": doc.metadata.get("message_id"), "decision": dec, "reason": rsn, "status": stat}
        for idx, (doc, dec, rsn, stat) in enumerate(graded_a)
    ]
    retained_ids_a = [
        doc.metadata.get("message_id")
        for doc, dec, _, _ in graded_a
        if dec != "irrelevant" and doc.metadata.get("message_id") is not None
    ]
    rejected_ids_a = [
        doc.metadata.get("message_id")
        for doc, dec, _, _ in graded_a
        if dec == "irrelevant" and doc.metadata.get("message_id") is not None
    ]
    status_a = "failure" if any(stat == "failure" for _, _, _, stat in graded_a) else "success"

    res_a = GraderConditionResult(
        condition_name="context_aware",
        prompt=prompt_a,
        model_decisions=decisions_a,
        retained_candidate_ids=retained_ids_a,
        rejected_candidate_ids=rejected_ids_a,
        grader_status=status_a,
        latency_seconds=latency_a,
    )

    # -------------------------------------------------------------
    # Condition B: Standalone Baseline (Preceding Context Suppressed)
    # -------------------------------------------------------------
    docs_b = copy.deepcopy(retrieved_docs)
    wrapper_b = PromptCapturingLLMWrapper(active_base_llm)

    t0_b = time.time()
    # Suppress preceding context lookup strictly for Condition B
    with patch("rag_graph.get_grading_preceding_context", return_value=None):
        graded_b = grade_candidate_batch(
            question=question,
            batch_docs=docs_b,
            grader_llm=wrapper_b,
            database_path=str(env.db_path),
        )
    latency_b = time.time() - t0_b

    prompt_b = wrapper_b.captured_prompts[0] if wrapper_b.captured_prompts else ""
    decisions_b = [
        {"candidate": idx + 1, "doc_id": doc.metadata.get("message_id"), "decision": dec, "reason": rsn, "status": stat}
        for idx, (doc, dec, rsn, stat) in enumerate(graded_b)
    ]
    retained_ids_b = [
        doc.metadata.get("message_id")
        for doc, dec, _, _ in graded_b
        if dec != "irrelevant" and doc.metadata.get("message_id") is not None
    ]
    rejected_ids_b = [
        doc.metadata.get("message_id")
        for doc, dec, _, _ in graded_b
        if dec == "irrelevant" and doc.metadata.get("message_id") is not None
    ]
    status_b = "failure" if any(stat == "failure" for _, _, _, stat in graded_b) else "success"

    res_b = GraderConditionResult(
        condition_name="standalone_baseline",
        prompt=prompt_b,
        model_decisions=decisions_b,
        retained_candidate_ids=retained_ids_b,
        rejected_candidate_ids=rejected_ids_b,
        grader_status=status_b,
        latency_seconds=latency_b,
    )

    return GraderComparisonResult(
        case_id=BM_03_CASE.case_id,
        question=question,
        retrieved_candidate_ids=retrieved_candidate_ids,
        context_aware=res_a,
        standalone_baseline=res_b,
        prompt_has_preceding_context_A=("Preceding Conversational Context:" in prompt_a),
        prompt_has_preceding_context_B=("Preceding Conversational Context:" in prompt_b),
        decision_difference_detected=(retained_ids_a != retained_ids_b or decisions_a != decisions_b),
    )


def run_bm03_e2e_comparison(
    env: SafeBM03Environment,
    retrieved_docs: List[Document],
    llm_instance: Optional[Any] = None,
) -> E2EComparisonResult:
    """
    Optional end-to-end comparison executing post-grading expansion, generation,
    and groundedness check for Condition A and Condition B.
    """
    env.assert_fail_closed_isolation()
    question = BM_03_CASE.question
    active_llm = llm_instance if llm_instance is not None else rag_graph.llm

    def execute_e2e_condition(condition_name: str, suppress_context: bool) -> E2EConditionResult:
        stage_latencies: Dict[str, float] = {}
        t_total_start = time.time()

        initial_state: GraphState = {
            "question": question,
            "original_question": question,
            "documents": copy.deepcopy(retrieved_docs),
            "generation": "",
            "rewrite_count": 0,
            "grounded": False,
            "df": env.df,
            "scope": "specific",
            "primary_intent": "semantic_rag",
            "database_path": str(env.db_path),
        }

        # 1. Grade
        t0 = time.time()
        if suppress_context:
            with patch("rag_graph.get_grading_preceding_context", return_value=None):
                state = grade_documents(initial_state, grader_llm=active_llm, database_path=str(env.db_path))
        else:
            state = grade_documents(initial_state, grader_llm=active_llm, database_path=str(env.db_path))
        stage_latencies["grade_documents"] = time.time() - t0

        # 2. Expand context
        t0 = time.time()
        state = expand_context(state)
        stage_latencies["expand_context"] = time.time() - t0

        # 3. Generate answer
        t0 = time.time()
        state = generate(state, generation_llm=active_llm)
        stage_latencies["generate"] = time.time() - t0

        # 4. Check groundedness
        t0 = time.time()
        state = check_grounded(state, grounding_llm=active_llm)
        stage_latencies["check_grounded"] = time.time() - t0

        total_latency = time.time() - t_total_start

        # Calculate metrics using benchmark evaluation engine
        case_res = CaseExecutionResult(
            case=BM_03_CASE,
            retrieved_message_ids=[
                d.metadata.get("message_id") for d in retrieved_docs if d.metadata.get("message_id") is not None
            ],
            available_context_ids=state.get("available_citation_ids", []),
            valid_citation_ids=state.get("valid_citation_ids", []),
            invalid_citation_ids=state.get("invalid_citation_ids", []),
            answer=state.get("generation", ""),
            grounded=state.get("grounded", False),
            citations_valid=(
                state.get("citation_validation", {}).get("all_cited_valid", False)
                if state.get("citation_validation")
                else (len(state.get("invalid_citation_ids", [])) == 0 and len(state.get("valid_citation_ids", [])) > 0)
            ),
            latency_seconds=total_latency,
        )
        evaluated = evaluate_case_metrics(case_res)

        return E2EConditionResult(
            condition_name=condition_name,
            answer=state.get("generation", ""),
            valid_citation_ids=state.get("valid_citation_ids", []),
            invalid_citation_ids=state.get("invalid_citation_ids", []),
            citations_valid=evaluated.citations_valid,
            grounded=evaluated.grounded,
            attribution_precision=evaluated.attribution_precision,
            attribution_recall=evaluated.attribution_recall,
            e2e_passed=evaluated.e2e_passed,
            stage_latencies=stage_latencies,
            total_latency_seconds=total_latency,
        )

    res_a = execute_e2e_condition("context_aware", suppress_context=False)
    res_b = execute_e2e_condition("standalone_baseline", suppress_context=True)

    return E2EComparisonResult(
        case_id=BM_03_CASE.case_id,
        question=question,
        context_aware=res_a,
        standalone_baseline=res_b,
    )
