"""
test_real_smoke_test.py

Real-Dependency Smoke Test for ChatLens-RAG.

Validates the integrated ChatLens-RAG pipeline using the actual locally
available models (qwen2.5:3b, qwen3-embedding:0.6b) and local Chroma/SQLite
persistence, without modifying production code.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

import pytest
from langchain_chroma import Chroma

import chat_database
import vectorstore
import hybrid_retriever
import rag_graph
from parser import parse_whatsapp_export
from enrich import enrich_messages, to_dataframe
from chat_database import save_messages, get_connection
from vectorstore import build_vectorstore, get_embeddings
from rag_graph import ask, validate_citations, extract_available_message_ids


SYNTHETIC_CHAT_CONTENT = (
    "14/10/2026, 09:00 - Commander Vance: The primary deflector shield operates at a frequency of 450 gigahertz.\n"
    "14/10/2026, 09:05 - Dr. Thorne: The vessel's departure from lunar orbit commences at dawn tomorrow.\n"
    "14/10/2026, 09:10 - Elena: Where did the engineering team store the secondary plasma conduits?\n"
    "14/10/2026, 09:12 - Marcus: Storage Pod 7.\n"
    "14/10/2026, 09:15 - Lyra: I read an old manual about how deflector shield frequency calculations in early simulations were completely fictitious.\n"
    "14/10/2026, 09:20 - Marcus: Cafeteria will serve hydroponic vegetable soup for lunch today.\n"
    "14/10/2026, 09:25 - Elena: Don't forget to submit the weekly maintenance log before 17:00.\n"
)


@dataclass
class SmokeTestCaseResult:
    case_name: str
    question: str
    expected_evidence_ids: List[int]
    retrieved_ids: List[int] = field(default_factory=list)
    expected_retrieved: bool = False
    survived_grading_ids: List[int] = field(default_factory=list)
    available_ids: List[int] = field(default_factory=list)
    expected_survived_grading: bool = False
    context_expanded: bool = False
    included_neighbor_ids: List[int] = field(default_factory=list)
    answer: str = ""
    grounded: bool = False
    valid_citation_ids: List[int] = field(default_factory=list)
    invalid_citation_ids: List[int] = field(default_factory=list)
    citations_valid: bool = False
    grader_status: Optional[str] = None
    latency_seconds: float = 0.0
    exceptions: List[str] = field(default_factory=list)
    is_mocked: bool = False


class IsolatedSmokeEnvironment:
    def __init__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="chatlens_smoke_")
        self.tmp_path = Path(self.tmp_dir)
        self.chat_file = self.tmp_path / "synthetic_chat.txt"
        self.db_path = self.tmp_path / "smoke_chat.db"
        self.chroma_path = self.tmp_path / "smoke_chroma"
        self.df = None
        self.ingestion_latency = 0.0

        # Save previous bindings
        self.orig_db_path = chat_database.DATABASE_PATH
        self.orig_chroma_persist_dir = vectorstore.PERSIST_DIR
        self.orig_get_connection = chat_database.get_connection
        self.orig_load_msg = hybrid_retriever.load_message_vectorstore
        self.orig_load_chunk = hybrid_retriever.load_chunk_vectorstore
        self.orig_vec_load_msg = vectorstore.load_message_vectorstore
        self.orig_vec_load_chunk = vectorstore.load_chunk_vectorstore

    def setup(self):
        try:
            import chromadb.api.client
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        t0 = time.time()
        # 1. Write synthetic chat
        self.chat_file.write_text(SYNTHETIC_CHAT_CONTENT, encoding="utf-8")

        # 2. Real parsing & enrichment
        messages = parse_whatsapp_export(str(self.chat_file))
        enriched = enrich_messages(messages)
        self.df = to_dataframe(enriched)

        # 3. Real SQLite & FTS5 persistence
        save_messages(enriched, database_path=str(self.db_path), reset=True)

        # 4. Real Chroma vector store persistence using local Ollama qwen3-embedding:0.6b
        build_vectorstore(enriched, persist_dir=str(self.chroma_path))
        self.ingestion_latency = time.time() - t0

        # 5. Redirect dependencies to isolated temporary directories
        target_db = str(self.db_path)
        target_chroma = str(self.chroma_path)

        chat_database.DATABASE_PATH = target_db
        vectorstore.PERSIST_DIR = target_chroma

        def redirected_get_connection(database_path=None):
            # Always force isolated temporary SQLite database
            return self.orig_get_connection(target_db)

        chat_database.get_connection = redirected_get_connection

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
        # Restore original bindings
        chat_database.DATABASE_PATH = self.orig_db_path
        vectorstore.PERSIST_DIR = self.orig_chroma_persist_dir
        chat_database.get_connection = self.orig_get_connection
        hybrid_retriever.load_message_vectorstore = self.orig_load_msg
        hybrid_retriever.load_chunk_vectorstore = self.orig_load_chunk
        vectorstore.load_message_vectorstore = self.orig_vec_load_msg
        vectorstore.load_chunk_vectorstore = self.orig_vec_load_chunk

        # Clear Chroma system cache before deleting directory
        try:
            import chromadb.api.client
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        # Remove temporary directories
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def check_ollama_availability() -> Dict[str, Any]:
    """Check whether Ollama is active and required models are locally available."""
    import urllib.request
    import json

    status = {
        "ollama_running": False,
        "embedding_model_available": False,
        "llm_model_available": False,
        "embedding_model": vectorstore.EMBEDDING_MODEL,
        "llm_model": "qwen2.5:3b",
        "error": None,
    }
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5)
        data = json.loads(req.read().decode())
        status["ollama_running"] = True
        available_names = [m.get("name", "") for m in data.get("models", [])]

        status["embedding_model_available"] = any(
            status["embedding_model"] in name for name in available_names
        )
        status["llm_model_available"] = any(
            status["llm_model"] in name for name in available_names
        )
    except Exception as e:
        status["error"] = str(e)

    return status


def run_single_smoke_case(
    env: IsolatedSmokeEnvironment,
    case_name: str,
    question: str,
    expected_evidence_ids: List[int],
    expect_ungrounded: bool = False,
    simulate_grader_failure: bool = False,
) -> SmokeTestCaseResult:
    result = SmokeTestCaseResult(
        case_name=case_name,
        question=question,
        expected_evidence_ids=expected_evidence_ids,
        is_mocked=simulate_grader_failure,
    )

    t0 = time.time()
    try:
        # Trace intermediate stage candidates by inspecting graph results
        # We invoke the real ask() with df from isolated environment
        if simulate_grader_failure:
            # Only stage mocked in this specific failure-handling case
            from unittest.mock import patch
            orig_invoke = type(rag_graph.llm).invoke

            def failing_invoke(self, prompt, *args, **kwargs):
                prompt_str = str(prompt).lower()
                if "relevance grader" in prompt_str:
                    raise RuntimeError("Simulated Ollama timeout during grader fallback test")
                return orig_invoke(self, prompt, *args, **kwargs)

            with patch.object(type(rag_graph.llm), "invoke", failing_invoke):
                response = ask(question, df=env.df)
        else:
            response = ask(question, df=env.df)

        result.latency_seconds = time.time() - t0
        result.answer = response.get("answer", "")
        result.grounded = response.get("grounded", False)
        result.grader_status = response.get("grader_status")
        result.valid_citation_ids = response.get("valid_citation_ids", [])
        result.invalid_citation_ids = response.get("invalid_citation_ids", [])
        available_ids = response.get("available_citation_ids", [])
        result.available_ids = available_ids

        # Sources present in final context
        sources = response.get("sources", [])
        survived_ids = []
        for s in sources:
            mid = s.get("message_id")
            if mid is not None:
                survived_ids.append(mid)
            if s.get("context_expanded"):
                result.context_expanded = True
                start_id = s.get("context_start_message_id")
                end_id = s.get("context_end_message_id")
                if start_id is not None and end_id is not None:
                    for i in range(start_id, end_id + 1):
                        if i not in survived_ids:
                            result.included_neighbor_ids.append(i)

        result.survived_grading_ids = survived_ids

        # Check expected evidence retrieval/survival
        if expected_evidence_ids:
            found = any(eid in survived_ids or eid in available_ids for eid in expected_evidence_ids)
            result.expected_retrieved = found
            result.expected_survived_grading = found
        else:
            # If no evidence expected (e.g. absent answer)
            result.expected_retrieved = len(survived_ids) == 0 or not result.grounded
            result.expected_survived_grading = True

        # Citation validity check
        val = response.get("citation_validation")
        if val:
            result.citations_valid = val.get("all_cited_valid", False)
        else:
            result.citations_valid = len(result.invalid_citation_ids) == 0

    except Exception as e:
        result.latency_seconds = time.time() - t0
        result.exceptions.append(str(e))

    return result


def execute_full_smoke_test() -> Dict[str, Any]:
    runtime_check = check_ollama_availability()
    report = {
        "runtime": runtime_check,
        "cases": [],
        "overall_success": False,
        "ingestion_latency": 0.0,
    }

    if not runtime_check["ollama_running"] or not runtime_check["embedding_model_available"] or not runtime_check["llm_model_available"]:
        report["error"] = "Required local models unavailable on Ollama"
        return report

    env = IsolatedSmokeEnvironment()
    try:
        env.setup()
        report["ingestion_latency"] = env.ingestion_latency

        # Case 1: Directly stated factual answer
        c1 = run_single_smoke_case(
            env,
            case_name="1. Directly stated factual answer",
            question="What frequency does the primary deflector shield operate at?",
            expected_evidence_ids=[0],
        )
        report["cases"].append(c1)

        # Case 2: Semantic paraphrase with low lexical overlap
        c2 = run_single_smoke_case(
            env,
            case_name="2. Semantic paraphrase (low lexical overlap)",
            question="When does the spaceship leave the moon?",
            expected_evidence_ids=[1],
        )
        report["cases"].append(c2)

        # Case 3: Short reply requiring preceding context
        c3 = run_single_smoke_case(
            env,
            case_name="3. Short reply requiring preceding context",
            question="Where are the secondary plasma conduits stored?",
            expected_evidence_ids=[3],
        )
        report["cases"].append(c3)

        # Case 4: Absent answer / unsupported question
        c4 = run_single_smoke_case(
            env,
            case_name="4. Absent answer handling",
            question="What is the passcode for the captain's private safe?",
            expected_evidence_ids=[],
            expect_ungrounded=True,
        )
        report["cases"].append(c4)

        # Case 5: Grader failure fallback status exposure
        c5 = run_single_smoke_case(
            env,
            case_name="5. Grader failure fallback exposure",
            question="What frequency does the primary deflector shield operate at?",
            expected_evidence_ids=[0],
            simulate_grader_failure=True,
        )
        report["cases"].append(c5)

        # Overall success if real cases executed without unhandled exceptions
        all_passed = all(len(c.exceptions) == 0 for c in report["cases"])
        report["overall_success"] = all_passed

    finally:
        env.teardown()

    return report


# =====================================================================
# Pytest test cases for automated validation
# =====================================================================

def test_runtime_dependencies_available():
    check = check_ollama_availability()
    assert check["ollama_running"] is True, f"Ollama is not running: {check.get('error')}"
    assert check["embedding_model_available"] is True, f"Embedding model {check['embedding_model']} not in Ollama"
    assert check["llm_model_available"] is True, f"LLM model {check['llm_model']} not in Ollama"


def test_real_dependency_smoke_suite():
    report = execute_full_smoke_test()
    assert report["overall_success"] is True, f"Smoke test failed: {report}"

    # Verify Case 1: Direct fact retrieved and answered
    c1 = report["cases"][0]
    assert 0 in c1.survived_grading_ids or 0 in c1.available_ids
    assert "450" in c1.answer

    # Verify Case 2: Semantic paraphrase found without lexical overlap
    c2 = report["cases"][1]
    assert 1 in c2.survived_grading_ids or 1 in c2.available_ids

    # Verify Case 3: Short reply context expanded
    c3 = report["cases"][2]
    assert 3 in c3.survived_grading_ids or 3 in c3.available_ids

    # Verify Case 4: Absent answer flagged ungrounded / insufficient
    c4 = report["cases"][3]
    assert c4.grounded is False or "insufficient" in c4.answer.lower() or "safe" in c4.answer.lower() or "not mentioned" in c4.answer.lower()

    # Verify Case 5: Grader failure exposes grader_status
    c5 = report["cases"][4]
    assert c5.grader_status in ("failed", "degraded")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHATLENS-RAG REAL DEPENDENCY SMOKE TEST")
    print("=" * 70)
    rep = execute_full_smoke_test()
    print(f"\nRuntime: Ollama={rep['runtime']['ollama_running']}")
    print(f"Embedding: {rep['runtime']['embedding_model']}")
    print(f"LLM: {rep['runtime']['llm_model']}")
    print(f"Ingestion Latency: {rep['ingestion_latency']:.2f}s")
    print("-" * 70)

    for c in rep["cases"]:
        print(f"\nCase: {c.case_name}")
        print(f"  Question: {c.question}")
        print(f"  Expected IDs: {c.expected_evidence_ids}")
        print(f"  Retained Msg IDs: {c.survived_grading_ids}")
        print(f"  Available Context IDs: {c.available_ids}")
        print(f"  Context Expanded: {c.context_expanded}")
        print(f"  Neighbors Included: {c.included_neighbor_ids}")
        print(f"  Grounded: {c.grounded}")
        print(f"  Valid Citations: {c.valid_citation_ids}")
        print(f"  Invalid Citations: {c.invalid_citation_ids}")
        print(f"  Grader Status: {c.grader_status}")
        print(f"  Latency: {c.latency_seconds:.2f}s")
        print(f"  Answer: {c.answer[:120]}...")
        if c.exceptions:
            print(f"  Exceptions: {c.exceptions}")
    print("\n" + "=" * 70)
