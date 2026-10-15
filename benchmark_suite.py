"""
benchmark_suite.py

Reusable, Universal-RAG Evaluation Benchmark Suite for ChatLens-RAG.

Features 10 distinct conversational phenomena across multiple independent
conversation fixtures (technical operations and bilingual collaboration):
  1. Direct factual statements
  2. Semantic paraphrases with low lexical overlap
  3. Short replies requiring preceding context
  4. Pronouns and anaphora (references to earlier messages)
  5. Multi-message answers requiring synthesis of evidence from >1 message
  6. Questions whose answers are absent
  7. Irrelevant lexical distractors
  8. Contradictory or changing information over time (temporal revision)
  9. Varied message lengths and conversation densities (bursts vs logs)
  10. Mixed-language / bilingual patterns

Provides deterministic metrics:
  - Retrieval recall at k
  - Citation-ID validity rate
  - Citation attribution precision & recall
  - Correct abstention rate on absent-answer queries
  - Grounding confusion matrix (TP, TN, FP, FN)
  - End-to-end pass rate
  - Latency recording per query
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import json
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

import pandas as pd
from langchain_core.documents import Document

import chat_database
import vectorstore
import hybrid_retriever
import rag_graph
from parser import parse_whatsapp_export
from enrich import enrich_messages, to_dataframe
from chat_database import save_messages, get_connection
from vectorstore import build_vectorstore, get_embeddings
from rag_graph import ask, validate_citations, extract_available_message_ids


# =====================================================================
# 1. Independent Synthetic Conversation Fixtures
# =====================================================================

# Conversation 1: Technical Operations / Engineering Group
TECH_OPS_CONVERSATION = (
    "14/10/2026, 09:00 - Dev_Alex: Database port for the replica cluster is configured to 5433.\n"
    "14/10/2026, 09:05 - Dev_Sam: The system experienced memory exhaustion and worker thread starvation during the nocturnal batch cycle.\n"
    "14/10/2026, 09:10 - Dev_Alex: Where did we store the backup SSL certificate?\n"
    "14/10/2026, 09:11 - Dev_Sam: In Vault 9.\n"
    "14/10/2026, 09:15 - Dev_Alex: Vikram refactored the caching middleware earlier today.\n"
    "14/10/2026, 09:16 - Dev_Sam: He mentioned the cache TTL was reduced from 60 seconds to 10 seconds.\n"
    "14/10/2026, 09:20 - Dev_Alex: We will deploy release 2.4 next Tuesday at 3 AM.\n"
    "14/10/2026, 09:21 - Dev_Sam: The staging sign-off has been approved by the QA lead.\n"
    "14/10/2026, 09:25 - Dev_Dana: I was reading an article about how the replica cluster database port in legacy systems was vulnerable to port scanning.\n"
    "14/10/2026, 10:00 - Dev_Alex: The team standup will happen at 11:00 AM in Meeting Room A.\n"
    "14/10/2026, 10:30 - Dev_Alex: Update: client emergency call at 11:00 AM, so standup is rescheduled to 2:00 PM in Meeting Room B.\n"
    "14/10/2026, 10:35 - Dev_Sam: ack\n"
    "14/10/2026, 10:36 - Dev_Dana: sounds good\n"
    "14/10/2026, 10:37 - Dev_Alex: I will upload the complete network diagnostics trace logs to the shared engineering drive shortly.\n"
)

# Conversation 2: Bilingual / Mixed-Language Team Collaboration
TEAM_COLLAB_CONVERSATION = (
    "14/10/2026, 18:00 - Priya: Kal ka client presentation kitne baje hai?\n"
    "14/10/2026, 18:02 - Rohan: Presentation is at 10:30 AM in Conference Room 2.\n"
    "14/10/2026, 18:03 - Priya: Theek hai, main slides finalize kar leti hoon.\n"
    "14/10/2026, 18:05 - Rohan: Don't forget to include the Q3 revenue metrics in the deck.\n"
    "14/10/2026, 18:06 - Priya: Added to slide 8.\n"
    "14/10/2026, 18:07 - Rohan: Perfect, see you tomorrow.\n"
)


# =====================================================================
# 2. Benchmark Case Definition
# =====================================================================

@dataclass
class BenchmarkCase:
    case_id: str
    phenomenon: str
    dataset_name: str  # "tech_ops" or "team_collab"
    question: str
    expected_supporting_ids: List[int]
    expected_answerable: bool
    distractor_ids: List[int] = field(default_factory=list)
    superseded_ids: List[int] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    description: str = ""


BENCHMARK_CASES: List[BenchmarkCase] = [
    # 1. Direct factual statement
    BenchmarkCase(
        case_id="BM-01-Direct-Fact",
        phenomenon="Direct factual statement",
        dataset_name="tech_ops",
        question="What port is configured for the replica database cluster?",
        expected_supporting_ids=[0],
        expected_answerable=True,
        distractor_ids=[8],
        expected_keywords=["5433"],
        description="Explicit single-message fact stating replica cluster port 5433.",
    ),
    # 2. Semantic paraphrase with low lexical overlap
    BenchmarkCase(
        case_id="BM-02-Semantic-Paraphrase",
        phenomenon="Semantic paraphrase (low lexical overlap)",
        dataset_name="tech_ops",
        question="What caused the nocturnal system malfunction?",
        expected_supporting_ids=[1],
        expected_answerable=True,
        expected_keywords=["memory", "starvation", "thread"],
        description="Matches 'memory exhaustion and worker thread starvation during nocturnal batch cycle' with low lexical overlap.",
    ),
    # 3. Short reply requiring preceding context
    BenchmarkCase(
        case_id="BM-03-Short-Reply-Context",
        phenomenon="Short reply requiring preceding context",
        dataset_name="tech_ops",
        question="Where was the backup SSL certificate placed?",
        expected_supporting_ids=[3],
        expected_answerable=True,
        expected_keywords=["vault 9"],
        description="Requires Msg 3 ('In Vault 9.') whose question is Msg 2 ('Where did we store the backup SSL certificate?').",
    ),
    # 4. Pronouns and references to earlier messages (Anaphora)
    BenchmarkCase(
        case_id="BM-04-Pronoun-Anaphora",
        phenomenon="Pronouns and references to earlier messages",
        dataset_name="tech_ops",
        question="What did Vikram change the cache TTL to?",
        expected_supporting_ids=[5],
        expected_answerable=True,
        expected_keywords=["10", "seconds"],
        description="Msg 4 introduces Vikram refactoring caching middleware; Msg 5 says 'He mentioned the cache TTL was reduced... to 10 seconds.'",
    ),
    # 5. Multi-message answers requiring evidence from >1 message
    BenchmarkCase(
        case_id="BM-05-Multi-Message-Synthesis",
        phenomenon="Multi-message synthesis",
        dataset_name="tech_ops",
        question="When is release 2.4 scheduled and who approved the staging sign-off?",
        expected_supporting_ids=[6, 7],
        expected_answerable=True,
        expected_keywords=["tuesday", "qa lead"],
        description="Synthesizes Msg 6 (deployment Tuesday 3 AM) and Msg 7 (staging sign-off approved by QA lead).",
    ),
    # 6. Questions whose answers are absent
    BenchmarkCase(
        case_id="BM-06-Absent-Answer",
        phenomenon="Absent answer handling",
        dataset_name="tech_ops",
        question="What is the root password for the production bastion host?",
        expected_supporting_ids=[],
        expected_answerable=False,
        expected_keywords=[],
        description="Unanswerable question absent from chat; system must abstain without inventing citations.",
    ),
    # 7. Irrelevant lexical distractors
    BenchmarkCase(
        case_id="BM-07-Lexical-Distractor",
        phenomenon="Irrelevant lexical distractor",
        dataset_name="tech_ops",
        question="What port is configured for the replica database cluster?",
        expected_supporting_ids=[0],
        expected_answerable=True,
        distractor_ids=[8],
        expected_keywords=["5433"],
        description="Distractor Msg 8 discusses legacy replica cluster database port in an article; must not be cited.",
    ),
    # 8. Contradictory or changing information over time (Temporal revision)
    BenchmarkCase(
        case_id="BM-08-Temporal-Contradiction",
        phenomenon="Temporal revision / changing information",
        dataset_name="tech_ops",
        question="When and where is the rescheduled team standup?",
        expected_supporting_ids=[10],
        expected_answerable=True,
        superseded_ids=[9],
        expected_keywords=["2:00", "meeting room b"],
        description="Msg 9 scheduled standup for 11:00 in Room A; Msg 10 rescheduled to 2:00 PM in Room B.",
    ),
    # 9. Different message lengths and conversation densities
    BenchmarkCase(
        case_id="BM-09-Varied-Density-Length",
        phenomenon="Varied message lengths and densities",
        dataset_name="tech_ops",
        question="Where will the network diagnostics trace logs be uploaded?",
        expected_supporting_ids=[13],
        expected_answerable=True,
        expected_keywords=["engineering drive"],
        description="Pulls detailed log upload statement following rapid conversational burst (ack, sounds good).",
    ),
    # 10. Mixed-language / bilingual patterns
    BenchmarkCase(
        case_id="BM-10-Mixed-Language",
        phenomenon="Mixed-language / bilingual patterns",
        dataset_name="team_collab",
        question="What time is the client presentation tomorrow?",
        expected_supporting_ids=[1],
        expected_answerable=True,
        expected_keywords=["10:30", "conference room 2"],
        description="Mixed Hinglish/English conversation: Priya asks 'Kal ka client presentation kitne baje hai?', Rohan replies with time/room.",
    ),
    # 10b. Mixed-language short reply
    BenchmarkCase(
        case_id="BM-11-Mixed-Language-Context",
        phenomenon="Mixed-language short reply with context",
        dataset_name="team_collab",
        question="Which slide contains the Q3 revenue metrics?",
        expected_supporting_ids=[4],
        expected_answerable=True,
        expected_keywords=["slide 8"],
        description="Mixed conversation where Msg 4 ('Added to slide 8.') answers Msg 3 about Q3 revenue metrics.",
    ),
]


# =====================================================================
# 3. Isolated Evaluation Environment
# =====================================================================

class IsolatedBenchmarkEnvironment:
    """
    Manages an isolated SQLite database and Chroma vector store for a specific
    conversation dataset, fully isolated from production storage.
    """
    def __init__(self, name: str, raw_chat: str, build_real_chroma: bool = True):
        self.name = name
        self.raw_chat = raw_chat
        self.build_real_chroma = build_real_chroma
        self.tmp_dir = tempfile.mkdtemp(prefix=f"chatlens_bench_{name}_")
        self.tmp_path = Path(self.tmp_dir)
        self.chat_file = self.tmp_path / f"{name}_chat.txt"
        self.db_path = self.tmp_path / f"{name}_chat.db"
        self.chroma_path = self.tmp_path / f"{name}_chroma"
        self.df = None
        self.ingestion_latency = 0.0

        # Saved bindings
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
        # 1. Write raw chat file
        self.chat_file.write_text(self.raw_chat, encoding="utf-8")

        # 2. Real parsing & analytical enrichment
        messages = parse_whatsapp_export(str(self.chat_file))
        enriched = enrich_messages(messages)
        self.df = to_dataframe(enriched)

        # 3. Real SQLite & FTS5 persistence
        save_messages(enriched, database_path=str(self.db_path), reset=True)

        # 4. Optional Real Chroma build
        if self.build_real_chroma:
            build_vectorstore(enriched, persist_dir=str(self.chroma_path))

        self.ingestion_latency = time.time() - t0
        self.activate()

    def activate(self):
        try:
            import chromadb.api.client
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        # Redirect global pointers to this isolated instance
        target_db = str(self.db_path)
        target_chroma = str(self.chroma_path)

        chat_database.DATABASE_PATH = target_db
        vectorstore.PERSIST_DIR = target_chroma

        def redirected_get_connection(database_path=None):
            if (
                database_path is None
                or database_path == target_db
                or database_path == self.orig_db_path
                or str(database_path).endswith("chat_data.db")
            ):
                return self.orig_get_connection(target_db)
            return self.orig_get_connection(database_path)

        chat_database.get_connection = redirected_get_connection

        if self.build_real_chroma:
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
        chat_database.DATABASE_PATH = self.orig_db_path
        vectorstore.PERSIST_DIR = self.orig_chroma_persist_dir
        chat_database.get_connection = self.orig_get_connection
        hybrid_retriever.load_message_vectorstore = self.orig_load_msg
        hybrid_retriever.load_chunk_vectorstore = self.orig_load_chunk
        vectorstore.load_message_vectorstore = self.orig_vec_load_msg
        vectorstore.load_chunk_vectorstore = self.orig_vec_load_chunk

        try:
            import chromadb.api.client
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

        shutil.rmtree(self.tmp_dir, ignore_errors=True)


# =====================================================================
# 4. Measurable Outcome & Metric Calculation Engine
# =====================================================================

@dataclass
class CaseExecutionResult:
    case: BenchmarkCase
    retrieved_message_ids: List[int] = field(default_factory=list)
    retrieved_chunk_ids: List[int] = field(default_factory=list)
    expanded_neighbor_ids: List[int] = field(default_factory=list)
    available_context_ids: List[int] = field(default_factory=list)
    valid_citation_ids: List[int] = field(default_factory=list)
    invalid_citation_ids: List[int] = field(default_factory=list)
    answer: str = ""
    grounded: bool = False
    semantic_grounded: bool = False
    citations_valid: bool = False
    latency_seconds: float = 0.0
    exceptions: List[str] = field(default_factory=list)

    # Deterministic metrics per case
    retrieval_recall_at_k: float = 0.0
    citation_validity_rate: float = 0.0
    attribution_precision: float = 0.0
    attribution_recall: float = 0.0
    correct_abstention: bool = False
    distractor_avoided: bool = True
    grounding_category: str = ""  # TP, TN, FP, FN, ERROR
    e2e_passed: bool = False


@dataclass
class BenchmarkSummaryMetrics:
    total_cases: int = 0
    passed_cases: int = 0
    infrastructure_error_count: int = 0
    e2e_pass_rate: float = 0.0
    mean_retrieval_recall: float = 0.0
    citation_id_validity_rate: float = 0.0
    mean_attribution_precision: float = 0.0
    mean_attribution_recall: float = 0.0
    correct_abstention_rate: float = 0.0
    tp_count: int = 0
    tn_count: int = 0
    fp_count: int = 0
    fn_count: int = 0
    mean_latency_seconds: float = 0.0
    total_latency_seconds: float = 0.0


def evaluate_case_metrics(res: CaseExecutionResult) -> CaseExecutionResult:
    case = res.case
    expected = set(case.expected_supporting_ids)
    cited = set(res.valid_citation_ids)
    available = set(res.available_context_ids)

    # 0. Infrastructure exception handling
    if res.exceptions:
        res.grounding_category = "ERROR"
        res.e2e_passed = False
        res.retrieval_recall_at_k = 0.0
        res.citation_validity_rate = 0.0
        res.attribution_precision = 0.0
        res.attribution_recall = 0.0
        res.correct_abstention = False
        res.distractor_avoided = False
        return res

    # 1. Retrieval recall at k (against evidence available to generation)
    if expected:
        hits = len(expected.intersection(available))
        res.retrieval_recall_at_k = hits / len(expected)
    else:
        # For absent answers, retrieval is satisfied if no hallucinated evidence is expected
        res.retrieval_recall_at_k = 1.0

    # 2. Citation-ID validity rate (evaluated on unique cited IDs)
    unique_valid = set(res.valid_citation_ids)
    unique_invalid = set(res.invalid_citation_ids)
    total_unique_cited = len(unique_valid) + len(unique_invalid)
    if total_unique_cited > 0:
        res.citation_validity_rate = len(unique_valid) / total_unique_cited
    else:
        # If no citations were made, validity is 1.0 for absent cases, 0.0 for answerable
        res.citation_validity_rate = 1.0 if not case.expected_answerable else 0.0

    # 3. Citation attribution precision & recall against known supporting messages
    if case.expected_answerable:
        if cited:
            correct_citations = len(cited.intersection(expected))
            res.attribution_precision = correct_citations / len(cited)
            res.attribution_recall = correct_citations / len(expected)
        else:
            res.attribution_precision = 0.0
            res.attribution_recall = 0.0
    else:
        # For absent cases, correct attribution requires zero citations
        has_any_citations = len(res.valid_citation_ids) > 0 or len(res.invalid_citation_ids) > 0
        res.attribution_precision = 1.0 if not has_any_citations else 0.0
        res.attribution_recall = 1.0 if not has_any_citations else 0.0

    # 4. Correct abstention on absent-answer cases
    if not case.expected_answerable:
        abstention_words = ["insufficient", "not mentioned", "not found", "cannot", "does not state", "no information"]
        answer_lower = res.answer.lower()
        has_abstention_phrase = any(w in answer_lower for w in abstention_words)
        has_any_citations = len(res.valid_citation_ids) > 0 or len(res.invalid_citation_ids) > 0
        # Clean abstention requires abstaining without appending citations or placeholders
        res.correct_abstention = (not res.grounded or has_abstention_phrase) and not has_any_citations
    else:
        res.correct_abstention = True  # Not applicable, considered satisfied

    # 5. Distractor avoided
    if case.distractor_ids:
        res.distractor_avoided = all(did not in cited for did in case.distractor_ids)
    else:
        res.distractor_avoided = True

    # 6. Grounding confusion matrix
    if case.expected_answerable:
        if res.grounded:
            res.grounding_category = "TP"
        else:
            res.grounding_category = "FN"
    else:
        has_any_citations = len(res.valid_citation_ids) > 0 or len(res.invalid_citation_ids) > 0
        if not res.grounded and not has_any_citations:
            res.grounding_category = "TN"
        else:
            res.grounding_category = "FP"

    # 7. End-to-end pass condition
    if case.expected_answerable:
        res.e2e_passed = (
            res.retrieval_recall_at_k > 0.0
            and len(res.invalid_citation_ids) == 0
            and res.distractor_avoided
            and res.grounded
            and res.attribution_precision > 0.0
            and res.attribution_recall > 0.0
            and len(res.exceptions) == 0
        )
    else:
        res.e2e_passed = (
            res.correct_abstention
            and len(res.invalid_citation_ids) == 0
            and len(res.valid_citation_ids) == 0
            and res.grounding_category == "TN"
            and len(res.exceptions) == 0
        )

    return res


def aggregate_benchmark_metrics(results: List[CaseExecutionResult]) -> BenchmarkSummaryMetrics:
    total = len(results)
    if total == 0:
        return BenchmarkSummaryMetrics()

    passed = sum(1 for r in results if r.e2e_passed)
    infra_errors = sum(1 for r in results if len(r.exceptions) > 0)

    # Retrieval recall averaged specifically over answerable cases without infra errors
    answerable_cases = [r for r in results if r.case.expected_answerable and not r.exceptions]
    mean_recall = (
        sum(r.retrieval_recall_at_k for r in answerable_cases) / len(answerable_cases)
        if answerable_cases else 1.0
    )

    valid_results = [r for r in results if not r.exceptions]
    mean_validity = (
        sum(r.citation_validity_rate for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )
    mean_prec = (
        sum(r.attribution_precision for r in valid_results) / len(valid_results)
        if valid_results else 0.0
    )
    mean_att_recall = (
        sum(r.attribution_recall for r in answerable_cases) / len(answerable_cases)
        if answerable_cases else 0.0
    )

    absent_cases = [r for r in results if not r.case.expected_answerable and not r.exceptions]
    abstention_rate = (
        sum(1 for r in absent_cases if r.correct_abstention) / len(absent_cases)
        if absent_cases else 1.0
    )

    tp = sum(1 for r in results if r.grounding_category == "TP")
    tn = sum(1 for r in results if r.grounding_category == "TN")
    fp = sum(1 for r in results if r.grounding_category == "FP")
    fn = sum(1 for r in results if r.grounding_category == "FN")

    total_latency = sum(r.latency_seconds for r in results)
    mean_latency = total_latency / total

    return BenchmarkSummaryMetrics(
        total_cases=total,
        passed_cases=passed,
        infrastructure_error_count=infra_errors,
        e2e_pass_rate=passed / total,
        mean_retrieval_recall=mean_recall,
        citation_id_validity_rate=mean_validity,
        mean_attribution_precision=mean_prec,
        mean_attribution_recall=mean_att_recall,
        correct_abstention_rate=abstention_rate,
        tp_count=tp,
        tn_count=tn,
        fp_count=fp,
        fn_count=fn,
        mean_latency_seconds=mean_latency,
        total_latency_seconds=total_latency,
    )


# =====================================================================
# 5. Live Benchmark Runner (Single Run Against Local Ollama)
# =====================================================================

def run_live_benchmark() -> Tuple[List[CaseExecutionResult], BenchmarkSummaryMetrics]:
    """
    Executes the benchmark suite against real local Ollama models and isolated
    SQLite/Chroma databases.
    """
    from test_real_smoke_test import check_ollama_availability

    chk = check_ollama_availability()
    if not chk["ollama_running"] or not chk["embedding_model_available"] or not chk["llm_model_available"]:
        raise RuntimeError(f"Local Ollama dependencies unavailable: {chk}")

    # Set up both isolated environments
    env_tech = IsolatedBenchmarkEnvironment("tech_ops", TECH_OPS_CONVERSATION, build_real_chroma=True)
    env_collab = IsolatedBenchmarkEnvironment("team_collab", TEAM_COLLAB_CONVERSATION, build_real_chroma=True)

    results: List[CaseExecutionResult] = []

    try:
        env_tech.setup()
        env_collab.setup()

        for case in BENCHMARK_CASES:
            env = env_tech if case.dataset_name == "tech_ops" else env_collab
            env.activate()

            t0 = time.time()
            res = CaseExecutionResult(case=case)
            try:
                ans = ask(case.question, df=env.df)
                res.latency_seconds = time.time() - t0
                res.answer = ans.get("answer", "")
                res.grounded = ans.get("grounded", False)
                res.semantic_grounded = ans.get("semantic_grounded", False)
                res.valid_citation_ids = ans.get("valid_citation_ids", [])
                res.invalid_citation_ids = ans.get("invalid_citation_ids", [])
                res.available_context_ids = ans.get("available_citation_ids", [])

                for s in ans.get("sources", []):
                    if s.get("source") == "hybrid_search":
                        mid = s.get("message_id")
                        if mid is not None and mid not in res.retrieved_message_ids:
                            res.retrieved_message_ids.append(mid)
                    elif s.get("source") == "conversation_chunk":
                        cid = s.get("chunk_id")
                        if cid is not None and cid not in res.retrieved_chunk_ids:
                            res.retrieved_chunk_ids.append(cid)
                    if s.get("context_expanded"):
                        start_id = s.get("context_start_message_id")
                        end_id = s.get("context_end_message_id")
                        if start_id is not None and end_id is not None:
                            for nid in range(start_id, end_id + 1):
                                if nid not in res.retrieved_message_ids and nid not in res.expanded_neighbor_ids:
                                    res.expanded_neighbor_ids.append(nid)

                val = ans.get("citation_validation")
                if val:
                    res.citations_valid = val.get("all_cited_valid", False)
                else:
                    res.citations_valid = len(res.invalid_citation_ids) == 0

            except Exception as e:
                res.latency_seconds = time.time() - t0
                res.exceptions.append(str(e))

            evaluate_case_metrics(res)
            results.append(res)

    finally:
        env_tech.teardown()
        env_collab.teardown()

    summary = aggregate_benchmark_metrics(results)
    return results, summary


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHATLENS-RAG UNIVERSAL EVALUATION BENCHMARK (LIVE)")
    print("=" * 80)
    try:
        results, summary = run_live_benchmark()
        print(f"\nTotal Cases: {summary.total_cases}")
        print(f"Passed Cases: {summary.passed_cases} ({summary.e2e_pass_rate*100:.1f}%)")
        print(f"Mean Retrieval Recall: {summary.mean_retrieval_recall*100:.1f}%")
        print(f"Citation-ID Validity Rate: {summary.citation_id_validity_rate*100:.1f}%")
        print(f"Mean Attribution Precision: {summary.mean_attribution_precision*100:.1f}%")
        print(f"Mean Attribution Recall: {summary.mean_attribution_recall*100:.1f}%")
        print(f"Correct Abstention Rate: {summary.correct_abstention_rate*100:.1f}%")
        print(f"Grounding Confusion Matrix: TP={summary.tp_count}, TN={summary.tn_count}, FP={summary.fp_count}, FN={summary.fn_count}")
        print(f"Mean Latency: {summary.mean_latency_seconds:.2f}s (Total: {summary.total_latency_seconds:.2f}s)")
        print("-" * 80)
        for r in results:
            print(f"\n[{r.case.case_id}] {r.case.phenomenon}")
            print(f"  Question: {r.case.question}")
            print(f"  Expected IDs: {r.case.expected_supporting_ids} | Available Context IDs: {r.available_context_ids}")
            print(f"  Valid Cited IDs: {r.valid_citation_ids} | Invalid Cited IDs: {r.invalid_citation_ids}")
            print(f"  Grounded: {r.grounded} | E2E Passed: {r.e2e_passed} | Category: {r.grounding_category}")
            print(f"  Latency: {r.latency_seconds:.2f}s")
            print(f"  Answer: {r.answer[:140]}...")
            if r.exceptions:
                print(f"  Exceptions: {r.exceptions}")
    except Exception as e:
        print(f"Benchmark error: {e}")
    print("\n" + "=" * 80)
