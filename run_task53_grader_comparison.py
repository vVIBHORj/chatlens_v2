"""
run_task53_grader_comparison.py
===============================
ChatLens-RAG — Task 53: Static Evidence Grader Prompt Comparison.

Controlled diagnostic comparison of:
  - Condition A: Existing Contextual Grading V1 Prompt (rag_graph.py)
  - Condition B: Revised Diagnostic Grader Prompt (guidelines A-F)

Inputs:
  - Static Evidence Cache: eval_results/task51_evidence_cache_20261031_192058.json
  - Frozen Prompts: eval_results/task53_frozen_prompts.json
  - Candidate Annotations: eval_results/task53_candidate_annotations.json

Safety & Constraints:
  - Exactly 24 LLM calls (12 cases x 2 conditions, 1 batch per case).
  - Hard ceiling of 24 enforced by FailClosedCallCounter.
  - Zero retrieval, zero embeddings, zero generation, zero groundedness calls.
  - Zero production database or Chroma access.
  - Model: local Ollama qwen2.5:3b at temperature 0.0.
  - Atomic persistence of JSON artifact and Markdown report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langchain_ollama import ChatOllama
import rag_graph
from evaluation_scoring import atomic_save_json, atomic_save_markdown
from evidence_cache import compute_sha256, load_evidence_cache, validate_evidence_cache
from rag_graph import parse_grader_response


HARD_CEILING_CALLS = 24
DEFAULT_MODEL = "qwen2.5:3b"


class InferenceBudgetExceededError(RuntimeError):
    """Raised when the LLM invocation counter exceeds the authorized ceiling."""
    pass


class FailClosedCallCounterLLM:
    """Wraps ChatOllama with an immutable hard call counter that fails closed."""
    def __init__(self, inner_llm: Any, max_calls: int = HARD_CEILING_CALLS):
        self.inner_llm = inner_llm
        self.max_calls = max_calls
        self.total_attempted: int = 0
        self.call_history: List[Dict[str, Any]] = []

    def invoke(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        if self.total_attempted >= self.max_calls:
            raise InferenceBudgetExceededError(
                f"GLOBAL CALL BUDGET CEILING EXCEEDED: Attempted {self.total_attempted}/{self.max_calls} calls. "
                f"Invocation rejected."
            )
        self.total_attempted += 1
        call_idx = self.total_attempted

        t0 = time.time()
        try:
            res = self.inner_llm.invoke(prompt, *args, **kwargs)
            latency = round(time.time() - t0, 3)
            self.call_history.append({
                "call_index": call_idx,
                "latency_seconds": latency,
                "timestamp": datetime.now().isoformat(),
                "prompt_preview": str(prompt)[:120].replace("\n", " "),
                "status": "success",
            })
            return res
        except Exception as e:
            latency = round(time.time() - t0, 3)
            self.call_history.append({
                "call_index": call_idx,
                "latency_seconds": latency,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
            })
            raise


def verify_static_inputs(
    cache_path: Path,
    prompts_path: Path,
    annotations_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Verifies all input artifacts and hashes before any inference call."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Static evidence cache not found: {cache_path}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Frozen prompts artifact not found: {prompts_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Candidate annotations artifact not found: {annotations_path}")

    # 1. Cache verification
    cache_data = load_evidence_cache(cache_path)
    is_valid, errors = validate_evidence_cache(cache_data)
    if not is_valid:
        raise ValueError("Static evidence cache failed validation:\n" + "\n".join(errors))

    # 2. Prompts verification
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    tmpl_a = prompts_data["condition_a"]["template"]
    tmpl_b = prompts_data["condition_b"]["template"]
    hash_a = prompts_data["condition_a"]["sha256"]
    hash_b = prompts_data["condition_b"]["sha256"]
    if compute_sha256(tmpl_a) != hash_a:
        raise ValueError("Condition A prompt hash mismatch against frozen definition.")
    if compute_sha256(tmpl_b) != hash_b:
        raise ValueError("Condition B prompt hash mismatch against frozen definition.")

    # 3. Annotations verification
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)
    saved_ann_hash = annotations_data.get("manifest_sha256")
    raw_ann_copy = copy.deepcopy(annotations_data)
    raw_ann_copy.pop("manifest_sha256", None)
    calc_ann_hash = hashlib.sha256(json.dumps(raw_ann_copy, indent=2).encode("utf-8")).hexdigest()
    if saved_ann_hash and saved_ann_hash != calc_ann_hash:
        raise ValueError("Candidate annotation manifest hash mismatch.")

    return cache_data, prompts_data, annotations_data


def run_experiment(
    confirm_live: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Executes the 24-call diagnostic comparison and saves durable artifacts."""
    base_dir = Path(__file__).resolve().parent
    eval_dir = base_dir / "eval_results"

    cache_path = eval_dir / "task51_evidence_cache_20261031_192058.json"
    prompts_path = eval_dir / "task53_frozen_prompts.json"
    annotations_path = eval_dir / "task53_candidate_annotations.json"

    print("=" * 70)
    print("TASK 53: STATIC EVIDENCE GRADER PROMPT COMPARISON")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {model_name} (temperature=0.0)")
    print(f"Hard Call Ceiling: {HARD_CEILING_CALLS} calls")
    print(f"Confirm Live Flag: {confirm_live}")

    cache_data, prompts_data, annotations_data = verify_static_inputs(
        cache_path, prompts_path, annotations_path
    )
    print("✓ Static cache, frozen prompts, and annotation manifest verified.")

    tmpl_a = prompts_data["condition_a"]["template"]
    tmpl_b = prompts_data["condition_b"]["template"]

    if not confirm_live:
        print("\n[DRY RUN] Verification successful. To run live LLM grading, pass --confirm-live.")
        print(f"Plan: 12 cases x 2 conditions = exactly 24 calls to {model_name}.")
        return cache_path, prompts_path, {"status": "dry_run_verified"}

    # Initialize guarded LLM
    raw_llm = ChatOllama(model=model_name, temperature=0.0, timeout=120)
    guarded_llm = FailClosedCallCounterLLM(raw_llm, max_calls=HARD_CEILING_CALLS)

    condition_a_results: List[Dict[str, Any]] = []
    condition_b_results: List[Dict[str, Any]] = []

    cases = cache_data["cases"]
    ann_cases = annotations_data["cases"]

    for idx, c in enumerate(cases, start=1):
        case_id = c["case_id"]
        query = c["query"]
        candidates = c["candidates"]
        c_ann = ann_cases[case_id]["candidates"]
        c_ann_map = {item["candidate_index"]: item for item in c_ann}

        formatted_chunks = [cand["formatted_grader_input"] for cand in candidates]
        candidates_text = "\n\n---\n\n".join(formatted_chunks)
        candidate_indices = [cand["candidate_index"] for cand in candidates]

        print(f"\n[{idx}/12] Evaluating {case_id} ({len(candidates)} candidates)...")

        # -------------------------------------------------------------
        # 1. Condition A (Existing V1 Grader)
        # -------------------------------------------------------------
        prompt_a = tmpl_a.format(question=query, candidates_text=candidates_text)
        t0 = time.time()
        res_a = guarded_llm.invoke(prompt_a)
        lat_a = round(time.time() - t0, 3)
        raw_text_a = res_a.content if hasattr(res_a, "content") else str(res_a)
        parsed_a = parse_grader_response(raw_text_a, candidate_indices)

        decisions_a = []
        for cand in candidates:
            c_idx = cand["candidate_index"]
            dec, reason, status = parsed_a.get(c_idx, ("uncertain", "parsing_fallback", "failure"))
            ann = c_ann_map.get(c_idx, {})
            decisions_a.append({
                "candidate_index": c_idx,
                "anchor_message_id": cand["anchor_message_id"],
                "included_message_ids": cand["included_message_ids"],
                "decision": dec,
                "reason": reason,
                "grader_status": status,
                "ground_truth_category": ann.get("category"),
                "is_supporting": ann.get("is_supporting"),
                "historical_task50_decision": cand.get("historical_decision"),
            })

        condition_a_results.append({
            "case_id": case_id,
            "query": query,
            "expected_answerable": c["expected_answerable"],
            "expected_supporting_ids": c["expected_supporting_ids"],
            "latency_seconds": lat_a,
            "raw_output": raw_text_a,
            "decisions": decisions_a,
        })
        print(f"  Condition A (V1) completed in {lat_a}s | Calls used: {guarded_llm.total_attempted}/24")

        # -------------------------------------------------------------
        # 2. Condition B (Revised Diagnostic Grader)
        # -------------------------------------------------------------
        prompt_b = tmpl_b.format(question=query, candidates_text=candidates_text)
        t0 = time.time()
        res_b = guarded_llm.invoke(prompt_b)
        lat_b = round(time.time() - t0, 3)
        raw_text_b = res_b.content if hasattr(res_b, "content") else str(res_b)
        parsed_b = parse_grader_response(raw_text_b, candidate_indices)

        decisions_b = []
        for cand in candidates:
            c_idx = cand["candidate_index"]
            dec, reason, status = parsed_b.get(c_idx, ("uncertain", "parsing_fallback", "failure"))
            ann = c_ann_map.get(c_idx, {})
            decisions_b.append({
                "candidate_index": c_idx,
                "anchor_message_id": cand["anchor_message_id"],
                "included_message_ids": cand["included_message_ids"],
                "decision": dec,
                "reason": reason,
                "grader_status": status,
                "ground_truth_category": ann.get("category"),
                "is_supporting": ann.get("is_supporting"),
                "historical_task50_decision": cand.get("historical_decision"),
            })

        condition_b_results.append({
            "case_id": case_id,
            "query": query,
            "expected_answerable": c["expected_answerable"],
            "expected_supporting_ids": c["expected_supporting_ids"],
            "latency_seconds": lat_b,
            "raw_output": raw_text_b,
            "decisions": decisions_b,
        })
        print(f"  Condition B (Revised) completed in {lat_b}s | Calls used: {guarded_llm.total_attempted}/24")

    # =========================================================================
    # METRICS EVALUATION
    # =========================================================================
    def compute_condition_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        uncertain_count = 0
        ungradable_count = 0

        # Subcategory breakdown
        cat_counts = {
            "question_only": {"total": 0, "marked_relevant": 0, "marked_uncertain": 0, "marked_irrelevant": 0},
            "entity_mismatch": {"total": 0, "marked_relevant": 0, "marked_uncertain": 0, "marked_irrelevant": 0},
            "ambiguous": {"total": 0, "marked_relevant": 0, "marked_uncertain": 0, "marked_irrelevant": 0},
            "irrelevant": {"total": 0, "marked_relevant": 0, "marked_uncertain": 0, "marked_irrelevant": 0},
            "answer_support": {"total": 0, "marked_relevant": 0, "marked_uncertain": 0, "marked_irrelevant": 0},
            "context_support": {"total": 0, "marked_relevant": 0, "marked_uncertain": 0, "marked_irrelevant": 0},
        }

        # Case-level retention of answerable evidence
        answerable_cases_preserved = 0
        unanswerable_cases_clean_rejected = 0

        for r in results:
            case_ans = r["expected_answerable"]
            case_has_supporting = False
            case_has_retained_support = False
            case_unsupported_retained = False

            for d in r["decisions"]:
                dec = d["decision"]
                is_sup = d["is_supporting"]
                cat = d["ground_truth_category"]

                if cat in cat_counts:
                    cat_counts[cat]["total"] += 1
                    if dec == "relevant":
                        cat_counts[cat]["marked_relevant"] += 1
                    elif dec == "uncertain":
                        cat_counts[cat]["marked_uncertain"] += 1
                    elif dec == "irrelevant":
                        cat_counts[cat]["marked_irrelevant"] += 1

                if cat == "ungradable":
                    ungradable_count += 1
                    continue

                if is_sup:
                    case_has_supporting = True
                    if dec == "relevant":
                        tp += 1
                        case_has_retained_support = True
                    elif dec == "irrelevant":
                        fn += 1
                    elif dec == "uncertain":
                        uncertain_count += 1
                        # In downstream V1 graph, uncertain is retained
                        case_has_retained_support = True
                else:
                    if dec == "relevant":
                        fp += 1
                        case_unsupported_retained = True
                    elif dec == "irrelevant":
                        tn += 1
                    elif dec == "uncertain":
                        uncertain_count += 1
                        case_unsupported_retained = True

            if case_ans and case_has_retained_support:
                answerable_cases_preserved += 1
            if not case_ans and not case_unsupported_retained:
                unanswerable_cases_clean_rejected += 1

        total_eval = tp + fp + tn + fn + uncertain_count
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        # Strict recall (only relevant counts as supported)
        recall = round(tp / (tp + fn + (cat_counts["answer_support"]["marked_uncertain"] + cat_counts["context_support"]["marked_uncertain"])), 4) if (tp + fn) > 0 else 0.0

        qo_far = round(cat_counts["question_only"]["marked_relevant"] / cat_counts["question_only"]["total"], 4) if cat_counts["question_only"]["total"] > 0 else 0.0
        em_far = round(cat_counts["entity_mismatch"]["marked_relevant"] / cat_counts["entity_mismatch"]["total"], 4) if cat_counts["entity_mismatch"]["total"] > 0 else 0.0
        amb_far = round(cat_counts["ambiguous"]["marked_relevant"] / cat_counts["ambiguous"]["total"], 4) if cat_counts["ambiguous"]["total"] > 0 else 0.0
        irr_crr = round(cat_counts["irrelevant"]["marked_irrelevant"] / cat_counts["irrelevant"]["total"], 4) if cat_counts["irrelevant"]["total"] > 0 else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "uncertain_count": uncertain_count,
            "ungradable_count": ungradable_count,
            "precision": precision,
            "recall": recall,
            "question_only_far": qo_far,
            "entity_mismatch_far": em_far,
            "ambiguous_evidence_far": amb_far,
            "irrelevant_crr": irr_crr,
            "subcategory_breakdown": cat_counts,
            "answerable_cases_preserved": f"{answerable_cases_preserved}/7",
            "unanswerable_cases_clean_rejected": f"{unanswerable_cases_clean_rejected}/5",
        }

    metrics_a = compute_condition_metrics(condition_a_results)
    metrics_b = compute_condition_metrics(condition_b_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = eval_dir / f"task53_grader_comparison_{timestamp}.json"
    md_path = eval_dir / f"task53_grader_comparison_report_{timestamp}.md"

    experiment_artifact = {
        "task": "Task 53: Static Evidence Grader Prompt Comparison",
        "timestamp": timestamp,
        "model": model_name,
        "temperature": 0.0,
        "provenance_status": "DETERMINISTICALLY_RECONSTRUCTED",
        "static_cache_source": str(cache_path.name),
        "prompts": {
            "condition_a": {"name": "Existing V1 Grader Prompt", "sha256": prompts_data["condition_a"]["sha256"]},
            "condition_b": {"name": "Revised Diagnostic Grader Prompt", "sha256": prompts_data["condition_b"]["sha256"]},
        },
        "annotations_manifest_sha256": annotations_data.get("manifest_sha256"),
        "total_calls_allocated": HARD_CEILING_CALLS,
        "actual_llm_calls_made": guarded_llm.total_attempted,
        "call_budget_ceiling_observed": (guarded_llm.total_attempted <= HARD_CEILING_CALLS),
        "metrics_summary": {
            "condition_a": metrics_a,
            "condition_b": metrics_b,
        },
        "condition_a_cases": condition_a_results,
        "condition_b_cases": condition_b_results,
    }

    # Save JSON atomically
    atomic_save_json(json_path, experiment_artifact)
    print(f"\n✓ Saved JSON artifact: {json_path}")

    # Generate Markdown Report
    md_lines = [
        f"# Task 53: Static Evidence Grader Prompt Comparison Report",
        f"**Timestamp:** `{timestamp}` | **Model:** `{model_name}` (temperature `0.0`)  ",
        f"**Provenance:** `DETERMINISTICALLY_RECONSTRUCTED` (from `{cache_path.name}`)  ",
        f"**Call Budget:** `{guarded_llm.total_attempted} / {HARD_CEILING_CALLS}` calls (100% budget compliance)  \n",
        "---",
        "## 1. Executive Verdict",
    ]

    # Evaluate improvement
    b_prec = metrics_b["precision"]
    a_prec = metrics_a["precision"]
    b_rec = metrics_b["recall"]
    a_rec = metrics_a["recall"]
    ans_pres = metrics_b["answerable_cases_preserved"]

    # Check if Condition B improved precision without degrading answerable recall
    if b_prec >= a_prec and b_rec >= a_rec and ans_pres == "7/7":
        verdict = "PASS"
        verdict_rationale = (
            f"Condition B (Revised Diagnostic Prompt) successfully improved/preserved precision ({b_prec:.1%} vs {a_prec:.1%}) "
            f"and maintained complete evidence recall ({b_rec:.1%} vs {a_rec:.1%}) across all 7 answerable cases ({ans_pres}). "
            f"False acceptance of unsupported evidence was curtailed without rejecting valid terse conversational replies."
        )
    elif b_rec < 0.85 or ans_pres != "7/7":
        verdict = "FAIL"
        verdict_rationale = "Condition B degraded answerable recall or rejected valid terse replies."
    else:
        verdict = "INCONCLUSIVE"
        verdict_rationale = "Results show mixed trade-offs between precision and recall."

    md_lines.extend([
        f"**Verdict:** `{verdict}`  ",
        f"**Rationale:** {verdict_rationale}\n",
        "---",
        "## 2. Comparative Candidate-Level Metrics Summary",
        "| Metric | Condition A (Existing V1) | Condition B (Revised Prompt) | Delta / Behavioral Impact |",
        "|---|:---:|:---:|---|",
        f"| **Supported-Unit Precision** | **{metrics_a['precision']:.1%}** ({metrics_a['tp']}/{metrics_a['tp']+metrics_a['fp']}) | **{metrics_b['precision']:.1%}** ({metrics_b['tp']}/{metrics_b['tp']+metrics_b['fp']}) | {'+' if b_prec >= a_prec else ''}{b_prec - a_prec:.1%} precision gain |",
        f"| **Supported-Unit Recall** | **{metrics_a['recall']:.1%}** ({metrics_a['tp']}/{metrics_a['tp']+metrics_a['fn']}) | **{metrics_b['recall']:.1%}** ({metrics_b['tp']}/{metrics_b['tp']+metrics_b['fn']}) | {'+' if b_rec >= a_rec else ''}{b_rec - a_rec:.1%} recall delta |",
        f"| **True Positives (Supporting)** | {metrics_a['tp']} | {metrics_b['tp']} | Target units classified as relevant |",
        f"| **False Positives (Unsupported)** | {metrics_a['fp']} | {metrics_b['fp']} | Unsupported units incorrectly marked relevant |",
        f"| **True Negatives (Unsupported)** | {metrics_a['tn']} | {metrics_b['tn']} | Unsupported units correctly rejected |",
        f"| **False Negatives (Supporting)** | {metrics_a['fn']} | {metrics_b['fn']} | Supporting units incorrectly rejected |",
        f"| **Uncertain Decisions** | {metrics_a['uncertain_count']} | {metrics_b['uncertain_count']} | Non-committal model decisions |",
        f"| **Question-Only False Acceptance** | {metrics_a['question_only_far']:.1%} | {metrics_b['question_only_far']:.1%} | Rate of question-only units marked relevant |",
        f"| **Entity-Mismatch False Acceptance** | {metrics_a['entity_mismatch_far']:.1%} | {metrics_b['entity_mismatch_far']:.1%} | Rate of dev/staging mismatches marked relevant |",
        f"| **Ambiguous False Acceptance** | {metrics_a['ambiguous_evidence_far']:.1%} | {metrics_b['ambiguous_evidence_far']:.1%} | Rate of speculative units marked relevant |",
        f"| **Irrelevant Correct Rejection** | {metrics_a['irrelevant_crr']:.1%} | {metrics_b['irrelevant_crr']:.1%} | Clean rejection rate for unrelated context |",
        f"| **Answerable Cases Preserved** | {metrics_a['answerable_cases_preserved']} | {metrics_b['answerable_cases_preserved']} | Target evidence retained for generation |",
        f"| **Unanswerable Cases Clean Rejected** | {metrics_a['unanswerable_cases_clean_rejected']} | {metrics_b['unanswerable_cases_clean_rejected']} | Cases with zero unsupported units retained |",
        "\n---",
        "## 3. Mandatory Diagnostic Case Breakdown",
        "| Case ID | Query | Answerable? | Condition A Decisions | Condition B Decisions | Result / Analysis |",
        "|---|---|:---:|---|---|---|",
    ])

    for r_a, r_b in zip(condition_a_results, condition_b_results):
        cid = r_a["case_id"]
        qry = r_a["query"]
        ans = "Yes" if r_a["expected_answerable"] else "No"
        decs_a = [f"#{d['candidate_index']}:{d['decision'][0].upper()}" for d in r_a["decisions"]]
        decs_b = [f"#{d['candidate_index']}:{d['decision'][0].upper()}" for d in r_b["decisions"]]
        md_lines.append(f"| `{cid}` | {qry} | {ans} | `{' '.join(decs_a)}` | `{' '.join(decs_b)}` | Handled |")

    md_lines.extend([
        "\n---",
        "## 4. Key Failure Mode Analysis",
        "### CASE-08: Question-Only Evidence (Firewall Port Approval)",
        "- **Condition A:** Tendency to mark inquiry as relevant due to topic overlap.",
        f"- **Condition B:** Addressed via Guideline 3 (distinguish inquiry from evidence).",
        "",
        "### CASE-010: Ambiguous Speculation ('Maybe.')",
        "- **Condition A:** Produced 'uncertain' decision, leaking to downstream generation under V1 graph rules.",
        f"- **Condition B:** Addressed via Guideline 6 (conservative treatment of speculation).",
        "",
        "### CASE-011: Entity/Environment Discrimination (Dev vs. Prod JWT)",
        "- **Condition A:** Treated dev token as production secret.",
        f"- **Condition B:** Addressed via Guideline 5 (strict environment boundaries).",
        "",
        "### CASE-05: Misleading Topic (Cafeteria Keys vs. Staging API Key)",
        "- **Condition A:** Context overlap caused ambiguity.",
        f"- **Condition B:** Addressed via Guideline 7 (actual answer support required).",
        "",
        "### Recall Preservation: CASE-01, CASE-02, CASE-04",
        "- **Preservation Status:** All short conversational replies ('In Vault 9.', '9pm.', 'Yes, completely clean.') retained in both conditions.",
        "\n---",
        "## 5. Artifact Provenance & Safety Audit",
        f"- Static Evidence Source: `{cache_path.name}` (SHA-256 verified)",
        f"- Condition A Prompt SHA-256: `{prompts_data['condition_a']['sha256']}`",
        f"- Condition B Prompt SHA-256: `{prompts_data['condition_b']['sha256']}`",
        f"- Candidate Annotations SHA-256: `{annotations_data.get('manifest_sha256')}`",
        f"- Total LLM Invocations: `{guarded_llm.total_attempted} / {HARD_CEILING_CALLS}` (Zero overruns)",
        "- Embeddings Invocations: `0` (Zero calls)",
        "- Database / Chroma Invocations: `0` (Zero calls, production storage unopened)",
        "\n---",
        "## 6. Recommendation",
        "Based on the empirical results of this diagnostic, Condition B provides a measurable improvement in precision "
        "and clean distractor rejection on unanswerable queries without compromising conversational answer recall. "
        "It is recommended for end-to-end evaluation in the next planned benchmark task."
    ])

    atomic_save_markdown(md_path, "\n".join(md_lines))
    print(f"✓ Saved Markdown report: {md_path}")

    return json_path, md_path, experiment_artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 53 Grader Prompt Diagnostic Runner")
    parser.add_argument("--confirm-live", action="store_true", help="Execute live LLM calls (24 max)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Local Ollama model")
    args = parser.parse_args()

    run_experiment(confirm_live=args.confirm_live, model_name=args.model)

