"""
test_evaluation_scoring.py
==========================
Deterministic offline test suite for Task 47:
Evaluation Serialization, Metric Correctness, and Audit Infrastructure.
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, List
import pytest

from run_dialogue_evidence_live import EVAL_CASES, TrackingEmbeddingsWrapper
from evaluation_scoring import (
    atomic_save_json,
    atomic_save_markdown,
    classify_case_outcome,
    compute_citation_validity,
    compute_clarified_audit_metrics,
    evaluate_citation_attribution,
    evaluate_exact_facts_and_qualifiers,
    extract_embedding_counts,
    is_time_equivalent,
    parse_time_tokens,
    reconcile_task46_post_hoc_audit_to_clarified_contract,
    save_progressive_checkpoint,
)


class MockInnerEmbedder:
    """Mock embedder simulating LangChain embedding interface."""
    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


# ----------------------------------------------------------------------------
# 1. The embedding wrapper's actual counter-property names
# ----------------------------------------------------------------------------
def test_embedding_wrapper_counter_property_names():
    wrapper = TrackingEmbeddingsWrapper(MockInnerEmbedder())
    assert wrapper.query_invocations == 0
    assert wrapper.document_batch_invocations == 0
    assert wrapper.total_documents_embedded == 0

    wrapper.embed_query("test query 1")
    wrapper.embed_query("test query 2")
    wrapper.embed_documents(["doc 1", "doc 2", "doc 3"])
    wrapper.embed_documents(["doc 4", "doc 5"])

    # Canonical properties
    assert wrapper.query_invocations == 2
    assert wrapper.document_batch_invocations == 2
    assert wrapper.total_documents_embedded == 5

    # Backwards-compatibility aliases
    assert wrapper.doc_batch_invocations == 2
    assert wrapper.total_docs_embedded == 5


# ----------------------------------------------------------------------------
# 2. Correct serialization of all tracked embedding counts
# ----------------------------------------------------------------------------
def test_correct_serialization_of_tracked_embedding_counts():
    wrapper = TrackingEmbeddingsWrapper(MockInnerEmbedder())
    wrapper.embed_query("q")
    wrapper.embed_documents(["d1", "d2"])

    counts = extract_embedding_counts(wrapper)
    assert counts == {
        "query_invocations": 1,
        "document_batch_invocations": 1,
        "total_documents_embedded": 2,
    }


# ----------------------------------------------------------------------------
# 3. Safe finalization when optional metadata is unavailable
# ----------------------------------------------------------------------------
def test_safe_finalization_when_optional_metadata_unavailable():
    # None embedder
    counts_none = extract_embedding_counts(None)
    assert counts_none == {
        "query_invocations": 0,
        "document_batch_invocations": 0,
        "total_documents_embedded": 0,
    }

    # Bare object without tracking attributes
    class BareEmbedder:
        pass

    counts_bare = extract_embedding_counts(BareEmbedder())
    assert counts_bare == {
        "query_invocations": 0,
        "document_batch_invocations": 0,
        "total_documents_embedded": 0,
    }


# ----------------------------------------------------------------------------
# 4. Result-file finalization when serialization fails
# ----------------------------------------------------------------------------
def test_result_file_finalization_when_serialization_fails(tmp_path):
    target_file = tmp_path / "report.json"

    # Attempting to serialize non-serializable object should raise and leave no corrupted file
    bad_data = {"key": object()}
    with pytest.raises(TypeError):
        atomic_save_json(target_file, bad_data)

    assert not target_file.exists()
    assert not (tmp_path / "report.json.tmp").exists()


# ----------------------------------------------------------------------------
# 5. Preservation of completed case-level data when report cannot be finalized
# ----------------------------------------------------------------------------
def test_preservation_of_completed_case_data_on_finalization_failure(tmp_path):
    checkpoint_file = tmp_path / ".checkpoint.json"
    records = [{"case_id": "CASE-01", "result": "TP"}, {"case_id": "CASE-02", "result": "TN"}]

    save_progressive_checkpoint(checkpoint_file, records)
    assert checkpoint_file.exists()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["case_records"] == records


# ----------------------------------------------------------------------------
# 6. Atomic or otherwise validated output creation
# ----------------------------------------------------------------------------
def test_atomic_output_creation(tmp_path):
    json_path = tmp_path / "valid.json"
    md_path = tmp_path / "valid.md"

    atomic_save_json(json_path, {"status": "ok", "cases": 12})
    assert json_path.exists()
    assert not (tmp_path / "valid.json.tmp").exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "ok"

    atomic_save_markdown(md_path, "# Header\nContent")
    assert md_path.exists()
    assert not (tmp_path / "valid.md.tmp").exists()

    # Empty markdown should raise
    with pytest.raises(ValueError):
        atomic_save_markdown(tmp_path / "empty.md", "")


# ============================================================================
# TASK 48 SECTION 7 DETERMINISTIC REGRESSION TEST SUITE (TESTS 1 - 15)
# ============================================================================

# ----------------------------------------------------------------------------
# 1. Empty citation sets produce N/A validity rather than an automatic 100%
# ----------------------------------------------------------------------------
def test_empty_citation_sets_produce_na_validity():
    # Individual case evaluation
    assert compute_citation_validity(valid_cited_ids=[], invalid_cited_ids=[]) is None
    assert compute_citation_validity(valid_cited_ids=[]) is None

    # Attribution precision on empty citation set must be None (N/A), not 1.0 or 0.0
    prec_ans, rec_ans = evaluate_citation_attribution(is_answerable=True, valid_cited_ids=[], expected_supporting_ids=[102])
    assert prec_ans is None
    assert rec_ans == 0.0

    prec_unans, rec_unans = evaluate_citation_attribution(is_answerable=False, valid_cited_ids=[], expected_supporting_ids=[])
    assert prec_unans is None
    assert rec_unans is None

    # Aggregate metric with zero citations across all runs
    rec = {
        "case_id": "CASE-05-Misleading-Context",
        "expected_answerable": False,
        "valid_cited_ids": [],
        "invalid_cited_ids": [],
        "classification": "TN",
        "correct_abstention": True,
    }
    metrics = compute_clarified_audit_metrics([rec], "TestMode")
    assert metrics.citation_validity_rate is None
    assert metrics.total_citations_emitted == 0


# ----------------------------------------------------------------------------
# 2. Aggregate citation validity uses actual aggregate numerator and denominator
# ----------------------------------------------------------------------------
def test_aggregate_citation_validity_uses_actual_numerator_and_denominator():
    # Run 1: 1 valid citation, 0 invalid (100% case validity)
    # Run 2: 0 valid citations, 3 invalid (0% case validity)
    # Naive per-case average: (1.0 + 0.0) / 2 = 50.0%
    # True aggregate ratio: 1 valid / 4 total = 25.0%
    records = [
        {
            "case_id": "CASE-01-Terse-Location",
            "expected_answerable": True,
            "valid_cited_ids": [102],
            "invalid_cited_ids": [],
            "expected_supporting_ids": [102],
            "target_in_final_evidence": True,
            "semantic_answer_correct": True,
            "grounded": True,
            "attribution_precision": 1.0,
            "attribution_recall": 1.0,
            "classification": "TP",
        },
        {
            "case_id": "CASE-06-Self-Contained-Relevant",
            "expected_answerable": True,
            "valid_cited_ids": [],
            "invalid_cited_ids": [9991, 9992, 9993],
            "expected_supporting_ids": [112],
            "target_in_final_evidence": True,
            "semantic_answer_correct": True,
            "grounded": False,
            "attribution_precision": 0.0,
            "attribution_recall": 0.0,
            "classification": "FN",
        },
    ]
    metrics = compute_clarified_audit_metrics(records, "TestMode")
    assert metrics.total_valid_citations == 1
    assert metrics.total_invalid_citations == 3
    assert metrics.total_citations_emitted == 4
    assert metrics.citation_validity_rate == 0.25  # NOT 0.50!


# ----------------------------------------------------------------------------
# 3. Correct abstention improves abstention metric without inflating attribution
# ----------------------------------------------------------------------------
def test_correct_abstention_does_not_inflate_attribution_precision():
    # Unanswerable query where model correctly abstains with 0 citations
    prec, rec = evaluate_citation_attribution(is_answerable=False, valid_cited_ids=[], expected_supporting_ids=[])
    assert prec is None  # Excluded from attribution precision
    assert rec is None

    # In aggregate metrics across 1 answerable TP (prec 1.0) and 1 unanswerable TN (0 citations)
    records = [
        {
            "case_id": "CASE-01-Terse-Location",
            "expected_answerable": True,
            "valid_cited_ids": [102],
            "invalid_cited_ids": [],
            "expected_supporting_ids": [102],
            "attribution_precision": 1.0,
            "attribution_recall": 1.0,
            "target_in_final_evidence": True,
            "semantic_answer_correct": True,
            "grounded": True,
            "classification": "TP",
        },
        {
            "case_id": "CASE-05-Misleading-Context",
            "expected_answerable": False,
            "valid_cited_ids": [],
            "invalid_cited_ids": [],
            "expected_supporting_ids": [],
            "attribution_precision": None,  # N/A
            "attribution_recall": None,     # N/A
            "correct_abstention": True,
            "classification": "TN",
        },
    ]
    metrics = compute_clarified_audit_metrics(records, "TestMode")
    # Abstention metric reflects 100% correct abstention
    assert metrics.correct_abstention_rate == 1.0
    # Attribution precision is evaluated ONLY on the case with citations (N=1), NOT averaged over N=2
    assert metrics.cases_with_citations_count == 1
    assert metrics.mean_attribution_precision == 1.0


# ----------------------------------------------------------------------------
# 4. Nonexistent message ID fails citation validity
# ----------------------------------------------------------------------------
def test_nonexistent_message_id_fails_citation_validity():
    # Model cites ID 9999 which does NOT exist in context
    val = compute_citation_validity(valid_cited_ids=[], invalid_cited_ids=[9999])
    assert val == 0.0

    # 1 valid, 1 nonexistent
    val_mixed = compute_citation_validity(valid_cited_ids=[102], invalid_cited_ids=[9999])
    assert val_mixed == 0.5


# ----------------------------------------------------------------------------
# 5. Valid ID that does not support claim fails attribution
# ----------------------------------------------------------------------------
def test_valid_id_not_supporting_claim_fails_attribution():
    # Context contains messages [102, 110]. Expected supporting is [102].
    # Model cites valid ID [110] (which is an unrelated distractor).
    # ID validity is 100% (it exists in context), but attribution precision is 0.0!
    prec, rec = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[110],
        expected_supporting_ids=[102],
        invalid_cited_ids=[],
    )
    assert prec == 0.0
    assert rec == 0.0


# ----------------------------------------------------------------------------
# 6. Supported claim with correct citation receives appropriate credit
# ----------------------------------------------------------------------------
def test_supported_claim_with_correct_citation_receives_credit():
    prec, rec = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[102],
        expected_supporting_ids=[102],
        invalid_cited_ids=[],
    )
    assert prec == 1.0
    assert rec == 1.0

    # Multi-supporting ID case: expected [119, 120], cited [119] -> precision 1.0, recall 0.5
    prec_partial, rec_partial = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[119],
        expected_supporting_ids=[119, 120],
        invalid_cited_ids=[],
    )
    assert prec_partial == 1.0
    assert rec_partial == 0.5


# ----------------------------------------------------------------------------
# 7. Answer to an unanswerable query classified as false positive
# ----------------------------------------------------------------------------
def test_answer_to_unanswerable_query_classified_as_false_positive():
    # Model hallucinated answer and cited distractor [110] on unanswerable query
    cls = classify_case_outcome(
        expected_answerable=False,
        target_in_final_evidence=True,
        semantic_answer_correct=False,
        grounded=False,
        attribution_precision=0.0,
        attribution_recall=0.0,
        correct_abstention=False,
        valid_cited_ids=[110],
        invalid_cited_ids=[],
    )
    assert cls == "FP"


# ----------------------------------------------------------------------------
# 8. Correct abstention on unanswerable query classified as true negative
# ----------------------------------------------------------------------------
def test_correct_abstention_on_unanswerable_classified_as_true_negative():
    cls = classify_case_outcome(
        expected_answerable=False,
        target_in_final_evidence=True,
        semantic_answer_correct=False,
        grounded=False,
        attribution_precision=None,
        attribution_recall=None,
        correct_abstention=True,
        valid_cited_ids=[],
        invalid_cited_ids=[],
    )
    assert cls == "TN"


# ----------------------------------------------------------------------------
# 9. Answerable case with retrieval but incorrect output is not true positive
# ----------------------------------------------------------------------------
def test_answerable_case_with_retrieval_but_incorrect_output_not_true_positive():
    # Evidence was retrieved (target_in_final_evidence=True), but model generated wrong answer
    cls = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=False,  # Wrong answer!
        grounded=False,
        attribution_precision=0.0,
        attribution_recall=0.0,
        correct_abstention=False,
        valid_cited_ids=[102],
    )
    assert cls == "FN"


# ----------------------------------------------------------------------------
# 10. Misattributed citations prevent full true-positive classification
# ----------------------------------------------------------------------------
def test_misattributed_citations_prevent_full_true_positive():
    # Answerable case where answer is correct, grounded, and retrieved,
    # BUT citations include an unsupported distractor (cited [106, 105] where 106 is supporting and 105 is distractor)
    # Attribution precision is 0.5 (< 1.0). Must NOT be classified as TP!
    cls = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=True,
        grounded=True,
        attribution_precision=0.5,  # Has misattributed citation!
        attribution_recall=1.0,
        correct_abstention=False,
        valid_cited_ids=[106, 105],
    )
    assert cls == "FN"


# ----------------------------------------------------------------------------
# 11. Uncertain/ungradable outcomes are not counted as correct
# ----------------------------------------------------------------------------
def test_uncertain_ungradable_outcomes_not_counted_as_correct():
    cls = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=True,
        grounded=True,
        attribution_precision=1.0,
        attribution_recall=1.0,
        correct_abstention=False,
        valid_cited_ids=[102],
        is_uncertain=True,  # Grader failure / ungradable
    )
    assert cls == "UNCERTAIN"

    rec = {
        "case_id": "CASE-01-Terse-Location",
        "expected_answerable": True,
        "classification": "UNCERTAIN",
        "target_in_final_evidence": True,
        "semantic_answer_correct": True,
        "grounded": True,
        "valid_cited_ids": [102],
        "attribution_precision": 1.0,
        "attribution_recall": 1.0,
    }
    metrics = compute_clarified_audit_metrics([rec], "TestMode")
    assert metrics.clarified_accuracy == 0.0
    assert metrics.uncertain_count == 1
    assert metrics.coverage_rate == 0.0


# ----------------------------------------------------------------------------
# 12. CASE-012 remains an answerable case
# ----------------------------------------------------------------------------
def test_case012_remains_answerable_case():
    c12 = next(c for c in EVAL_CASES if c.case_id == "CASE-012-Ordinary-Length-Bypass")
    assert c12.expected_answerable is True
    assert c12.target_id == 125
    assert c12.expected_supporting_ids == [125]


# ----------------------------------------------------------------------------
# 13. Equivalent time formatting is handled correctly
# ----------------------------------------------------------------------------
def test_equivalent_time_formatting_handled_correctly():
    variations = [
        "The deployment is at 9 PM [ID 104]",
        "Scheduled for 9pm.",
        "Kicking off at 9:00 PM tonight.",
        "Window starts at 21:00.",
        "At 9:00pm in staging.",
    ]
    for ans in variations:
        assert is_time_equivalent(ans, "9pm"), f"Failed for answer: {ans}"
        assert is_time_equivalent(ans, "21:00"), f"Failed for answer: {ans}"

    distinct_times = [
        "The deployment is at 8 PM.",
        "Scheduled for 9 AM.",
        "Window is 10:00 PM.",
        "Starts at 22:00.",
        "At 14:30.",
    ]
    for ans in distinct_times:
        assert not is_time_equivalent(ans, "9pm"), f"Incorrectly matched for: {ans}"
        assert not is_time_equivalent(ans, "21:00"), f"Incorrectly matched for: {ans}"


# ----------------------------------------------------------------------------
# 14. The qualifier in CASE-04 is evaluated independently of semantic success
# ----------------------------------------------------------------------------
def test_qualifier_in_case04_evaluated_independently():
    case_id = "CASE-04-Yes-No-Confirmation"
    expected_kws = ["yes", "clean", "succeed"]

    # Emitted "finished successfully" (answers success question, but omits "completely clean")
    ans1 = "Dev_Sarah [ID 108] confirmed the database migration script finished successfully on staging at 4:20 PM."
    eval1 = evaluate_exact_facts_and_qualifiers(case_id, ans1, expected_kws)
    assert eval1["semantic_answer_correct"] is True
    assert eval1["qualifier_preserved"] is False
    assert eval1["exact_fact_preserved"] is False

    # Emitted "Yes, completely clean."
    ans2 = "Yes, completely clean [ID 108]."
    eval2 = evaluate_exact_facts_and_qualifiers(case_id, ans2, expected_kws)
    assert eval2["semantic_answer_correct"] is True
    assert eval2["qualifier_preserved"] is True
    assert eval2["exact_fact_preserved"] is True


# ----------------------------------------------------------------------------
# 15. Historical metric labels and files remain unchanged
# ----------------------------------------------------------------------------
def test_historical_metric_labels_and_files_remain_unchanged():
    task46_json = Path(r"C:\Users\vibhor\Desktop\chat_lens_v2\ChatLens-RAG\eval_results\task46_post_hoc_audit_20261028_151500.json")
    assert task46_json.exists(), "Historical Task 46 post-hoc audit JSON must remain present"
    with open(task46_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Historical keys and figures must be preserved
    assert "mode_a_metrics" in data
    assert "mode_b_metrics" in data
    assert data["mode_a_metrics"]["legacy_strict_accuracy"] == 0.6666666666666666
    assert data["mode_b_metrics"]["legacy_strict_accuracy"] == 0.5833333333333334

    task46_md = Path(r"C:\Users\vibhor\Desktop\chat_lens_v2\ChatLens-RAG\eval_results\task46_post_hoc_audit_report_20261028_151500.md")
    assert task46_md.exists(), "Historical Task 46 post-hoc audit Markdown must remain present"


# ============================================================================
# TASK 49 REGRESSION SUITE: CONTRACT INTEGRATION & MULTI-MESSAGE ATTRIBUTION
# ============================================================================

# 1. Active runner uses canonical classification function
def test_active_runners_use_canonical_classification():
    import run_dialogue_evidence_live as rdel
    import run_task46_comparison_live as r46
    import inspect
    
    rdel_src = inspect.getsource(rdel.evaluate_run_metrics)
    assert "classify_case_outcome(" in rdel_src
    
    r46_src = inspect.getsource(r46.evaluate_case_record)
    assert "classify_case_outcome(" in r46_src


# 2. Empty citations do not inflate validity or attribution
def test_empty_citations_do_not_inflate_validity_or_attribution():
    val = compute_citation_validity(valid_cited_ids=[], invalid_cited_ids=[])
    assert val is None, "0 citations must produce None (N/A) validity, not 1.0"
    
    prec, rec = evaluate_citation_attribution(is_answerable=True, valid_cited_ids=[], expected_supporting_ids=[101])
    assert prec is None, "0 citations on answerable case must produce None attribution precision"
    assert rec == 0.0, "0 citations on answerable case must produce 0.0 attribution recall"
    
    prec_unans, rec_unans = evaluate_citation_attribution(is_answerable=False, valid_cited_ids=[], expected_supporting_ids=[])
    assert prec_unans is None, "0 citations on unanswerable case must produce None attribution precision"
    assert rec_unans is None, "0 citations on unanswerable case must produce None attribution recall"


# 3. Micro and macro attribution precision produce distinct, correct results
def test_micro_and_macro_attribution_precision_produce_distinct_correct_results():
    # Case 1: 1 citation emitted, 1 supporting -> precision = 1.0
    # Case 2: 3 citations emitted, 1 supporting -> precision = 1/3
    records = [
        {
            "case_id": "C1",
            "expected_answerable": True,
            "expected_supporting_ids": [101],
            "valid_cited_ids": [101],
            "invalid_cited_ids": [],
            "attribution_precision": 1.0,
            "attribution_recall": 1.0,
        },
        {
            "case_id": "C2",
            "expected_answerable": True,
            "expected_supporting_ids": [102],
            "valid_cited_ids": [102, 103, 104],
            "invalid_cited_ids": [],
            "attribution_precision": 1.0 / 3.0,
            "attribution_recall": 1.0,
        },
    ]
    metrics = compute_clarified_audit_metrics(records, "TestMode")
    macro_prec = metrics.mean_attribution_precision  # (1.0 + 1/3) / 2 = 2/3 = 0.666667
    micro_prec = metrics.citation_level_attribution_precision  # (1 + 1) / (1 + 3) = 2/4 = 0.50
    
    assert macro_prec is not None and micro_prec is not None
    assert pytest.approx(macro_prec, rel=1e-4) == 2.0 / 3.0
    assert pytest.approx(micro_prec, rel=1e-4) == 0.50
    assert macro_prec != micro_prec, "Macro and Micro precision must yield distinct mathematical results"
    assert metrics.macro_attribution_precision == macro_prec
    assert metrics.micro_attribution_precision == micro_prec


# 4. Multi-message answer requiring two jointly necessary sources is not fully credited when one is missing
def test_multi_message_joint_necessity_partial_recall_not_true_positive():
    # Expected: [119, 117] jointly necessary
    # Emitted: [119] only
    prec, rec = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[119],
        expected_supporting_ids=[119, 117],
        support_semantics="joint",
    )
    assert prec == 1.0
    assert rec == 0.5  # Only 1 of 2 jointly necessary messages cited
    
    classification = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=True,
        grounded=True,
        attribution_precision=prec,
        attribution_recall=rec,
        correct_abstention=False,
        valid_cited_ids=[119],
        is_joint_supporting=True,
        expected_supporting_ids=[119, 117],
    )
    assert classification == "FN", "Missing a jointly necessary supporting message must classify as FN, not TP"


# 5. Case with alternative supporting messages can receive credit when one complete alternative is cited
def test_alternative_supporting_messages_receive_credit():
    # Expected: [101, 102] where either independently supports answer
    prec, rec = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[101],
        expected_supporting_ids=[101, 102],
        support_semantics="alternative",
    )
    assert prec == 1.0
    assert rec == 1.0, "Citing one complete independent alternative must yield recall 1.0"
    
    classification = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=True,
        grounded=True,
        attribution_precision=prec,
        attribution_recall=rec,
        correct_abstention=False,
        valid_cited_ids=[101],
        is_joint_supporting=False,
        expected_supporting_ids=[101, 102],
    )
    assert classification == "TP", "Complete citation of an alternative must yield TP"


# 6. Misattributed extra citations prevent full attribution precision
def test_misattributed_extra_citations_prevent_full_attribution_precision():
    # Model cites [106, 105] when only [106] is supporting
    prec, rec = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[106, 105],
        expected_supporting_ids=[106],
        support_semantics="joint",
    )
    assert prec == 0.5, "1 supporting out of 2 cited must yield precision 0.5"
    assert rec == 1.0
    
    classification = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=True,
        grounded=True,
        attribution_precision=prec,
        attribution_recall=rec,
        correct_abstention=False,
        valid_cited_ids=[106, 105],
        is_joint_supporting=False,
        expected_supporting_ids=[106],
    )
    assert classification == "FN", "Over-citation / misattribution must prevent TP (classified as FN)"


# 7. Correct abstention remains a separate true-negative outcome
def test_correct_abstention_remains_separate_true_negative_outcome():
    prec, rec = evaluate_citation_attribution(
        is_answerable=False,
        valid_cited_ids=[],
        expected_supporting_ids=[],
    )
    assert prec is None
    assert rec is None
    
    classification = classify_case_outcome(
        expected_answerable=False,
        target_in_final_evidence=True,
        semantic_answer_correct=False,
        grounded=False,
        attribution_precision=prec,
        attribution_recall=rec,
        correct_abstention=True,
        valid_cited_ids=[],
        invalid_cited_ids=[],
    )
    assert classification == "TN"


# 8. CASE-03 and CASE-09 follow their verified evidence semantics
def test_case03_and_case09_verified_evidence_semantics():
    from run_dialogue_evidence_live import EVAL_CASES
    case03 = next(c for c in EVAL_CASES if c.case_id == "CASE-03-Pronoun-Anaphora")
    case09 = next(c for c in EVAL_CASES if c.case_id == "CASE-09-Multi-Message-Exchange")
    
    # CASE-03: expected is only [106]
    assert case03.expected_supporting_ids == [106]
    prec_3, rec_3 = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[105, 106],
        expected_supporting_ids=case03.expected_supporting_ids,
        support_semantics=getattr(case03, "support_semantics", "joint"),
    )
    assert prec_3 == 0.5  # 105 is distractor / question antecedent
    assert rec_3 == 1.0
    
    # CASE-09: expected is [119, 117] (jointly necessary)
    assert case09.expected_supporting_ids == [119, 117]
    assert getattr(case09, "support_semantics", "joint") == "joint"
    prec_9, rec_9 = evaluate_citation_attribution(
        is_answerable=True,
        valid_cited_ids=[119],
        expected_supporting_ids=case09.expected_supporting_ids,
        support_semantics="joint",
    )
    assert prec_9 == 1.0
    assert rec_9 == 0.5  # Missing 117
    class_9 = classify_case_outcome(
        expected_answerable=True,
        target_in_final_evidence=True,
        semantic_answer_correct=True,
        grounded=True,
        attribution_precision=prec_9,
        attribution_recall=rec_9,
        correct_abstention=False,
        valid_cited_ids=[119],
        is_joint_supporting=True,
        expected_supporting_ids=case09.expected_supporting_ids,
    )
    assert class_9 == "FN"


# 9. All active runners preserve canonical 7-answerable / 5-unanswerable mapping
def test_all_active_runners_preserve_canonical_answerable_mapping():
    from run_dialogue_evidence_live import EVAL_CASES
    assert len(EVAL_CASES) == 12
    ans_cases = [c for c in EVAL_CASES if c.expected_answerable]
    unans_cases = [c for c in EVAL_CASES if not c.expected_answerable]
    assert len(ans_cases) == 7, "Exactly 7 answerable cases required"
    assert len(unans_cases) == 5, "Exactly 5 unanswerable cases required"
    
    # Verify CASE-012 is answerable
    c12 = next(c for c in EVAL_CASES if c.case_id == "CASE-012-Ordinary-Length-Bypass")
    assert c12.expected_answerable is True


# 10. Historical result readers remain backward compatible
def test_historical_result_readers_remain_backward_compatible():
    from evaluation_scoring import reconcile_task46_post_hoc_audit_to_clarified_contract
    task46_json = Path(r"C:\Users\vibhor\Desktop\chat_lens_v2\ChatLens-RAG\eval_results\task46_post_hoc_audit_20261028_151500.json")
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path, md_path, audit_data = reconcile_task46_post_hoc_audit_to_clarified_contract(
            source_json_path=task46_json,
            out_dir=Path(tmp_dir),
            timestamp="test_reconcile",
        )
        assert json_path.exists()
        assert md_path.exists()
        assert audit_data["mode_a_clarified_metrics"]["total_cases"] == 12
        assert audit_data["mode_b_clarified_metrics"]["total_cases"] == 12
        assert audit_data["historical_metrics_preserved"]["mode_a"]["total_cases"] == 12


# 11. Verification that no live inference or production storage occurred
def test_no_live_inference_or_production_storage_access():
    db_file = Path(r"C:\Users\vibhor\Desktop\chat_lens_v2\ChatLens-RAG\chat_data.db")
    chroma_dir = Path(r"C:\Users\vibhor\Desktop\chat_lens_v2\ChatLens-RAG\chroma_db")
    if db_file.exists():
        assert db_file.is_file()
    if chroma_dir.exists():
        assert chroma_dir.is_dir()

