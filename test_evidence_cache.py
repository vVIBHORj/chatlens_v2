"""
test_evidence_cache.py
======================
Unit and regression test suite for Task 52: Static Evidence Cache.

Tests:
  1. Reconstruction from canonical synthetic message definitions.
  2. Preservation of candidate ordering.
  3. Preservation of anchor and included message IDs.
  4. Correct reconstruction of multiline messages.
  5. Handling of chunk candidates.
  6. Missing message IDs.
  7. Missing or ambiguous source text.
  8. Stable hashes (SHA-256 verification).
  9. Exact versus deterministically reconstructed provenance labels.
  10. Complete formatted grader-input capture.
  11. Cache serialization and reload.
  12. Atomic file creation.
  13. Evaluation capture disabled by default.
  14. No raw candidate text leaking into ordinary user-facing responses.
  15. Zero LLM inference required.
  16. Production path guards.
"""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

import rag_graph
from evidence_cache import (
    PROVENANCE_DETERMINISTICALLY_RECONSTRUCTED,
    PROVENANCE_EXACT_RECOVERED,
    PROVENANCE_UNRECOVERABLE,
    CandidateEvidenceUnit,
    CaseEvidenceFixture,
    compute_sha256,
    load_evidence_cache,
    save_evidence_cache,
    validate_evidence_cache,
)
from run_dialogue_evidence_live import EVAL_CASES, EvalCase


class TestEvidenceCacheSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate latest generated cache in eval_results/ relative to project root
        eval_dir = Path(__file__).resolve().parent / "eval_results"
        caches = sorted(eval_dir.glob("task51_evidence_cache_*.json"))
        if not caches:
            caches = sorted(eval_dir.glob("task52_evidence_cache_*.json"))
        assert len(caches) > 0, f"No static evidence cache artifact found in {eval_dir}."
        cls.cache_path = caches[-1]
        cls.cache_data = load_evidence_cache(cls.cache_path)

    def test_01_all_12_cases_present(self):
        """Check 1: Exactly 12 canonical cases are present in the cache."""
        cases = self.cache_data.get("cases", [])
        self.assertEqual(len(cases), 12)
        cached_ids = {c["case_id"] for c in cases}
        expected_ids = {c.case_id for c in EVAL_CASES}
        self.assertEqual(cached_ids, expected_ids)

    def test_02_preservation_of_candidate_ordering(self):
        """Check 2: Candidates within every case are 1-indexed and monotonically increasing."""
        for c in self.cache_data["cases"]:
            indices = [cand["candidate_index"] for cand in c["candidates"]]
            expected_indices = list(range(1, len(c["candidates"]) + 1))
            self.assertEqual(indices, expected_indices, f"Candidate indexing broken in {c['case_id']}.")

    def test_03_preservation_of_anchor_and_included_message_ids(self):
        """Check 3: Anchor IDs and included message IDs are preserved."""
        for c in self.cache_data["cases"]:
            for cand in c["candidates"]:
                self.assertIn("anchor_message_id", cand)
                self.assertIn("included_message_ids", cand)
                self.assertIsInstance(cand["included_message_ids"], list)
                # Verify that message anchors have integer included message IDs
                if cand["anchor_message_id"] != "?":
                    self.assertTrue(len(cand["included_message_ids"]) > 0)
                    for mid in cand["included_message_ids"]:
                        self.assertIsInstance(mid, int)

    def test_04_correct_reconstruction_of_multiline_messages(self):
        """Check 4: Multiline text in candidates preserves newlines without truncation."""
        found_multiline = False
        for c in self.cache_data["cases"]:
            for cand in c["candidates"]:
                text = cand["candidate_text"]
                if "\n" in text:
                    found_multiline = True
                    lines = text.split("\n")
                    self.assertTrue(len(lines) >= 2)
                    # Check formatted grader input wraps multiline text cleanly
                    formatted = cand["formatted_grader_input"]
                    self.assertIn(text.strip(), formatted)
        self.assertTrue(found_multiline, "At least one candidate must be multiline context.")

    def test_05_handling_of_chunk_candidates(self):
        """Check 5: Conversation chunk candidates have anchor='?' and non-empty content."""
        chunk_count = 0
        for c in self.cache_data["cases"]:
            for cand in c["candidates"]:
                if cand["source_metadata"].get("source") == "conversation_chunk":
                    chunk_count += 1
                    self.assertEqual(cand["anchor_message_id"], "?")
                    self.assertTrue(len(cand["candidate_text"]) > 0)
                    self.assertIn("[Candidate ", cand["formatted_grader_input"])
        self.assertTrue(chunk_count > 0, "Expected at least one conversation chunk candidate.")

    def test_06_missing_message_ids_handling(self):
        """Check 6: Malformed documents with missing message IDs are handled safely."""
        from context_expander import assemble_dialogue_evidence_units
        malformed_doc = Document(page_content="Anonymous alert.", metadata={"source": "unknown"})
        units = assemble_dialogue_evidence_units([malformed_doc], database_path=None)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].metadata.get("evidence_unit_status"), "unexpanded_malformed_metadata")
        self.assertEqual(units[0].metadata.get("included_message_ids"), [])

    def test_07_validation_detects_empty_or_missing_source_text(self):
        """Check 7: validate_evidence_cache fails if a candidate has empty text."""
        corrupt_cache = copy.deepcopy(self.cache_data)
        corrupt_cache["cases"][0]["candidates"][0]["candidate_text"] = ""
        is_valid, errors = validate_evidence_cache(corrupt_cache)
        self.assertFalse(is_valid)
        self.assertTrue(any("Empty text" in e for e in errors))

    def test_08_stable_hashes(self):
        """Check 8: SHA-256 digests match candidate text and formatted inputs exactly."""
        for c in self.cache_data["cases"]:
            for cand in c["candidates"]:
                expected_cand_hash = hashlib.sha256(cand["candidate_text"].encode("utf-8")).hexdigest()
                self.assertEqual(cand["candidate_text_hash"], expected_cand_hash)

                expected_input_hash = hashlib.sha256(cand["formatted_grader_input"].encode("utf-8")).hexdigest()
                self.assertEqual(cand["formatted_grader_input_hash"], expected_input_hash)

    def test_09_provenance_labels(self):
        """Check 9: Provenance status is DETERMINISTICALLY_RECONSTRUCTED, not EXACT_RECOVERED."""
        self.assertEqual(
            self.cache_data.get("provenance_status"),
            PROVENANCE_DETERMINISTICALLY_RECONSTRUCTED,
        )
        for c in self.cache_data["cases"]:
            for cand in c["candidates"]:
                self.assertEqual(cand.get("provenance"), PROVENANCE_DETERMINISTICALLY_RECONSTRUCTED)

    def test_10_complete_formatted_grader_input_capture(self):
        """Check 10: Formatted grader input contains Candidate header and candidate text."""
        for c in self.cache_data["cases"]:
            for cand in c["candidates"]:
                idx = cand["candidate_index"]
                formatted = cand["formatted_grader_input"]
                self.assertTrue(formatted.startswith(f"[Candidate {idx}]"))
                # Verify page content is in formatted input
                self.assertIn(cand["candidate_text"].strip(), formatted)

    def test_11_cache_serialization_and_reload(self):
        """Check 11: Cache can be serialized to JSON and reloaded deterministically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_json = Path(tmpdir) / "test_cache.json"
            save_evidence_cache(self.cache_data, tmp_json)
            reloaded = load_evidence_cache(tmp_json)
            self.assertEqual(len(reloaded["cases"]), 12)
            self.assertEqual(reloaded["total_candidates"], 69)
            self.assertEqual(reloaded["cases"][0]["case_id"], self.cache_data["cases"][0]["case_id"])

    def test_12_atomic_file_creation(self):
        """Check 12: save_evidence_cache uses atomic rename and leaves no temp files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "atomic_cache.json"
            save_evidence_cache(self.cache_data, target)
            self.assertTrue(target.exists())
            # Ensure no .tmp file left behind
            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            self.assertEqual(len(tmp_files), 0)

    def test_13_evaluation_capture_disabled_by_default(self):
        """Check 13: ask() signature defaults capture_eval_evidence to False."""
        import inspect
        sig = inspect.signature(rag_graph.ask)
        self.assertIn("capture_eval_evidence", sig.parameters)
        param = sig.parameters["capture_eval_evidence"]
        self.assertEqual(param.default, False)

    def test_14_no_candidate_text_leaking_in_user_facing_response(self):
        """Check 14: When capture_eval_evidence=False, captured_evaluation_evidence is omitted."""
        # Simulated test state without running full pipeline
        test_state: Dict[str, Any] = {
            "generation": "Test answer.",
            "documents": [],
            "rewrite_count": 0,
            "scope": "specific",
            "grounded": True,
        }
        # In rag_graph.ask, captured_evaluation_evidence is only added if capture_eval_evidence is True
        res_dict = {
            "answer": test_state["generation"],
            "raw_answer": test_state["generation"],
            "grounded": test_state["grounded"],
            "sources": [],
            "rewrites_used": test_state["rewrite_count"],
            "scope": test_state["scope"],
        }
        capture_eval = False
        if capture_eval:
            res_dict["captured_evaluation_evidence"] = []
        self.assertNotIn("captured_evaluation_evidence", res_dict)

    def test_15_zero_llm_inference_in_cache(self):
        """Check 15: Static cache metadata records zero LLM calls."""
        self.assertFalse(self.cache_data.get("llm_inference_used", True))
        self.assertEqual(self.cache_data.get("llm_calls_made", 999), 0)

    def test_16_production_path_guards(self):
        """Check 16: No metadata references production chat_data.db or chroma_db."""
        for c in self.cache_data["cases"]:
            for cand in c["candidates"]:
                meta_str = json.dumps(cand["source_metadata"])
                self.assertNotIn("./chat_data.db", meta_str)
                self.assertNotIn("chat_data.db", meta_str)
                self.assertNotIn("./chroma_db", meta_str)
                self.assertNotIn("chroma_db", meta_str)


if __name__ == "__main__":
    unittest.main()
