import pytest
from app import get_citation_diagnostics


def test_ui_citation_diagnostics_fully_verified():
    result = {
        "answer": "The meeting starts at 10am [ID 10].",
        "grounded": True,
        "scope": "specific",
        "valid_citation_ids": [10],
        "invalid_citation_ids": [],
        "citation_validation": {
            "has_citations": True,
            "all_cited_valid": True,
            "cited_ids": [10],
            "valid_cited_ids": [10],
            "invalid_cited_ids": [],
            "malformed_citations": [],
            "missing_citations": False,
        },
    }
    diag = get_citation_diagnostics(result)
    assert diag["is_grounded"] is True
    assert diag["has_invalid_citations"] is False
    assert diag["invalid_ids"] == []
    assert diag["valid_ids"] == [10]
    assert diag["warning_message"] is None


def test_ui_citation_diagnostics_invalid_citations_warning():
    result = {
        "answer": "The meeting starts at 10am [ID 999].",
        "grounded": False,
        "scope": "specific",
        "valid_citation_ids": [],
        "invalid_citation_ids": [999],
        "citation_validation": {
            "has_citations": True,
            "all_cited_valid": False,
            "cited_ids": [999],
            "valid_cited_ids": [],
            "invalid_cited_ids": [999],
            "malformed_citations": [],
            "missing_citations": False,
        },
    }
    diag = get_citation_diagnostics(result)
    assert diag["is_grounded"] is False
    assert diag["has_invalid_citations"] is True
    assert diag["invalid_ids"] == [999]
    assert diag["warning_message"] is not None
    assert "Citation Integrity Warning" in diag["warning_message"]
    assert "[ID 999]" in diag["warning_message"]


def test_ui_citation_diagnostics_malformed_and_invalid_citations_listed():
    result = {
        "answer": "The meeting starts at 10am [ID 50] [ID chunk_abc].",
        "grounded": False,
        "scope": "specific",
        "valid_citation_ids": [],
        "invalid_citation_ids": [50],
        "citation_validation": {
            "has_citations": True,
            "all_cited_valid": False,
            "cited_ids": [50],
            "valid_cited_ids": [],
            "invalid_cited_ids": [50],
            "malformed_citations": ["[ID chunk_abc]"],
            "missing_citations": False,
        },
    }
    diag = get_citation_diagnostics(result)
    assert diag["is_grounded"] is False
    assert diag["has_invalid_citations"] is True
    assert diag["invalid_ids"] == [50]
    assert diag["malformed_refs"] == ["[ID chunk_abc]"]
    assert "[ID 50]" in diag["warning_message"]
    assert "[ID chunk_abc]" in diag["warning_message"]


def test_ui_citation_diagnostics_missing_citations_specific_scope():
    result = {
        "answer": "The meeting is at 10am.",
        "grounded": True,
        "scope": "specific",
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "citation_validation": {
            "has_citations": False,
            "all_cited_valid": False,
            "cited_ids": [],
            "valid_cited_ids": [],
            "invalid_cited_ids": [],
            "malformed_citations": [],
            "missing_citations": True,
        },
    }
    diag = get_citation_diagnostics(result)
    assert diag["missing_citations"] is True
    assert diag["warning_message"] is not None
    assert "Missing Citations" in diag["warning_message"]


def test_ui_citation_diagnostics_broad_scope_no_warning():
    result = {
        "answer": "Participants exchanged 120 messages over 2 weeks.",
        "grounded": True,
        "scope": "broad",
        "valid_citation_ids": [],
        "invalid_citation_ids": [],
        "citation_validation": {
            "has_citations": False,
            "all_cited_valid": False,
            "cited_ids": [],
            "valid_cited_ids": [],
            "invalid_cited_ids": [],
            "malformed_citations": [],
            "missing_citations": False,
        },
    }
    diag = get_citation_diagnostics(result)
    assert diag["is_grounded"] is True
    assert diag["missing_citations"] is False
    assert diag["warning_message"] is None


def test_ui_citation_diagnostics_not_verified_when_grounded_false():
    # Grounded is False even with no invalid citations (e.g. semantic check failed)
    result = {
        "answer": "The meeting is at 10am [ID 10].",
        "grounded": False,
        "scope": "specific",
        "valid_citation_ids": [10],
        "invalid_citation_ids": [],
        "citation_validation": {
            "has_citations": True,
            "all_cited_valid": True,
            "cited_ids": [10],
            "valid_cited_ids": [10],
            "invalid_cited_ids": [],
            "malformed_citations": [],
            "missing_citations": False,
        },
    }
    diag = get_citation_diagnostics(result)
    assert diag["is_grounded"] is False


def test_ui_citation_diagnostics_not_verified_when_invalid_citations_even_if_grounded_true():
    # Defensive test: if older result had grounded=True but invalid IDs exist
    result = {
        "answer": "The meeting is at 10am [ID 999].",
        "grounded": True,
        "scope": "specific",
        "invalid_citation_ids": [999],
    }
    diag = get_citation_diagnostics(result)
    assert diag["is_grounded"] is False
    assert "[ID 999]" in diag["warning_message"]


def test_ui_citation_diagnostics_legacy_and_empty_results_safe():
    # Handles None, empty dict, or older result dictionaries safely without crashing
    diag_none = get_citation_diagnostics(None)
    assert diag_none["is_grounded"] is False
    assert diag_none["warning_message"] is None

    diag_empty = get_citation_diagnostics({})
    assert diag_empty["is_grounded"] is False
    assert diag_empty["warning_message"] is None

    legacy_result = {
        "answer": "Legacy answer",
        "grounded": True,
        "scope": "specific",
        "sources": [],
    }
    diag_legacy = get_citation_diagnostics(legacy_result)
    assert diag_legacy["is_grounded"] is True
    assert diag_legacy["warning_message"] is None
