"""Offline regression checks for evidence boundaries and content synchronization."""

import json
from copy import deepcopy

import pytest
import sync_practice_2026_10 as sync
import validate_practice_2026_10 as validator


def test_snapshot_passes():
    assert validator.validate() == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("authentic_company_question_verified", True, "must be false"),
        ("question_text_verified_against_primary", True, "must be false"),
        ("question_kind", "official_interview", "must remain an editorial paraphrase"),
        ("answer_status", "official", "answer status changed"),
        ("reference_answer", "", "missing reference_answer"),
        ("answer_key_points", [], "missing/invalid answer_key_points"),
        ("answer_reference_ids", ["MISSING"], "dangling answer reference"),
        ("source_ids", ["MISSING"], "dangling interview source"),
        ("document_path", "字节跳动/面试题与面经.md", "wrong document location"),
        ("answer_code_symbols", ["not_a_real_symbol"], "unknown code symbol"),
    ],
)
def test_rejects_question_corruption(monkeypatch, field, value, message):
    original_load = validator.load

    def changed(path):
        data = deepcopy(original_load(path))
        if path == sync.DATA / "questions.json":
            data[0][field] = value
        return data

    monkeypatch.setattr(validator, "load", changed)
    assert any(message in error for error in validator.validate())


def test_rejects_source_confidence_upgrade(monkeypatch):
    original_load = validator.load

    def changed(path):
        data = deepcopy(original_load(path))
        if path == sync.ROOT / "data/interviews.json":
            next(s for s in data["items"] if s.get("collection_batch") == "2026-10-22")[
                "confidence"
            ] = "high"
        return data

    monkeypatch.setattr(validator, "load", changed)
    assert any("confidence upgraded" in error for error in validator.validate())


def test_rejects_stale_output(monkeypatch):
    original_outputs = validator.outputs

    def changed():
        result = original_outputs()
        result[sync.INDEX] += "\nThis content was not synchronized.\n"
        return result

    monkeypatch.setattr(validator, "outputs", changed)
    assert any("Stale generated output" in error for error in validator.validate())


def test_sync_does_not_change_existing_company_groups():
    current = sync.load(sync.ROOT / "data.json")
    rendered = json.loads(sync.outputs()[sync.ROOT / "data.json"])
    assert [g for g in current if g["company"] != sync.GROUP] == [
        g for g in rendered if g["company"] != sync.GROUP
    ]
