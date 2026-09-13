#!/usr/bin/env python3
"""Validate repository data files used by the static site and collector.

Checks:
- data/interviews.json: required fields, unique ids, unique non-null source
  URLs, score range, and that unstable Xiaohongshu direct links are not stored.
- data.json: top-level list of companies, each with a name and a question list.

Exits non-zero when any problem is found so it can gate CI.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_INTERVIEW_FIELDS = ("id", "platform", "title", "score")
ALLOWED_SOURCE_TYPES = {"first_hand", "aggregation", "reference"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
REQUIRED_SIGNAL_FIELDS = (
    "id",
    "question",
    "category",
    "companies",
    "rounds",
    "source_ids",
    "source_count",
    "last_seen_at",
    "source_types",
    "confidence",
    "answer_outline",
    "evaluation_points",
)


def validate_iso_date(value: object, loc: str, field: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        errors.append(f"{loc}: '{field}' must be an ISO date string")
        return
    try:
        if re.fullmatch(r"\d{4}-\d{2}", value):
            date.fromisoformat(f"{value}-01")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            date.fromisoformat(value)
        else:
            raise ValueError
    except ValueError:
        errors.append(
            f"{loc}: invalid '{field}' date '{value}' "
            "(expected YYYY-MM or YYYY-MM-DD)"
        )


def validate_interviews(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: file not found"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc})"]

    items = data.get("items")
    if not isinstance(items, list):
        return [f"{path}: missing top-level 'items' list"]

    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for index, item in enumerate(items):
        loc = f"{path} items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{loc}: expected an object")
            continue

        for field in REQUIRED_INTERVIEW_FIELDS:
            value = item.get(field)
            if value is None or value == "":
                errors.append(f"{loc}: missing '{field}'")

        item_id = item.get("id")
        if item_id:
            if item_id in seen_ids:
                errors.append(f"{loc}: duplicate id '{item_id}'")
            seen_ids.add(item_id)

        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int):
            errors.append(f"{loc}: score must be an integer")
        elif not 1 <= score <= 5:
            errors.append(f"{loc}: score {score} out of range 1-5")

        url = item.get("source_url")
        if url:
            if url in seen_urls:
                errors.append(f"{loc}: duplicate source_url '{url}'")
            seen_urls.add(url)
            if "xiaohongshu.com/search_result" in url:
                errors.append(f"{loc}: unstable Xiaohongshu direct link in source_url")

        topics = item.get("topics")
        if topics is not None and not isinstance(topics, list):
            errors.append(f"{loc}: 'topics' must be a list when present")

        validate_iso_date(item.get("published_at"), loc, "published_at", errors)
        validate_iso_date(item.get("verified_at"), loc, "verified_at", errors)

        source_type = item.get("source_type")
        if source_type is not None and source_type not in ALLOWED_SOURCE_TYPES:
            errors.append(
                f"{loc}: source_type '{source_type}' must be one of "
                f"{sorted(ALLOWED_SOURCE_TYPES)}"
            )

        confidence = item.get("confidence")
        if confidence is not None and confidence not in ALLOWED_CONFIDENCE:
            errors.append(
                f"{loc}: confidence '{confidence}' must be one of "
                f"{sorted(ALLOWED_CONFIDENCE)}"
            )

    return errors


def validate_questions(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: file not found"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc})"]

    if not isinstance(data, list):
        return [f"{path}: expected a top-level list"]

    for index, company in enumerate(data):
        loc = f"{path} [{index}]"
        if not isinstance(company, dict):
            errors.append(f"{loc}: expected an object")
            continue
        if not company.get("company"):
            errors.append(f"{loc}: missing 'company'")
        if not isinstance(company.get("questions"), list):
            errors.append(f"{loc}: missing 'questions' list")

    return errors


def validate_question_signals(path: Path, interviews_path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: file not found"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        interviews = json.loads(interviews_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON ({exc})"]

    items = data.get("items")
    if not isinstance(items, list):
        return [f"{path}: missing top-level 'items' list"]

    interview_ids = {
        item.get("id")
        for item in interviews.get("items", [])
        if isinstance(item, dict) and item.get("id")
    }
    seen_ids: set[str] = set()

    for index, item in enumerate(items):
        loc = f"{path} items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{loc}: expected an object")
            continue

        for field in REQUIRED_SIGNAL_FIELDS:
            if field not in item or item[field] in (None, "", []):
                errors.append(f"{loc}: missing '{field}'")

        signal_id = item.get("id")
        if signal_id:
            if signal_id in seen_ids:
                errors.append(f"{loc}: duplicate id '{signal_id}'")
            seen_ids.add(signal_id)

        source_ids = item.get("source_ids")
        if isinstance(source_ids, list):
            missing_sources = sorted(set(source_ids) - interview_ids)
            if missing_sources:
                errors.append(
                    f"{loc}: unknown source_ids {missing_sources}"
                )
            if item.get("source_count") != len(set(source_ids)):
                errors.append(
                    f"{loc}: source_count must equal distinct source_ids"
                )
        else:
            errors.append(f"{loc}: 'source_ids' must be a list")

        for field in ("companies", "rounds", "source_types", "answer_outline", "evaluation_points"):
            if field in item and not isinstance(item[field], list):
                errors.append(f"{loc}: '{field}' must be a list")

        invalid_source_types = set(item.get("source_types", [])) - ALLOWED_SOURCE_TYPES
        if invalid_source_types:
            errors.append(
                f"{loc}: invalid source_types {sorted(invalid_source_types)}"
            )

        confidence = item.get("confidence")
        if confidence not in ALLOWED_CONFIDENCE:
            errors.append(
                f"{loc}: confidence '{confidence}' must be one of "
                f"{sorted(ALLOWED_CONFIDENCE)}"
            )

        validate_iso_date(item.get("last_seen_at"), loc, "last_seen_at", errors)

    return errors


def main() -> int:
    errors: list[str] = []
    interviews_path = ROOT / "data" / "interviews.json"
    errors += validate_interviews(interviews_path)
    errors += validate_questions(ROOT / "data.json")
    errors += validate_question_signals(
        ROOT / "data" / "question_signals.json",
        interviews_path,
    )

    if errors:
        print("Data validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Data validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
