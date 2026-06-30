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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_INTERVIEW_FIELDS = ("id", "platform", "title", "score")


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


def main() -> int:
    errors: list[str] = []
    errors += validate_interviews(ROOT / "data" / "interviews.json")
    errors += validate_questions(ROOT / "data.json")

    if errors:
        print("Data validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Data validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
