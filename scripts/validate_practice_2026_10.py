#!/usr/bin/env python3
"""Check October question coverage, evidence boundaries, bindings and local links."""

from __future__ import annotations

import ast
import re
from collections import Counter
from urllib.parse import unquote, urlsplit

from sync_practice_2026_10 import DATA, GROUP, ROOT, load, outputs

COUNTS = {
    "Agent架构": 12,
    "工具与MCP": 12,
    "状态与恢复": 12,
    "上下文与记忆": 12,
    "RAG与检索": 18,
    "评测与归因": 12,
    "多Agent协作": 10,
    "安全与治理": 10,
    "性能与成本": 10,
    "后端基础": 13,
    "算法与手写": 15,
    "项目与研发": 14,
    "模型与算法": 6,
}


def validate() -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    questions = load(DATA / "questions.json")
    sources = [
        s
        for s in load(ROOT / "data/interviews.json")["items"]
        if s.get("collection_batch") == "2026-10-22"
    ]
    refs = load(DATA / "answer-references.json")
    source_ids = {s["id"] for s in sources}
    ref_ids = {r["id"] for r in refs}
    require(len(questions) == 156, "Expected 156 practice questions")
    require(Counter(q["topic"] for q in questions) == COUNTS, "Topic counts changed")
    require(
        len({q["id"] for q in questions}) == len(questions), "Duplicate question ID"
    )
    require(
        len({q["question"] for q in questions}) == len(questions),
        "Duplicate question text",
    )
    require(
        [q["position"] for q in questions] == list(range(1, 157)),
        "Position order changed",
    )
    require(
        len(sources) == len(source_ids) == 63, "Expected 63 unique batch source IDs"
    )
    require(len(refs) == len(ref_ids) == 26, "Expected 26 unique mechanism references")
    code = ROOT / "面试算法题/python/algorithm_answers.py"
    code_symbols = {
        node.name
        for node in ast.parse(code.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for source in sources:
        sid = source["id"]
        require(source.get("source_type") == "aggregation", f"{sid}: evidence upgraded")
        require(source.get("confidence") == "low", f"{sid}: confidence upgraded")
        require(
            source.get("original_interview_verified") is False,
            f"{sid}: claimed verification",
        )
        require(
            source.get("source_url") == (source.get("primary_url") or source["url"]),
            f"{sid}: source URL no longer matches recorded evidence",
        )
        require(
            str(source.get("published_at", "")) != source.get("date_raw_untrusted"),
            f"{sid}: untrusted day-1 placeholder used as publication date",
        )
    for q in questions:
        qid = q["id"]
        require(bool(re.fullmatch(r"Q-[0-9a-f]{12}", qid)), f"{qid}: invalid stable ID")
        require(
            q.get("question_kind") == "editorial_paraphrase_of_reported_topics",
            f"{qid}: must remain an editorial paraphrase",
        )
        for field in (
            "authentic_company_question_verified",
            "question_text_verified_against_primary",
        ):
            require(q.get(field) is False, f"{qid}: {field} must be false")
        require(
            q.get("answer_status") == "editorial_reference_draft",
            f"{qid}: answer status changed",
        )
        require(q.get("curated_at") == "2026-10-22", f"{qid}: snapshot date changed")
        require(q.get("answer_date") == "2026-10-23", f"{qid}: answer date changed")
        for field in (
            "question",
            "reference_answer",
            "answer_follow_up",
            "answer_pitfall",
        ):
            require(
                isinstance(q.get(field), str) and bool(q[field].strip()),
                f"{qid}: missing {field}",
            )
        for field in ("answer_key_points", "source_ids"):
            values = q.get(field)
            require(
                isinstance(values, list)
                and bool(values)
                and all(isinstance(v, str) and bool(v.strip()) for v in values),
                f"{qid}: missing/invalid {field}",
            )
        values = q.get("answer_reference_ids")
        require(
            isinstance(values, list) and all(isinstance(v, str) and v for v in values),
            f"{qid}: invalid optional mechanism references",
        )
        require(
            set(q.get("source_ids", [])) <= source_ids,
            f"{qid}: dangling interview source",
        )
        require(
            set(q.get("answer_reference_ids", [])) <= ref_ids,
            f"{qid}: dangling answer reference",
        )
        folder = (
            "面试算法题"
            if q["topic"] == "算法与手写"
            else ("项目实战" if q["topic"] == "项目与研发" else "通用知识")
        )
        expected = f"{folder}/2026-10-{q['topic']}面试题与参考答案.md"
        require(q.get("document_path") == expected, f"{qid}: wrong document location")
        doc = ROOT / expected
        require(doc.is_file(), f"{qid}: missing Markdown document")
        if doc.is_file():
            require(
                doc.read_text(encoding="utf-8").count(f'<a id="{qid}"></a>') == 1,
                f"{qid}: Markdown anchor missing or duplicated",
            )
        symbols = q.get("answer_code_symbols", [])
        require(
            bool(symbols) == (q["topic"] == "算法与手写"),
            f"{qid}: incorrect code binding",
        )
        require(set(symbols) <= code_symbols, f"{qid}: unknown code symbol")
        if symbols:
            require(
                q.get("code_file") == code.relative_to(ROOT).as_posix(),
                f"{qid}: wrong code path",
            )

    practice_groups = [g for g in load(ROOT / "data.json") if g["company"] == GROUP]
    require(
        len(practice_groups) == 1, "Expected exactly one non-company UI practice group"
    )
    if len(practice_groups) == 1:
        require(
            [q["id"] for q in practice_groups[0]["questions"]]
            == [q["id"] for q in questions],
            "UI question IDs/order do not match canonical data",
        )
    # Re-rendering also checks all answers, labels, source links and warnings in
    # the interactive group, not just its IDs.
    try:
        generated = outputs()
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"Cannot render canonical inputs: {exc}")
        return errors
    for path, expected_content in generated.items():
        require(
            path.is_file() and path.read_text(encoding="utf-8") == expected_content,
            f"Stale generated output: {path.relative_to(ROOT)}",
        )
    documents = [p for p in generated if p.suffix == ".md"]
    documents += [
        ROOT / "面试算法题/python/README.md",
        ROOT / "data/practice-2026-10/README.md",
    ]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        require(
            "/Users/" not in text and "/private/tmp/" not in text,
            f"{path.relative_to(ROOT)}: leaked local path",
        )
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            parsed = urlsplit(match.group(1))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            require(
                target.is_relative_to(ROOT) and target.is_file(),
                f"{path.relative_to(ROOT)}: broken local link {match.group(1)}",
            )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(
            "Practice validation FAILED:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )
        return 1
    print(
        "Practice validation passed: 156 questions, 13 topics, 63 sources, 26 references."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
