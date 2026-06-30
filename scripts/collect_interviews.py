#!/usr/bin/env python3
"""Collect public interview-source candidates for agent-interview-hub.

This script is intentionally conservative:
- it only searches/reads through local CLI tools and public pages;
- it does not bypass login walls or anti-bot controls;
- it creates candidate files for human/agent review instead of blindly
  rewriting curated repository documents.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "interview_candidates.json"
DEFAULT_REPORT = ROOT / "data" / "interview_candidates.md"

DEFAULT_QUERIES = [
    "AI Agent 大模型 面经 2026",
    "AI Agent RAG 面经 2026",
    "大模型应用开发 Agent 面经",
]

PLATFORM_PRESETS = {
    "nowcoder": ["site:nowcoder.com {query}"],
    "zhihu": ["site:zhihu.com {query}", "site:zhuanlan.zhihu.com {query}"],
    "blogs": [
        "site:csdn.net {query}",
        "site:cnblogs.com {query}",
        "site:juejin.cn {query}",
    ],
    "web": ["{query}"],
}

COMPANY_KEYWORDS = [
    "字节跳动",
    "字节",
    "阿里巴巴",
    "阿里",
    "淘天",
    "蚂蚁",
    "腾讯",
    "百度",
    "美团",
    "小红书",
    "快手",
    "华为",
    "京东",
    "理想汽车",
    "拼多多",
    "PDD",
    "TEMU",
    "OpenAI",
    "Google",
    "DeepMind",
    "Anthropic",
]

TOPIC_KEYWORDS = OrderedDict(
    [
        ("Agent", ["Agent", "智能体", "agent"]),
        ("RAG", ["RAG", "检索", "向量", "知识库", "召回"]),
        ("MCP", ["MCP"]),
        ("Function Calling", ["Function Calling", "Tool Calling", "工具调用", "tool"]),
        ("Multi-Agent", ["多Agent", "Multi-Agent", "A2A", "协作"]),
        ("LangGraph", ["LangGraph"]),
        ("Memory", ["Memory", "记忆", "上下文"]),
        ("Rerank", ["Rerank", "重排"]),
        ("BM25", ["BM25"]),
        ("LoRA/SFT", ["LoRA", "SFT", "微调", "QLoRA"]),
        ("Evaluation", ["评估", "RAGAS", "NDCG", "Recall", "指标"]),
        ("Inference", ["推理", "vLLM", "SGLang", "KV Cache", "量化"]),
        ("Safety", ["安全", "权限", "风控", "注入", "沙箱"]),
    ]
)


@dataclass
class Candidate:
    id: str
    platform: str
    title: str
    company: str | None = None
    role: str | None = None
    published_at: str | None = None
    source_url: str | None = None
    source_note_id: str | None = None
    source_lookup: str | None = None
    score: int = 3
    topics: list[str] = field(default_factory=list)
    summary: str = ""
    raw_query: str | None = None


def run_command(args: list[str], timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(f"Missing command: {args[0]}") from None
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out: {' '.join(args)}") from exc

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{message}")
    return result.stdout


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:80] or "candidate"


def normalize_title(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "", value)
    return value


def detect_company(text: str) -> str | None:
    for company in COMPANY_KEYWORDS:
        if company.lower() in text.lower():
            if company in {"字节"}:
                return "字节跳动"
            if company in {"阿里"}:
                return "阿里巴巴"
            if company in {"DeepMind"}:
                return "Google"
            return company
    return None


def detect_topics(text: str) -> list[str]:
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            topics.append(topic)
    return topics


def infer_platform_from_url(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if "nowcoder.com" in host:
        return "牛客"
    if "zhihu.com" in host:
        return "知乎"
    if "csdn.net" in host or "gitcode.csdn.net" in host or "devpress.csdn.net" in host:
        return "CSDN"
    if "cnblogs.com" in host:
        return "博客园"
    if "juejin.cn" in host:
        return "掘金"
    if "github.com" in host:
        return "GitHub"
    return host or "web"


def score_candidate(title: str, summary: str, company: str | None, published_at: str | None, platform: str) -> int:
    text = f"{title} {summary}"
    score = 3
    if company:
        score += 1
    if published_at and re.search(r"202[5-9]|2026", published_at):
        score += 1
    if len(detect_topics(text)) >= 3:
        score += 1
    if platform in {"牛客", "小红书"}:
        score += 1
    if platform == "GitHub" and re.search(r"面试|interview|Agent|RAG|LLM", text, re.I):
        score += 1
    return max(1, min(5, score))


def make_candidate(
    *,
    platform: str,
    title: str,
    source_url: str | None,
    summary: str,
    published_at: str | None = None,
    raw_query: str | None = None,
    extra: dict | None = None,
) -> Candidate:
    text = " ".join(filter(None, [title, summary, raw_query or ""]))
    company = detect_company(text)
    topics = detect_topics(text)

    source_note_id = None
    source_lookup = None
    if platform == "小红书":
        if source_url:
            match = re.search(r"/search_result/([^?/#]+)", source_url)
            if match:
                source_note_id = match.group(1)
        source_lookup = f"小红书站内搜索原标题：{title}"
        source_url = None

    stable = source_note_id or source_url or title
    cid = f"{platform}-{slugify(stable)}"
    return Candidate(
        id=cid,
        platform=platform,
        title=title.strip(),
        company=company,
        published_at=published_at,
        source_url=source_url,
        source_note_id=source_note_id,
        source_lookup=source_lookup,
        score=score_candidate(title, summary, company, published_at, platform),
        topics=topics,
        summary=summary.strip(),
        raw_query=raw_query,
        **(extra or {}),
    )


def parse_exa_output(output: str, raw_query: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    blocks = re.split(r"\n---+\n", output.strip())
    for block in blocks:
        title = re.search(r"^Title:\s*(.+)$", block, re.M)
        url = re.search(r"^URL:\s*(.+)$", block, re.M)
        if not title or not url:
            continue

        published = re.search(r"^Published:\s*(.+)$", block, re.M)
        highlights = ""
        if "Highlights:" in block:
            highlights = block.split("Highlights:", 1)[1]
        highlights = re.sub(r"\n{3,}", "\n\n", highlights).strip()
        summary = " ".join(line.strip() for line in highlights.splitlines() if line.strip())[:500]
        source_url = url.group(1).strip()
        platform = infer_platform_from_url(source_url)
        candidates.append(
            make_candidate(
                platform=platform,
                title=title.group(1).strip(),
                source_url=source_url,
                summary=summary,
                published_at=(published.group(1).strip() if published else None),
                raw_query=raw_query,
            )
        )
    return candidates


def parse_xiaohongshu_output(output: str, raw_query: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    current: dict[str, str] | None = None
    pending_key: str | None = None

    def finish() -> None:
        if not current or not current.get("title"):
            return
        title = current["title"].strip().strip("'\"")
        summary_parts = []
        if current.get("author"):
            summary_parts.append(f"作者：{current['author']}")
        if current.get("likes"):
            summary_parts.append(f"点赞：{current['likes']}")
        summary = "；".join(summary_parts) or "小红书搜索结果候选。"
        candidates.append(
            make_candidate(
                platform="小红书",
                title=title,
                source_url=current.get("url"),
                summary=summary,
                published_at=current.get("published_at"),
                raw_query=raw_query,
            )
        )

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if line.startswith("- rank:"):
            finish()
            current = {}
            pending_key = None
            continue
        if current is None:
            continue

        if pending_key and line.startswith("    "):
            current[pending_key] = line.strip().strip("'\"")
            pending_key = None
            continue

        match = re.match(r"\s{2}([\w_]+):\s*(.*)$", line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value == ">-":
            pending_key = key
        else:
            current[key] = value.strip("'\"")

    finish()
    return candidates


def search_exa(query: str, limit: int) -> list[Candidate]:
    query_literal = json.dumps(query, ensure_ascii=False)
    command = f"exa.web_search_exa(query: {query_literal}, numResults: {limit})"
    output = run_command(["mcporter", "call", command], timeout=90)
    return parse_exa_output(output, query)


def search_xiaohongshu(query: str) -> list[Candidate]:
    output = run_command(
        ["opencli", "xiaohongshu", "search", query, "-f", "yaml", "--window", "background"],
        timeout=120,
    )
    return parse_xiaohongshu_output(output, query)


def search_github(query: str, limit: int) -> list[Candidate]:
    output = run_command(
        [
            "gh",
            "search",
            "repos",
            query,
            "--limit",
            str(limit),
            "--json",
            "fullName,description,stargazersCount,url,updatedAt",
        ],
        timeout=60,
    )
    rows = json.loads(output or "[]")
    candidates = []
    for row in rows:
        title = row.get("fullName") or row.get("url") or "GitHub repository"
        desc = row.get("description") or ""
        stars = row.get("stargazersCount")
        summary = f"{desc} Stars: {stars}" if stars is not None else desc
        candidates.append(
            make_candidate(
                platform="GitHub",
                title=title,
                source_url=row.get("url"),
                summary=summary,
                published_at=(row.get("updatedAt") or "")[:10] or None,
                raw_query=query,
            )
        )
    return candidates


def expand_queries(platforms: Iterable[str], queries: Iterable[str]) -> list[tuple[str, str]]:
    expanded: list[tuple[str, str]] = []
    for platform in platforms:
        if platform in {"github", "xiaohongshu"}:
            for query in queries:
                expanded.append((platform, query))
            continue
        templates = PLATFORM_PRESETS.get(platform)
        if not templates:
            continue
        for query in queries:
            for template in templates:
                expanded.append((platform, template.format(query=query)))
    return expanded


def dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    deduped: OrderedDict[str, Candidate] = OrderedDict()
    for candidate in candidates:
        key = (
            candidate.source_url
            or (f"xhs:{candidate.source_note_id}" if candidate.source_note_id else None)
            or f"title:{candidate.platform}:{normalize_title(candidate.title)}"
        )
        existing = deduped.get(key)
        if not existing or candidate.score > existing.score:
            deduped[key] = candidate
    return list(deduped.values())


def load_existing(path: Path) -> list[Candidate]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("items", data if isinstance(data, list) else [])
    candidates = []
    for row in rows:
        allowed = {field.name for field in Candidate.__dataclass_fields__.values()}
        candidates.append(Candidate(**{k: v for k, v in row.items() if k in allowed}))
    return candidates


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "Raw interview-source candidates for human/agent review before promotion to data/interviews.json.",
        "items": [asdict(candidate) for candidate in candidates],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_report(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_platform: OrderedDict[str, list[Candidate]] = OrderedDict()
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.platform, item.title)):
        by_platform.setdefault(candidate.platform, []).append(candidate)

    lines = [
        "# 面经采集候选报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"候选数：{len(candidates)}",
        "",
    ]
    for platform, rows in by_platform.items():
        lines.extend([f"## {platform}", "", "| 分数 | 公司 | 标题 | 来源/检索 | 摘要 |", "|---:|---|---|---|---|"])
        for row in rows:
            source = row.source_url or row.source_lookup or row.source_note_id or ""
            title = f"[{row.title}]({row.source_url})" if row.source_url else row.title
            topics = "、".join(row.topics[:6])
            summary = row.summary.replace("|", "｜")
            if topics:
                summary = f"{summary}<br>标签：{topics}"
            lines.append(
                f"| {row.score} | {row.company or ''} | {title} | {source} | {summary} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def command_doctor(_: argparse.Namespace) -> int:
    checks = [
        ("curl", "public page reading through Jina Reader"),
        ("mcporter", "Exa search"),
        ("opencli", "Xiaohongshu search"),
        ("gh", "GitHub search"),
        ("agent-reach", "tool routing and diagnostics"),
    ]
    for command, purpose in checks:
        status = "ok" if shutil.which(command) else "missing"
        print(f"{command:12} {status:8} {purpose}")
    return 0


def command_search(args: argparse.Namespace) -> int:
    platforms = args.platform or ["nowcoder", "zhihu", "blogs", "github"]
    queries = args.query or DEFAULT_QUERIES
    new_candidates: list[Candidate] = []
    errors: list[str] = []

    for platform, query in expand_queries(platforms, queries):
        try:
            print(f"searching {platform}: {query}", file=sys.stderr)
            if platform == "xiaohongshu":
                new_candidates.extend(search_xiaohongshu(query))
            elif platform == "github":
                new_candidates.extend(search_github(query, args.limit))
            else:
                new_candidates.extend(search_exa(query, args.limit))
        except RuntimeError as exc:
            errors.append(str(exc))
            print(f"warning: {exc}", file=sys.stderr)

    if errors and not new_candidates:
        print(f"completed with {len(errors)} warning(s)", file=sys.stderr)
        print("no candidates collected because all requested searches failed", file=sys.stderr)
        return 1

    candidates = new_candidates
    if args.append:
        candidates = load_existing(args.output) + candidates
    candidates = dedupe_candidates(candidates)
    write_candidates(args.output, candidates)
    if args.report:
        render_report(args.report, candidates)

    print(f"wrote {len(candidates)} candidates to {args.output}")
    if args.report:
        print(f"wrote report to {args.report}")
    if errors:
        print(f"completed with {len(errors)} warning(s)", file=sys.stderr)
    return 0


def command_render(args: argparse.Namespace) -> int:
    candidates = dedupe_candidates(load_existing(args.input))
    render_report(args.output, candidates)
    print(f"wrote report to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub_doctor = sub.add_parser("doctor", help="Check optional collection tools")
    sub_doctor.set_defaults(func=command_doctor)

    sub_search = sub.add_parser("search", help="Search public sources and write candidates")
    sub_search.add_argument(
        "--platform",
        action="append",
        choices=["nowcoder", "zhihu", "blogs", "github", "xiaohongshu", "web"],
        help="Platform preset. Repeatable. Default: nowcoder, zhihu, blogs, github.",
    )
    sub_search.add_argument("--query", action="append", help="Search query. Repeatable.")
    sub_search.add_argument("--limit", type=int, default=10, help="Search results per query")
    sub_search.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub_search.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    sub_search.add_argument("--append", action="store_true", help="Append to existing output before dedupe")
    sub_search.set_defaults(func=command_search)

    sub_render = sub.add_parser("render", help="Render a Markdown report from candidate JSON")
    sub_render.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    sub_render.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    sub_render.set_defaults(func=command_render)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
