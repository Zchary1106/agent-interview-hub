#!/usr/bin/env python3
"""Render the October practice snapshot without upgrading its evidence level.

Canonical inputs: data/practice-2026-10/*.json and the matching collection_batch
in data/interviews.json. --check detects stale generated Markdown/UI entries.
No network, source crawling, or modification of existing company questions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "practice-2026-10"
GROUP = "2026-10 跨公司主题练习（非真题）"
INDEX = ROOT / "通用知识" / "2026-10-Agent面试题索引.md"
SOURCES = ROOT / "通用知识" / "2026-10-Agent面经来源与答案资料.md"
NOTICE = (
    "本批题目按公开面经报道主题改写，是练习题，不是公司官方真题；"
    "答案为编辑综合参考，未经独立专家逐题复核，不是公司标准答案。"
)
DATES = "来源快照：2026-10-22；答案整理：2026-10-23。此次迁入不代表重新抓取。"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(target: Path, document: Path) -> str:
    return Path(os.path.relpath(target, document.parent)).as_posix()


def grade(source: dict) -> str:
    return "B：曾定位公开原帖" if source.get("original_post_found") else "C：仅聚合线索"


def source_label(source: dict) -> str:
    return f"[{source['id']}]({source['source_url']})（{grade(source)}）"


def render_topic(path: Path, questions: list[dict], sources: dict, refs: dict) -> str:
    topic = questions[0]["topic"]
    lines = [
        f"# 2026-10 {topic}面试题与参考答案",
        "",
        f"> {DATES}",
        f"> {NOTICE}",
        "",
        (
            f"[返回本批索引]({relative(INDEX, path)}) · "
            f"[来源分级与资料]({relative(SOURCES, path)})"
        ),
        "",
        "## 使用方式",
        "",
        "先独立回答，再对照得分点、追问和易错点。难度为编辑估计，非公司定级。",
        (
            "公司标签只表示来源报道过相关主题，不证明逐字问过本题。"
            "B 仅代表曾找到原帖，不保证全文可读、作者亲历或题目真实；C 为待核验聚合线索。"
        ),
        "",
    ]
    if topic == "项目与研发":
        lines += [
            (
                "> 项目题给出表达与验证框架，不提供可冒充个人经历的模板数据。"
                "只陈述本人确实做过、测过和部署过的部分，不编造 ROI 或生产成功率。"
            ),
            "",
        ]
    if topic == "算法与手写":
        lines += [
            "## 可执行代码与验证",
            "",
            (
                "[Python 参考实现](python/algorithm_answers.py) · "
                "[离线测试](python/test_algorithm_answers.py) · "
                "[输入契约与测试说明](python/README.md)"
            ),
            "",
            (
                "Python 3.11+；实现仅用标准库，测试依赖 pytest。"
                "这些实现是面试练习，不是生产库。增量 JSON 仅支持顶层对象/数组；"
                "区间使用半开约定；链表、树与深合并的具体输入契约以代码为准。"
            ),
            "",
        ]
    for q in questions:
        lines += [
            f'<a id="{q["id"]}"></a>',
            "",
            f"## {q['position']}. {q['question']}",
            "",
            f"**题目 ID**：`{q['id']}` · **编辑难度**：`{q['difficulty_editorial']}`",
            "",
            f"**相关主题的报道公司（非逐题归属）**：{'、'.join(q['companies_reported'])}",
            "",
            "**参考答案**",
            "",
            q["reference_answer"],
            "",
            "**得分点**",
            "",
            *[f"- {point}" for point in q["answer_key_points"]],
            "",
            f"**追问**：{q['answer_follow_up']}",
            "",
            f"**易错点**：{q['answer_pitfall']}",
            "",
            "**面经主题线索**："
            + "；".join(source_label(sources[sid]) for sid in q["source_ids"]),
            "",
            "**答案机制资料**："
            + (
                "；".join(
                    f"[{rid}：{refs[rid]['title']}]({refs[rid]['url']})"
                    for rid in q["answer_reference_ids"]
                )
                or "未单列外部机制资料，属于编辑推导，须结合具体契约独立验证。"
            ),
            "",
        ]
        for sid in q["source_ids"]:
            if note := sources[sid].get("company_attribution_note"):
                lines += [f"> 归属提醒（{sid}）：{note}", ""]
        if q["answer_code_symbols"]:
            lines += [
                "**代码入口**：[algorithm_answers.py](python/algorithm_answers.py) — "
                + "、".join(f"`{symbol}`" for symbol in q["answer_code_symbols"]),
                "",
            ]
        lines += ["---", ""]
    return "\n".join(lines)


def render_index(questions: list[dict], sources: dict) -> str:
    documents = dict.fromkeys(q["document_path"] for q in questions)
    lines = [
        "# 2026-10 Agent 面试题索引：156 道练习题与参考答案",
        "",
        f"> {DATES}",
        f"> {NOTICE}",
        "",
        "## 放置规则与阅读入口",
        "",
        (
            "遵循仓库原有目录：通用技术按主题放入 `通用知识/`，"
            "算法放入 `面试算法题/`，项目表达与研发放入 `项目实战/`。"
            "不把改写题拆入公司“真实面经”目录，也不计入高频题信号榜。"
        ),
        "",
        (
            "新版交互题库使用根目录 `data.json`，本批作为独立的"
            f"「{GROUP}」分组，可搜索并折叠答案；旧分组保持不变。"
        ),
        "",
        "| 分类 | 题数 | 文档 |",
        "|---|---:|---|",
    ]
    for document in documents:
        items = [q for q in questions if q["document_path"] == document]
        lines.append(
            f"| {items[0]['topic']} | {len(items)} | "
            f"[题目与答案]({relative(ROOT / document, INDEX)}) |"
        )
    located = sum(bool(s.get("original_post_found")) for s in sources.values())
    preview = sum(
        s.get("primary_access_scope") == "public_preview_only" for s in sources.values()
    )
    lines += [
        "",
        "## 证据、日期与使用边界",
        "",
        f"- 共 {len(questions)} 题、{len(documents)} 类，均保留稳定 ID、得分点、追问和易错点。",
        (
            f"- 共 {len(sources)} 条来源，其中 {located} 条曾定位公开原帖、"
            f"{len(sources) - located} 条为聚合线索；{preview} 条检测为付费预览范围。"
        ),
        (
            "- 只处理公开内容，不解锁、搬运付费全文。曾定位不代表当前能读到全文；"
            "最近一次抓取不可读的状态原样保留。"
        ),
        (
            "- B/C 是本批研究证据标签，不等同于仓库其他来源的 high/medium/low；"
            "迁入来源统一保守标为 aggregation / low，不自动升级一手或高置信度。"
        ),
        (
            "- 原帖发帖日期、报道的面试日期、聚合站月份不同；"
            "聚合站每月 1 日占位值不当成准确发布日期。"
        ),
        (
            "- 公司标签仅为报道主题关联；同一来源可支持多个练习角度，"
            "不代表多场独立面试，不计算出现频率或招聘热度。"
        ),
        (
            "- 算法测试只证明固定契约内的离线结果，不证明生产可靠性；"
            "其余答案为编辑参考，未做逐题独立专家复核。"
        ),
        (
            f"- {sum(bool(q['answer_reference_ids']) for q in questions)} 题附外部机制资料；"
            "其余题明确标注未单列外部资料，不宣称全部答案均获文献验证。"
        ),
        "",
        f"[完整来源与答案资料]({relative(SOURCES, INDEX)})",
        "",
        "## 按报道公司查来源（不是公司真题分类）",
        "",
        "| 报道公司（别名归一化，归属未认证） | 来源条数（未按面试事件去重） |",
        "|---|---:|",
    ]
    for company, count in sorted(
        Counter(
            s.get("company_normalized", s["company_reported"]) for s in sources.values()
        ).items()
    ):
        lines.append(f"| {company} | {count} |")
    lines += [
        "",
        "阿里巴巴 / 蚂蚁混称等归属疑点见来源逐条备注，不推断为官方公司题。",
        "",
        "## 数据与复现",
        "",
        (
            "[题目、答案与路径映射](../data/practice-2026-10/questions.json) · "
            "[面经来源主目录](../data/interviews.json) · "
            "[答案资料](../data/practice-2026-10/answer-references.json)"
        ),
        "",
        "```bash",
        "# 从仓库根目录运行；同步器不联网、不重新采集",
        "python3 scripts/sync_practice_2026_10.py --check",
        "python3 scripts/validate_data.py",
        "python3 scripts/validate_practice_2026_10.py",
        "python3 -m pytest -q 面试算法题/python/test_algorithm_answers.py",
        "python3 scripts/build_site.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def render_sources(sources: dict, refs: dict) -> str:
    lines = [
        "# 2026-10 Agent 面经来源与答案资料",
        "",
        f"> {DATES}",
        f"> {NOTICE}",
        "",
        f"[返回本批题目索引]({relative(INDEX, SOURCES)})",
        "",
        "## 证据分级",
        "",
        (
            "- **B**：曾定位公开原帖，可能仅预览或最近一次抓取不可读；"
            "不是作者身份认证，也不是逐题核验。"
        ),
        "- **C**：只找到聚合线索，原帖或具体内容待核验。",
        (
            "- `evidence` 保留更细分的原始状态。`published_month` 为报道月份；"
            "原帖日期和面试日期另外列示，未知则不猜。"
        ),
        (
            "- 全部来源在 `data/interviews.json` 中维护，供题目按稳定 ID 引用。"
            "同帖多种练习不增加独立来源数，不把 63 条当作 63 场真实面试。"
        ),
        "",
        "## 面经主题来源",
        "",
    ]
    for sid, s in sources.items():
        lines += [
            f"### {s['display_id']} · {s['title']}",
            "",
            f"**ID**：`{sid}` · **报道公司**：{s['company_reported']} · **证据**：{grade(s)}",
            "",
            (
                f"**细分状态**：`{s['evidence']}` · "
                f"**最近抓取可读**：{s.get('last_fetch_post_available', '未知')} · "
                f"**访问范围**：`{s.get('primary_access_scope', 'aggregator_only')}`"
            ),
            "",
            (
                f"**报道月份**：{s['published_month']} · "
                f"**原帖日期**：{s.get('primary_published_at', '未知')} · "
                f"**报道面试日期**：{s.get('reported_interview_date', '未知')}"
            ),
            "",
            f"[公开来源]({s['source_url']}) · [原始采集链接]({s['url']})",
            "",
        ]
        for key in ("date_note", "post_identity_note", "company_attribution_note"):
            if s.get(key):
                lines += [f"> {s[key]}", ""]
    lines += [
        "## 答案机制资料",
        "",
        (
            "这些规范、官方文档与原始论文仅支持机制解释，不证明公司曾问过对应题目。"
            "保留原整理记录中的检查日期，此次迁入不声称逐链接重新在线核验。"
        ),
        "",
    ]
    for rid, ref in refs.items():
        lines += [
            (
                f"- **{rid}**：[{ref['title']}]({ref['url']})"
                f"（{ref['kind']}；原记录检查：{ref['checked_on']}）"
            )
        ]
    return "\n".join(lines) + "\n"


def ui_question(q: dict, sources: dict, refs: dict) -> dict:
    sections = [
        q["reference_answer"],
        "得分点：" + "；".join(q["answer_key_points"]),
        "追问：" + q["answer_follow_up"],
        "易错点：" + q["answer_pitfall"],
        "机制资料："
        + (
            "；".join(refs[rid]["url"] for rid in q["answer_reference_ids"])
            or "未单列外部机制资料，属于编辑推导，须结合具体契约独立验证。"
        ),
        "主题线索：" + "；".join(sources[sid]["source_url"] for sid in q["source_ids"]),
    ]
    if q["answer_code_symbols"]:
        sections.append(
            "代码：" + q["code_file"] + "；入口：" + "、".join(q["answer_code_symbols"])
        )
    return {
        "id": q["id"],
        "category": q["topic"],
        "question": q["question"],
        "answer": "\n\n".join(sections),
        "thinking": f"{NOTICE} {DATES} 主题：{q['topic']}；"
        f"报道相关公司（非逐题归属）：{'、'.join(q['companies_reported'])}；"
        f"来源 ID：{'、'.join(q['source_ids'])}。"
        + " ".join(
            sources[sid].get("company_attribution_note", "") for sid in q["source_ids"]
        ),
    }


def outputs() -> dict[Path, str]:
    questions = load(DATA / "questions.json")
    sources = {
        s["id"]: s
        for s in load(ROOT / "data/interviews.json")["items"]
        if s.get("collection_batch") == "2026-10-22"
    }
    refs = {r["id"]: r for r in load(DATA / "answer-references.json")}
    result = {
        INDEX: render_index(questions, sources),
        SOURCES: render_sources(sources, refs),
    }
    for document in dict.fromkeys(q["document_path"] for q in questions):
        path = ROOT / document
        result[path] = render_topic(
            path,
            [q for q in questions if q["document_path"] == document],
            sources,
            refs,
        )
    groups = [g for g in load(ROOT / "data.json") if g["company"] != GROUP]
    groups.append(
        {
            "company": GROUP,
            "icon": "📝",
            "questions": [ui_question(q, sources, refs) for q in questions],
        }
    )
    result[ROOT / "data.json"] = json.dumps(groups, ensure_ascii=False, indent=2) + "\n"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if generated outputs differ"
    )
    args = parser.parse_args()
    stale = []
    for path, content in outputs().items():
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        if args.check:
            stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"Wrote {path.relative_to(ROOT)}")
    if stale:
        print("Stale practice outputs:\n" + "\n".join(stale))
        return 1
    print("Practice outputs are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
