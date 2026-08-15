#!/usr/bin/env python3
"""Build the GitHub Pages site from Markdown, JSON, and SVG assets."""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

try:
    import markdown
    import pymdownx.superfences  # noqa: F401
except ImportError:
    sys.exit(
        "Missing dependencies. Install them with "
        "`python3 -m pip install -r requirements.txt`."
    )


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIAGRAMS_DIR = ROOT / "diagrams"
DATA_DIR = ROOT / "data"
INTERVIEW_ALGORITHMS_DIR = ROOT / "面试算法题"
LEGACY_DIR = ROOT / "legacy"

ROOT_DOCS = ["Agent工程师学习路线图.md"]

GENERAL_ORDER = [
    "最新AI-Agent面经索引.md",
    "12周Agent工程师进阶路线.md",
    "Agent核心概念与设计模式.md",
    "Agent框架全景.md",
    "LangChain与LangGraph深度解析.md",
    "RAG核心知识与面试题.md",
    "Agentic RAG与GraphRAG深度解析.md",
    "Context Engineering上下文工程.md",
    "Agent安全与评估体系.md",
    "大模型推理优化与部署.md",
    "Function Calling与Tool Use专题.md",
    "MCP与工具生态.md",
    "Agentic Coding与AI编程工具.md",
    "Agent Harness与编码代理测评.md",
    "核心概念详解与参考答案.md",
    "八股文完整答案集.md",
    "八股文题库-DataWhale开源.md",
    "高频拷打题-牛客热帖.md",
    "技术知识点汇总.md",
    "其他公司面经-快手携程等.md",
    "Agent核心概念面试题-进阶篇.md",
    "系统设计面试题-进阶篇.md",
    "AI协作与工程化面试题-进阶篇.md",
    "海外顶级AI公司面试攻略-2026.md",
]

PROJECT_ORDER = [
    "01-RAG知识问答系统.md",
    "02-多Agent协作系统.md",
    "03-生产级Agent应用.md",
    "实操考题/01-智能文档问答Agent.md",
    "实操考题/02-多Agent团队协作.md",
    "实操考题/03-ReAct模式Agent.md",
    "实操考题/04-AI限时全栈开发.md",
    "实操考题/05-AI调试挑战.md",
    "实操考题/06-AI-CodeReview-Agent.md",
]

COMPANY_ORDER = [
    "字节跳动",
    "阿里巴巴",
    "腾讯",
    "百度",
    "美团",
    "小红书",
    "快手",
    "蚂蚁集团",
    "华为",
    "OpenAI",
    "Anthropic",
    "谷歌",
    "微软",
    "初创公司",
    "商汤科技",
]

COMPANY_ICONS = {
    "字节跳动": "🔥",
    "阿里巴巴": "🟠",
    "腾讯": "💬",
    "百度": "🔍",
    "美团": "🟡",
    "小红书": "📕",
    "快手": "⚡",
    "蚂蚁集团": "🐜",
    "华为": "📱",
    "OpenAI": "🧠",
    "Anthropic": "🛡️",
    "谷歌": "🌍",
    "微软": "🪟",
    "初创公司": "🚀",
    "商汤科技": "👁️",
}

DIAGRAM_TITLES = {
    "agent-architecture.svg": "Agent 核心架构",
    "framework-decision-tree.svg": "Agent 框架选型决策树",
    "langgraph-architecture.svg": "LangGraph 图结构示意",
    "multi-agent-patterns.svg": "Multi-Agent 协作模式",
    "rag-pipeline.svg": "RAG Pipeline 全流程",
    "rag-vs-finetune.svg": "RAG vs Fine-tuning",
    "react-loop.svg": "ReAct 循环模式",
    "agent-engineer-competency-model.svg": "AI Agent 工程师能力模型",
    "supervisor-agent-team.svg": "Supervisor Agent 团队分工",
    "agentic-rag-loop-flow.svg": "Agentic RAG 自主检索闭环",
    "personalized-rag-flow.svg": "个性化 RAG 流程",
    "agent-loop-cycle.svg": "Agent Loop 循环",
    "supervisor-multi-agent-simple.svg": "Supervisor 多 Agent 分发",
    "langgraph-stategraph-flow.svg": "LangGraph StateGraph 流程",
    "autogen-group-chat.svg": "AutoGen Group Chat 协作",
    "dify-platform-overview.svg": "Dify 平台能力总览",
    "openai-agents-loop.svg": "OpenAI Agents SDK 执行循环",
    "agent-framework-selection-tree.svg": "Agent 框架选型决策树",
    "customer-service-agent-architecture.svg": "客服 Agent 架构分层",
    "mcp-architecture.svg": "MCP 四大核心组件",
    "mcp-integration-comparison.svg": "MCP 集成复杂度对比",
    "learning-rate-warmup-cosine.svg": "Warmup + Cosine 学习率曲线",
    "deepspeed-zero-stages.svg": "DeepSpeed ZeRO 三阶段对比",
    "lora-adapter-flow.svg": "LoRA 低秩适配器结构",
    "qlora-training-stack.svg": "QLoRA 训练结构",
    "finetune-method-comparison.svg": "微调方法资源对比",
    "rlhf-three-stage-flow.svg": "RLHF 三阶段训练流程",
    "ppo-training-loop.svg": "PPO 训练循环",
    "training-scale-playbook.svg": "训练规模与典型配置",
    "finetune-memory-estimation.svg": "微调显存估算速查",
    "rag-hybrid-retrieval-flow.svg": "RAG 混合检索与生成链路",
    "document-ingestion-pipeline.svg": "RAG 文档入库流水线",
    "rag-deployment-topology.svg": "RAG 系统部署拓扑",
    "agentic-coding-stack.svg": "Agentic Coding 工具分层架构",
    "tool-permission-layers.svg": "工具调用权限控制 5 层",
    "plan-and-execute.svg": "Plan-and-Execute 模式",
    "llm-serving-topology.svg": "大模型推理服务部署拓扑",
    "multi-agent-langgraph.svg": "Multi-Agent 协作（LangGraph Supervisor）",
    "production-agent-pipeline.svg": "生产级 Agent 应用全链路",
}

MARKDOWN_EXTENSIONS = ["extra", "sane_lists", "toc", "pymdownx.superfences"]
LIST_ITEM_RE = re.compile(r"^((?:[-+*]\s+|\d+[.)]\s+))")
FENCE_RE = re.compile(r"^\s*(?:`{3,}|~{3,})")


@dataclass(frozen=True)
class Doc:
    path: Path
    rel_path: str
    title: str
    section_id: str
    group: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def is_list_item(line: str) -> bool:
    return bool(LIST_ITEM_RE.match(line))


def is_block_start(line: str) -> bool:
    if not line.strip():
        return False
    if FENCE_RE.match(line):
        return True
    if line[0].isspace():
        return False
    return is_table_line(line) or is_list_item(line) or line.startswith(">")


def is_same_block(previous: str, current: str) -> bool:
    return (
        (is_table_line(previous) and is_table_line(current))
        or (is_list_item(previous) and is_list_item(current))
        or (previous.startswith(">") and current.startswith(">"))
    )


def normalize_list_continuation_indent(line: str, continuation_indent: int | None) -> str:
    if continuation_indent is None or continuation_indent >= 4 or not line.strip():
        return line

    prefix = " " * continuation_indent
    if line.startswith(prefix) and not line.startswith(" " * 4):
        return " " * (4 - continuation_indent) + line
    return line


def normalize_markdown_blocks(text: str) -> str:
    """Insert block boundaries that GitHub renders implicitly but Python-Markdown requires."""
    normalized: list[str] = []
    in_fenced_code = False
    list_continuation_indent: int | None = None
    previous_was_closing_fence = False

    for raw_line in text.splitlines():
        if previous_was_closing_fence:
            if raw_line.strip():
                normalized.append("")
            previous_was_closing_fence = False

        if not in_fenced_code:
            list_match = LIST_ITEM_RE.match(raw_line)
            if list_match:
                list_continuation_indent = len(list_match.group(1))
            elif raw_line.strip() and not raw_line[0].isspace():
                list_continuation_indent = None

        line = normalize_list_continuation_indent(raw_line, list_continuation_indent)

        if not in_fenced_code and normalized and is_block_start(line):
            previous = normalized[-1]
            if previous.strip() and not is_same_block(previous, line):
                normalized.append("")

        normalized.append(line)

        is_fence = bool(FENCE_RE.match(line))
        is_closing_fence = is_fence and in_fenced_code
        if is_fence:
            in_fenced_code = not in_fenced_code
            previous_was_closing_fence = is_closing_fence

    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(normalized) + trailing_newline


def slugify(value: str) -> str:
    value = value.replace("/", "-").replace(" ", "-")
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "section"


def strip_numeric_prefix(title: str) -> str:
    return re.sub(r"^\d+[-_、.\s]*", "", title)


def extract_title(path: Path) -> str:
    text = read_text(path)
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return strip_numeric_prefix(path.stem)


def order_key(path: Path, ordered_names: list[str]) -> tuple[int, str]:
    rel = path.as_posix()
    try:
        return (ordered_names.index(rel), rel)
    except ValueError:
        return (len(ordered_names), rel)


def company_doc_key(path: Path) -> tuple[int, str]:
    order = {
        "岗位要求.md": 0,
        "面试题与面经.md": 1,
        "真实面经-牛客实录.md": 2,
        "真实面经-网络实录.md": 3,
    }
    return (order.get(path.name, 99), path.name)


def make_doc(path: Path, group: str) -> Doc:
    rel_path = path.relative_to(ROOT).as_posix()
    section_id = slugify(path.relative_to(ROOT).with_suffix("").as_posix())
    return Doc(
        path=path,
        rel_path=rel_path,
        title=extract_title(path),
        section_id=section_id,
        group=group,
    )


def collect_docs() -> OrderedDict[str, list[Doc]]:
    groups: OrderedDict[str, list[Doc]] = OrderedDict()
    seen: set[Path] = set()

    def add(group: str, paths: list[Path]) -> None:
        docs = []
        for path in paths:
            if not path.exists() or path in seen:
                continue
            seen.add(path)
            docs.append(make_doc(path, group))
        if docs:
            groups.setdefault(group, []).extend(docs)

    add("🗺️ 学习路线", [ROOT / name for name in ROOT_DOCS])

    general_dir = ROOT / "通用知识"
    general_paths = sorted(
        general_dir.glob("*.md"),
        key=lambda p: order_key(p.relative_to(general_dir), GENERAL_ORDER),
    )
    add("📚 通用知识", general_paths)

    project_dir = ROOT / "项目实战"
    project_paths = sorted(
        project_dir.rglob("*.md"),
        key=lambda p: order_key(p.relative_to(project_dir), PROJECT_ORDER),
    )
    add("🛠️ 项目实战", project_paths)

    for company in COMPANY_ORDER:
        company_dir = ROOT / company
        if not company_dir.exists():
            continue
        add(
            f"{COMPANY_ICONS.get(company, '🏢')} {company}",
            sorted(company_dir.glob("*.md"), key=company_doc_key),
        )

    remaining = sorted(
        path
        for path in ROOT.rglob("*.md")
        if path.name != "README.md"
        and path not in seen
        and ".git" not in path.parts
        and "dist" not in path.parts
        and "agents" not in path.parts
        and "templates" not in path.parts
        and not (path.parent == DATA_DIR and path.name.startswith("interview_candidates"))
    )
    add("📄 其他文档", remaining)

    return groups


def copy_data_assets() -> None:
    if not DATA_DIR.exists():
        return

    target = DIST / "data"
    target.mkdir(parents=True, exist_ok=True)
    for path in DATA_DIR.iterdir():
        if path.name.startswith("interview_candidates"):
            continue
        if path.is_dir():
            shutil.copytree(path, target / path.name)
        else:
            shutil.copy2(path, target / path.name)


def copy_interview_algorithm_page() -> None:
    if not INTERVIEW_ALGORITHMS_DIR.is_dir():
        return

    shutil.copytree(
        INTERVIEW_ALGORITHMS_DIR,
        DIST / INTERVIEW_ALGORITHMS_DIR.name,
    )


def build_md_link_map(docs: list[Doc]) -> dict[str, str]:
    return {doc.rel_path: doc.section_id for doc in docs}


def rewrite_markdown_links(markup: str, doc: Doc, md_link_map: dict[str, str]) -> str:
    def rewrite_href(match: re.Match[str]) -> str:
        href = html.unescape(match.group(1))
        if href.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        url, _, _ = href.partition("#")
        decoded = urllib.parse.unquote(url)
        if not decoded.endswith(".md"):
            target = (doc.path.parent / decoded).resolve()
            try:
                rel = target.relative_to(ROOT).as_posix()
            except ValueError:
                return match.group(0)
            if not target.exists():
                return match.group(0)
            return f'href="{html.escape(rel, quote=True)}"'

        target = (doc.path.parent / decoded).resolve()
        try:
            rel = target.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)

        section_id = md_link_map.get(rel)
        if not section_id:
            return match.group(0)

        return f'href="#{html.escape(section_id, quote=True)}" data-section-link'

    def rewrite_src(match: re.Match[str]) -> str:
        src = html.unescape(match.group(1))
        if src.startswith(("http://", "https://", "data:", "#")):
            return match.group(0)

        target = (doc.path.parent / urllib.parse.unquote(src)).resolve()
        try:
            rel = target.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)

        return f'src="{html.escape(rel, quote=True)}"'

    markup = re.sub(r'href="([^"]+)"', rewrite_href, markup)
    return re.sub(r'src="([^"]+)"', rewrite_src, markup)


def copy_canvas_assets() -> None:
    for path in ROOT.glob("*.canvas"):
        shutil.copy2(path, DIST / path.name)


def render_markdown_doc(doc: Doc, md_link_map: dict[str, str]) -> str:
    converted = markdown.markdown(
        normalize_markdown_blocks(read_text(doc.path)),
        extensions=MARKDOWN_EXTENSIONS,
        output_format="html5",
    )
    converted = rewrite_markdown_links(converted, doc, md_link_map)
    source_url = (
        "https://github.com/Zchary1106/agent-interview-hub/blob/main/"
        + urllib.parse.quote(doc.rel_path, safe="/")
    )
    return f"""
    <section class="content-section" id="{html.escape(doc.section_id, quote=True)}" data-title="{html.escape(doc.title, quote=True)}">
      <div class="section-header">
        <button class="back-home" type="button" data-target="welcome">返回备考首页</button>
        <p class="eyebrow">{html.escape(doc.group)}</p>
        <h2>{html.escape(doc.title)}</h2>
        <a class="source-link" href="{html.escape(source_url, quote=True)}" target="_blank" rel="noopener">查看源文件</a>
      </div>
      <article class="markdown-body">
        {converted}
      </article>
    </section>
    """


def load_questions() -> list[dict]:
    data_path = ROOT / "data.json"
    if not data_path.exists():
        return []
    return json.loads(read_text(data_path))


def render_sidebar(groups: OrderedDict[str, list[Doc]]) -> str:
    items = [
        """
        <div class="nav-section">
          <button class="nav-item nav-item-strong active" type="button" data-target="welcome">首页</button>
          <a class="nav-item nav-link" href="interview-questions.html">交互式面试题库</a>
          <a class="nav-item nav-link" href="面试算法题/">面试算法题图谱</a>
          <button class="nav-item" type="button" data-target="diagrams">架构图</button>
        </div>
        """
    ]
    for group, docs in groups.items():
        doc_items = "\n".join(
            f'<button class="nav-item" type="button" data-target="{html.escape(doc.section_id, quote=True)}">{html.escape(doc.title)}</button>'
            for doc in docs
        )
        items.append(
            f"""
            <div class="nav-section">
              <div class="nav-category-title">{html.escape(group)}</div>
              {doc_items}
            </div>
            """
        )
    return "\n".join(items)


def render_diagram_gallery() -> str:
    if not DIAGRAMS_DIR.exists():
        return "<p>暂无架构图。</p>"

    cards = []
    for svg in sorted(DIAGRAMS_DIR.glob("*.svg"), key=lambda p: list(DIAGRAM_TITLES).index(p.name) if p.name in DIAGRAM_TITLES else 999):
        title = DIAGRAM_TITLES.get(svg.name, strip_numeric_prefix(svg.stem))
        rel = f"diagrams/{svg.name}"
        cards.append(
            f"""
            <a class="diagram-card" href="{html.escape(rel, quote=True)}" target="_blank" rel="noopener">
              <img src="{html.escape(rel, quote=True)}" alt="{html.escape(title, quote=True)}" loading="lazy">
              <span>{html.escape(title)}</span>
            </a>
            """
        )
    return "\n".join(cards)


def render_index(groups: OrderedDict[str, list[Doc]]) -> str:
    docs = [doc for group_docs in groups.values() for doc in group_docs]
    md_link_map = build_md_link_map(docs)
    questions = load_questions()
    question_count = sum(len(company.get("questions", [])) for company in questions)
    sections = "\n".join(render_markdown_doc(doc, md_link_map) for doc in docs)
    sidebar = render_sidebar(groups)
    diagram_gallery = render_diagram_gallery()
    by_path = {doc.rel_path: doc.section_id for doc in docs}
    learning_target = by_path.get("Agent工程师学习路线图.md", "welcome")
    latest_target = by_path.get("通用知识/最新AI-Agent面经索引.md", "welcome")
    core_target = by_path.get("通用知识/Agent核心概念与设计模式.md", "welcome")
    project_target = by_path.get("项目实战/01-RAG知识问答系统.md", "welcome")
    company_docs_map = {
        company: [doc for doc in docs if doc.path.parent.name == company]
        for company in COMPANY_ORDER
    }
    company_count = sum(bool(company_docs) for company_docs in company_docs_map.values())

    def render_company_routes(companies: list[str]) -> str:
        return "\n".join(
            f"""
            <button class="company-route" type="button" data-target="{html.escape(company_docs_map[company][0].section_id, quote=True)}">
              <strong>{html.escape(company)}</strong>
              <small>{len(company_docs_map[company])} 篇资料</small>
              <span aria-hidden="true">→</span>
            </button>
            """
            for company in companies
            if company_docs_map.get(company)
        )

    domestic_company_cards = render_company_routes(
        COMPANY_ORDER[:9] + ["商汤科技"]
    )
    global_company_cards = render_company_routes(
        ["OpenAI", "Anthropic", "谷歌", "微软", "初创公司"]
    )
    library_groups = "\n".join(
        f"""
        <section class="library-group">
          <h3>{html.escape(group)}</h3>
          <div class="library-links">
            {"".join(f'<button type="button" data-target="{html.escape(doc.section_id, quote=True)}">{html.escape(doc.title)}</button>' for doc in group_docs)}
          </div>
        </section>
        """
        for group, group_docs in groups.items()
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Agent 面试知识库</title>
  <style>
    :root {{
      --bg: #f6f7f5;
      --panel: #ffffff;
      --panel-2: #eef1ed;
      --border: #dce1dc;
      --text: #20262b;
      --muted: #687169;
      --accent: #356b54;
      --accent-soft: #e5efe9;
      --accent-ink: #ffffff;
      --code: #20262b;
      --shadow: rgba(34, 45, 38, 0.08);
      --sidebar-width: 340px;
    }}
    [data-theme="dark"] {{
      --bg: #171b19;
      --panel: #202522;
      --panel-2: #2a302c;
      --border: #3a423d;
      --text: #eef1ee;
      --muted: #a3aca5;
      --accent: #82b49c;
      --accent-soft: #293c32;
      --accent-ink: #142219;
      --code: #101311;
      --shadow: rgba(0, 0, 0, 0.24);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      min-height: 100dvh;
      background: var(--bg);
      color: var(--text);
      font: 16px/1.72 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      transition: background-color 0.2s ease, color 0.2s ease;
    }}
    a {{ color: var(--accent); }}
    button, input {{ font: inherit; }}
    button:active, a:active {{ transform: translateY(1px); }}
    :focus-visible {{ outline: 3px solid var(--accent); outline-offset: 3px; }}
    .layout {{ min-height: 100dvh; }}
    .sidebar-backdrop {{
      position: fixed;
      inset: 0;
      z-index: 25;
      background: rgba(12, 16, 20, 0.38);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }}
    .sidebar-backdrop.open {{ opacity: 1; pointer-events: auto; }}
    .sidebar {{
      position: fixed;
      inset: 0 auto 0 0;
      z-index: 30;
      width: min(90vw, var(--sidebar-width));
      overflow: auto;
      padding-bottom: 28px;
      border-right: 1px solid var(--border);
      background: var(--panel);
      box-shadow: 24px 0 70px var(--shadow);
      transform: translateX(-105%);
      transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .sidebar.open {{ transform: translateX(0); }}
    .brand {{ padding: 28px 24px 20px; border-bottom: 1px solid var(--border); }}
    .brand h1 {{ margin: 0; font-size: 20px; letter-spacing: -0.04em; }}
    .brand p {{ margin: 5px 0 0; color: var(--muted); font-size: 12px; }}
    .nav-section {{ padding: 12px 10px; border-bottom: 1px solid var(--border); }}
    .nav-category-title {{ padding: 10px 12px 7px; color: var(--accent); font-size: 12px; font-weight: 700; }}
    .nav-item {{
      display: block;
      width: 100%;
      border: 0;
      border-radius: 8px;
      padding: 8px 12px 8px 18px;
      background: transparent;
      color: var(--muted);
      text-align: left;
      text-decoration: none;
      font-size: 13px;
      cursor: pointer;
      transition: background 0.18s ease, color 0.18s ease;
    }}
    .nav-item:hover, .nav-item.active {{ background: var(--accent-soft); color: var(--text); }}
    .nav-item-strong {{ color: var(--text); font-weight: 700; }}
    .main {{ min-width: 0; }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: auto auto minmax(220px, 620px) 1fr auto auto auto auto;
      gap: 10px;
      align-items: center;
      min-height: 70px;
      padding: 10px clamp(16px, 3vw, 40px);
      border-bottom: 1px solid var(--border);
      background: color-mix(in srgb, var(--bg) 92%, transparent);
      backdrop-filter: blur(16px);
    }}
    .topbar button, .topbar-link {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 12px;
      background: var(--panel);
      color: var(--text);
      text-decoration: none;
      cursor: pointer;
      white-space: nowrap;
      transition: border-color 0.18s ease, background 0.18s ease;
    }}
    .topbar button:hover, .topbar-link:hover {{ border-color: var(--accent); background: var(--accent-soft); }}
    .menu-btn {{ font-weight: 700; }}
    .brand-button {{ border-color: transparent !important; background: transparent !important; color: var(--text) !important; font-weight: 700; letter-spacing: -0.025em; }}
    .search {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 13px;
      background: var(--panel);
      color: var(--text);
    }}
    .search::placeholder {{ color: var(--muted); }}
    .topbar-spacer {{ min-width: 0; }}
    .reader-tool {{ color: var(--muted) !important; font-size: 13px; }}
    .content {{ width: min(1240px, calc(100% - 40px)); margin: 0 auto; padding: 32px 0 88px; }}
    .content-section {{ display: none; }}
    .content-section.active {{ display: block; animation: section-in 0.35s cubic-bezier(0.16, 1, 0.3, 1); }}
    @keyframes section-in {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .home-hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
      gap: clamp(36px, 6vw, 72px);
      align-items: end;
      min-height: 440px;
      padding: 64px 0 52px;
      border-bottom: 1px solid var(--border);
    }}
    .prep-kicker {{ margin: 0 0 14px; color: var(--accent); font-size: 14px; font-weight: 700; letter-spacing: 0; }}
    .hero-copy h1 {{ max-width: 13ch; margin: 0; font-size: clamp(38px, 5vw, 58px); line-height: 1.08; letter-spacing: -0.045em; text-wrap: balance; }}
    .hero-copy > p:not(.prep-kicker) {{ max-width: 38ch; margin: 20px 0 0; color: var(--muted); font-size: 17px; }}
    .hero-actions {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-top: 30px; }}
    .primary-action {{
      border: 1px solid var(--accent);
      border-radius: 8px;
      padding: 12px 17px;
      background: var(--accent);
      color: var(--accent-ink);
      font-weight: 700;
      cursor: pointer;
    }}
    .text-action {{ border: 0; padding: 10px 0; background: transparent; color: var(--accent); font-weight: 700; cursor: pointer; }}
    .prep-note {{ align-self: center; padding: 24px; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); }}
    .prep-note h2 {{ margin: 0 0 18px; font-size: 20px; letter-spacing: -0.03em; }}
    .prep-note ol {{ margin: 0; padding: 0; list-style: none; counter-reset: prep; }}
    .prep-note li {{ display: grid; grid-template-columns: 30px 1fr; gap: 10px; padding: 10px 0; border-top: 1px solid var(--border); color: var(--muted); }}
    .prep-note li::before {{ counter-increment: prep; content: counter(prep, decimal-leading-zero); color: var(--accent); font: 700 12px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .metric-strip {{ display: grid; grid-template-columns: repeat(3, 1fr); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }}
    .metric {{ padding: 22px 0; }}
    .metric + .metric {{ padding-left: 28px; border-left: 1px solid var(--border); }}
    .metric b {{ display: block; font: 750 30px/1 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--accent); }}
    .metric span {{ display: block; margin-top: 6px; color: var(--muted); font-size: 13px; }}
    .journey-section {{ padding-top: 72px; }}
    .section-lead {{ max-width: 620px; margin: 0 0 24px; }}
    .section-lead h2, .section-lead h3 {{ margin: 0; font-size: clamp(28px, 3.2vw, 40px); line-height: 1.12; letter-spacing: -0.04em; }}
    .section-lead p {{ margin: 12px 0 0; color: var(--muted); }}
    .path-grid {{ display: block; border-bottom: 1px solid var(--border); }}
    .path-card {{
      display: grid;
      grid-template-columns: minmax(150px, .35fr) minmax(0, 1fr) auto;
      gap: 28px;
      align-items: center;
      min-height: 112px;
      border: 0;
      border-top: 1px solid var(--border);
      border-radius: 0;
      padding: 22px 8px;
      background: transparent;
      color: var(--text);
      text-align: left;
      text-decoration: none;
      cursor: pointer;
      transition: background 0.2s ease, padding 0.2s ease;
    }}
    .path-card:hover {{ padding-left: 20px; background: var(--accent-soft); }}
    .path-card.featured {{ min-height: 128px; background: transparent; color: var(--text); }}
    .path-card small, .path-card.featured small {{ color: var(--muted); }}
    .path-card strong {{ max-width: none; margin: 0; font-size: clamp(20px, 2.4vw, 30px); line-height: 1.15; letter-spacing: -0.035em; }}
    .path-card span {{ margin: 0; color: var(--accent); font-weight: 800; }}
    .review-order {{ display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }}
    .review-step {{
      display: block;
      min-height: 150px;
      border: 0;
      border-right: 1px solid var(--border);
      padding: 22px;
      background: transparent;
      color: var(--text);
      text-align: left;
      text-decoration: none;
      cursor: pointer;
      transition: background 0.18s ease;
    }}
    .review-step:last-child {{ border-right: 0; }}
    .review-step:hover {{ background: var(--accent-soft); }}
    .review-step b {{ display: block; color: var(--accent); font: 700 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .review-step span {{ display: block; margin: 18px 0 5px; font-weight: 750; }}
    .review-step small {{ color: var(--muted); }}
    .company-board {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .company-cluster {{ padding: 24px; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); }}
    .company-cluster h3 {{ margin: 0 0 18px; color: var(--text); font-size: 17px; }}
    .company-routes {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .company-route {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 1px 12px;
      padding: 13px 0;
      border: 0;
      border-bottom: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      text-align: left;
      cursor: pointer;
    }}
    .company-route strong {{ font-size: 14px; }}
    .company-route small {{ color: var(--muted); }}
    .company-route span {{ grid-column: 2; grid-row: 1 / span 2; align-self: center; color: var(--accent); }}
    .company-route:hover strong {{ color: var(--accent); }}
    .library-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 34px 54px; }}
    .library-group {{ padding-top: 18px; border-top: 1px solid var(--border); }}
    .library-group h3 {{ margin: 0 0 12px; color: var(--accent); font-size: 15px; }}
    .library-links {{ display: flex; flex-wrap: wrap; gap: 7px 12px; }}
    .library-links button {{ border: 0; padding: 0; background: transparent; color: var(--muted); font: inherit; font-size: 14px; text-align: left; cursor: pointer; }}
    .library-links button:hover {{ color: var(--text); text-decoration: underline; text-decoration-color: var(--accent); text-underline-offset: 4px; }}
    .home-links {{ display: flex; flex-wrap: wrap; gap: 18px; margin-top: 56px; padding-top: 22px; border-top: 1px solid var(--border); }}
    .home-links a, .home-links button {{ border: 0; padding: 0; background: transparent; color: var(--muted); text-decoration: none; cursor: pointer; }}
    .home-links a:hover, .home-links button:hover {{ color: var(--accent); }}
    .content-section:not(#welcome) {{
      width: min(100%, 980px);
      margin: 24px auto 0;
      padding: clamp(28px, 5vw, 60px);
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
      box-shadow: 0 14px 42px var(--shadow);
    }}
    .section-header {{ display: grid; grid-template-columns: 1fr auto; gap: 8px 20px; margin-bottom: 34px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }}
    .back-home {{ grid-column: 1 / -1; justify-self: start; border: 0; padding: 0; background: transparent; color: var(--accent); font-weight: 700; cursor: pointer; }}
    .section-header h2 {{ grid-column: 1 / -1; max-width: 22ch; margin: 4px 0 0; font-size: clamp(30px, 4vw, 46px); line-height: 1.08; letter-spacing: -0.045em; text-wrap: balance; }}
    .eyebrow {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .source-link {{ color: var(--muted); font-size: 13px; }}
    .markdown-body {{ max-width: 76ch; min-width: 0; margin: 0 auto; overflow-wrap: anywhere; }}
    .markdown-body h1, .markdown-body h2 {{ margin-top: 1.8em; color: var(--text); letter-spacing: -0.035em; }}
    .markdown-body h3 {{ margin-top: 1.6em; color: var(--accent); }}
    .markdown-body p, .markdown-body li {{ color: color-mix(in srgb, var(--text) 88%, var(--muted)); }}
    .markdown-body pre {{ overflow: auto; padding: 18px; border-radius: 10px; background: var(--code); color: #edf0f2; }}
    .markdown-body code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .markdown-body :not(pre) > code {{ padding: 2px 5px; border-radius: 5px; background: var(--panel-2); }}
    .markdown-body table {{ display: block; width: 100%; overflow: auto; border-collapse: collapse; }}
    .markdown-body th, .markdown-body td {{ border: 1px solid var(--border); padding: 9px 11px; vertical-align: top; }}
    .markdown-body blockquote {{ margin-left: 0; padding: 10px 18px; border-left: 4px solid var(--accent); background: var(--accent-soft); color: var(--muted); }}
    .diagram-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .diagram-card {{ display: flex; flex-direction: column; gap: 10px; padding: 12px; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); text-decoration: none; color: var(--text); }}
    .diagram-card img {{ width: 100%; aspect-ratio: 16 / 10; object-fit: contain; border-radius: 7px; background: var(--code); }}
    .search-result {{ width: 100%; margin: 8px 0; padding: 14px; border: 1px solid var(--border); border-radius: 8px; color: var(--text); background: var(--bg); text-align: left; cursor: pointer; }}
    .search-result:hover {{ border-color: var(--accent); }}
    mark {{ background: var(--accent-soft); color: var(--text); }}
    @media (max-width: 1040px) {{
      .topbar {{ grid-template-columns: auto auto minmax(160px, 1fr) auto auto; }}
      .topbar-spacer, .reader-tool {{ display: none; }}
      .home-hero {{ grid-template-columns: 1fr; min-height: auto; }}
      .prep-note {{ max-width: 620px; }}
      .company-board {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 720px) {{
      .topbar {{ grid-template-columns: auto 1fr auto; }}
      .brand-button, .topbar-link {{ display: none; }}
      .content {{ width: min(100% - 28px, 1240px); padding-top: 10px; }}
      .home-hero {{ padding-top: 52px; }}
      .hero-copy h1 {{ font-size: clamp(44px, 14vw, 62px); }}
      .metric-strip {{ grid-template-columns: 1fr; }}
      .metric + .metric {{ padding-left: 0; border-left: 0; border-top: 1px solid var(--border); }}
      .review-order, .library-grid {{ grid-template-columns: 1fr; }}
      .path-card, .path-card.featured {{ grid-template-columns: 1fr; gap: 10px; min-height: 0; padding: 22px 0; }}
      .path-card:hover {{ padding-left: 12px; }}
      .review-step {{ border-right: 0; border-bottom: 1px solid var(--border); }}
      .review-step:last-child {{ border-bottom: 0; }}
      .company-routes {{ grid-template-columns: 1fr; }}
      .content-section:not(#welcome) {{ margin-top: 14px; border-radius: 12px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{ scroll-behavior: auto !important; animation: none !important; transition: none !important; }}
    }}
  </style>
</head>
<body data-theme="dark">
  <div class="sidebar-backdrop" id="sidebarBackdrop"></div>
  <div class="layout">
    <aside class="sidebar" id="sidebar">
      <div class="brand">
        <h1>Agent Interview Hub</h1>
        <p>学习路线、公司面经、题库与项目准备</p>
      </div>
      <nav>{sidebar}</nav>
    </aside>
    <main class="main">
      <div class="topbar">
        <button class="menu-btn" type="button" id="menuBtn">资料目录</button>
        <button class="brand-button" type="button" data-target="welcome">Agent Interview Hub</button>
        <input class="search" id="searchInput" type="search" placeholder="搜索知识点、公司、题目..." autocomplete="off">
        <span class="topbar-spacer"></span>
        <button class="reader-tool" type="button" id="expandBtn">全部展开</button>
        <button class="reader-tool" type="button" id="collapseBtn">全部收起</button>
        <a class="topbar-link" href="interview-questions.html">进入题库</a>
        <button type="button" id="themeBtn"><span class="theme-label">浅色</span></button>
      </div>
      <div class="content">
        <section class="content-section active" id="welcome" data-title="首页">
          <div class="home-hero">
            <div class="hero-copy">
              <p class="prep-kicker">AI Agent 面试准备</p>
              <h1>从目标岗位开始，按顺序准备下一场面试。</h1>
              <p>先看岗位和面经，再补知识、练题、整理项目表达。</p>
              <div class="hero-actions">
                <button class="primary-action" type="button" data-target="{html.escape(latest_target, quote=True)}">开始一轮准备</button>
                <button class="text-action" type="button" data-target="{html.escape(learning_target, quote=True)}">查看完整路线 →</button>
              </div>
            </div>
            <aside class="prep-note">
              <h2>如果明天面试</h2>
              <ol>
                <li>确认岗位要求和近期面经</li>
                <li>限时回答 5 道高频问题</li>
                <li>完整讲述 1 个项目</li>
                <li>准备追问与反问</li>
              </ol>
            </aside>
          </div>

          <div class="metric-strip">
            <div class="metric"><b>{len(docs)}</b><span>篇现有资料</span></div>
            <div class="metric"><b>{company_count}</b><span>家公司与岗位类别</span></div>
            <div class="metric"><b>{question_count}</b><span>道可交互练习题</span></div>
          </div>

          <section class="journey-section">
            <div class="section-lead">
              <h2>按你离面试的时间进入</h2>
              <p>不要从资料目录第一篇开始。先选择最符合当前状态的路径。</p>
            </div>
            <div class="path-grid">
              <button class="path-card featured" type="button" data-target="{html.escape(learning_target, quote=True)}">
                <small>尚未开始投递</small>
                <strong>先建立 Agent 工程师的完整知识框架</strong>
                <span>进入学习路线 →</span>
              </button>
              <a class="path-card" href="interview-questions.html">
                <small>1-2 周内有面试</small>
                <strong>用题库定位短板</strong>
                <span>开始限时自测 →</span>
              </a>
              <button class="path-card" type="button" data-target="{html.escape(latest_target, quote=True)}">
                <small>已经拿到面试邀约</small>
                <strong>围绕目标公司准备</strong>
                <span>查看近期面经 →</span>
              </button>
            </div>
          </section>

          <section class="journey-section">
            <div class="section-lead">
              <h2>完成一轮有效准备</h2>
              <p>按这个顺序走完，比同时打开十篇资料更有效。</p>
            </div>
            <div class="review-order">
              <button class="review-step" type="button" data-target="{html.escape(latest_target, quote=True)}"><b>01</b><span>校准岗位</span><small>先看近期面经与目标公司要求</small></button>
              <button class="review-step" type="button" data-target="{html.escape(core_target, quote=True)}"><b>02</b><span>补核心概念</span><small>用 Agent、RAG、工具调用打底</small></button>
              <a class="review-step" href="interview-questions.html"><b>03</b><span>限时自测</span><small>按公司筛选题目并记录卡点</small></a>
              <button class="review-step" type="button" data-target="{html.escape(project_target, quote=True)}"><b>04</b><span>准备项目表达</span><small>把项目讲成可追问的系统设计</small></button>
            </div>
          </section>

          <section class="journey-section">
            <div class="section-lead">
              <h2>先准备目标公司</h2>
              <p>岗位要求、面试问题和公开面经放在同一个入口里。</p>
            </div>
            <div class="company-board">
              <div class="company-cluster">
                <h3>国内公司</h3>
                <div class="company-routes">{domestic_company_cards}</div>
              </div>
              <div class="company-cluster">
                <h3>海外与初创公司</h3>
                <div class="company-routes">{global_company_cards}</div>
              </div>
            </div>
          </section>

          <section class="journey-section">
            <div class="section-lead">
              <h2>需要时，再深入资料库</h2>
              <p>现有内容全部保留。点击标题进入沉浸式单篇阅读。</p>
            </div>
            <div class="library-grid">{library_groups}</div>
          </section>

          <div class="home-links">
            <button type="button" data-target="diagrams">查看架构图</button>
            <a href="index.html">返回经典版</a>
            <a href="面试算法题/">打开算法题图谱</a>
            <a href="https://github.com/Zchary1106/agent-interview-hub" target="_blank" rel="noopener">GitHub 仓库</a>
          </div>
        </section>

        <section class="content-section" id="diagrams" data-title="架构图">
          <div class="section-header">
            <button class="back-home" type="button" data-target="welcome">返回备考首页</button>
            <p class="eyebrow">架构图</p>
            <h2>架构图</h2>
          </div>
          <div class="diagram-grid">{diagram_gallery}</div>
        </section>

        <section class="content-section" id="search-results" data-title="搜索结果">
          <div class="section-header">
            <button class="back-home" type="button" data-target="welcome">返回备考首页</button>
            <p class="eyebrow">搜索</p>
            <h2>搜索结果</h2>
          </div>
          <div id="searchResults"></div>
        </section>

        {sections}
      </div>
    </main>
  </div>
  <script>
    const sections = [...document.querySelectorAll('.content-section')];
    const navItems = [...document.querySelectorAll('[data-target]')];
    let currentSection = 'welcome';

    function showSection(id, updateHash = true) {{
      const target = document.getElementById(id) || document.getElementById('welcome');
      sections.forEach(section => section.classList.toggle('active', section === target));
      navItems.forEach(item => item.classList.toggle('active', item.dataset.target === target.id));
      currentSection = target.id;
      document.getElementById('sidebar').classList.remove('open');
      document.getElementById('sidebarBackdrop').classList.remove('open');
      window.scrollTo({{ top: 0, behavior: 'auto' }});
      if (updateHash && target.id !== 'welcome') {{
        history.replaceState(null, '', '#' + encodeURIComponent(target.id));
      }} else if (updateHash) {{
        history.replaceState(null, '', location.pathname);
      }}
    }}

    function renderSearch(query) {{
      const normalized = query.trim().toLowerCase();
      if (normalized.length < 2) {{
        if (currentSection === 'search-results') showSection('welcome');
        return;
      }}
      const results = sections
        .filter(section => !['welcome', 'search-results'].includes(section.id))
        .map(section => ({{
          id: section.id,
          title: section.dataset.title || section.querySelector('h2')?.textContent || section.id,
          text: section.textContent.toLowerCase()
        }}))
        .filter(section => section.text.includes(normalized))
        .slice(0, 80);

      const box = document.getElementById('searchResults');
      box.textContent = '';
      const count = document.createElement('p');
      count.textContent = `找到 ${{results.length}} 个结果`;
      box.appendChild(count);
      if (results.length === 0) {{
        const empty = document.createElement('p');
        empty.textContent = '未找到相关内容。';
        box.appendChild(empty);
      }}
      results.forEach(result => {{
        const button = document.createElement('button');
        button.className = 'search-result';
        button.type = 'button';
        button.textContent = result.title;
        button.addEventListener('click', () => showSection(result.id));
        box.appendChild(button);
      }});
      showSection('search-results', false);
    }}

    document.addEventListener('click', event => {{
      const target = event.target.closest('[data-target]');
      if (target) {{
        event.preventDefault();
        showSection(target.dataset.target);
      }}
      const sectionLink = event.target.closest('[data-section-link]');
      if (sectionLink) {{
        event.preventDefault();
        showSection(decodeURIComponent(sectionLink.getAttribute('href').slice(1)));
      }}
    }});

    document.getElementById('searchInput').addEventListener('input', event => renderSearch(event.target.value));
    document.getElementById('menuBtn').addEventListener('click', () => {{
      document.getElementById('sidebar').classList.toggle('open');
      document.getElementById('sidebarBackdrop').classList.toggle('open');
    }});
    document.getElementById('sidebarBackdrop').addEventListener('click', () => {{
      document.getElementById('sidebar').classList.remove('open');
      document.getElementById('sidebarBackdrop').classList.remove('open');
    }});
    document.getElementById('themeBtn').addEventListener('click', event => {{
      const dark = document.body.getAttribute('data-theme') === 'dark';
      if (dark) {{
        document.body.removeAttribute('data-theme');
        event.currentTarget.innerHTML = '<span class="theme-label">深色</span>';
      }} else {{
        document.body.setAttribute('data-theme', 'dark');
        event.currentTarget.innerHTML = '<span class="theme-label">浅色</span>';
      }}
    }});
    document.getElementById('expandBtn').addEventListener('click', () => document.querySelectorAll('details').forEach(detail => detail.open = true));
    document.getElementById('collapseBtn').addEventListener('click', () => document.querySelectorAll('details').forEach(detail => detail.open = false));
    document.addEventListener('keydown', event => {{
      if (event.key === '/' && document.activeElement !== document.getElementById('searchInput')) {{
        event.preventDefault();
        document.getElementById('searchInput').focus();
      }}
    }});
    if (location.hash) showSection(decodeURIComponent(location.hash.slice(1)), false);
  </script>
</body>
</html>
"""


def render_interview_questions() -> str:
    data = load_questions()
    question_count = sum(len(company.get("questions", [])) for company in data)
    json_text = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Agent 面试题库</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0a0a0a; color: #e5e7eb; font: 15px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color: #07c160; }}
    .header {{ position: sticky; top: 0; z-index: 10; background: rgba(17, 17, 17, 0.95); border-bottom: 1px solid #222; padding: 22px 20px; backdrop-filter: blur(10px); }}
    .header-inner, .tags, .controls, .main {{ max-width: 980px; margin: 0 auto; }}
    h1 {{ margin: 0 0 10px; font-size: 26px; }}
    h1 span, .stats b {{ color: #07c160; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 18px; color: #8b949e; margin-bottom: 14px; }}
    .stats b {{ font-size: 20px; }}
    .search {{ width: 100%; padding: 11px 14px; border: 1px solid #333; border-radius: 10px; background: #171717; color: #e5e7eb; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 16px 20px 0; }}
    .tag, .ctrl-btn {{ border: 1px solid #333; border-radius: 999px; color: #aaa; background: #171717; cursor: pointer; }}
    .tag {{ padding: 6px 13px; }}
    .tag:hover, .tag.active, .ctrl-btn:hover {{ border-color: #07c160; color: #07c160; }}
    .tag.active {{ background: #07c160; color: #000; font-weight: 700; }}
    .controls {{ display: flex; gap: 10px; padding: 16px 20px; }}
    .ctrl-btn {{ border-radius: 8px; padding: 7px 14px; }}
    .main {{ padding: 0 20px 30px; }}
    .company {{ margin-bottom: 16px; border: 1px solid #242424; border-radius: 14px; overflow: hidden; background: #111; }}
    .company-header, .q-header {{ width: 100%; border: 0; color: inherit; background: transparent; text-align: left; cursor: pointer; }}
    .company-header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 16px 18px; font-size: 18px; font-weight: 700; }}
    .company-header:hover, .q-header:hover {{ background: #1a1a1a; }}
    .company-count {{ color: #07c160; font-size: 13px; background: rgba(7, 193, 96, 0.12); border-radius: 999px; padding: 2px 10px; }}
    .company-body {{ display: none; border-top: 1px solid #222; }}
    .company.open .company-body {{ display: block; }}
    .question {{ border-bottom: 1px solid #1f1f1f; }}
    .question:last-child {{ border-bottom: 0; }}
    .q-header {{ display: flex; gap: 12px; padding: 14px 18px; }}
    .q-num {{ min-width: 36px; color: #07c160; font-weight: 800; }}
    .q-text {{ flex: 1; font-weight: 600; }}
    .q-body {{ display: none; padding: 0 18px 16px 66px; }}
    .question.open .q-body {{ display: block; }}
    .q-section {{ margin-top: 12px; }}
    .q-section-title {{ margin-bottom: 6px; font-size: 13px; font-weight: 700; }}
    .thinking {{ color: #faad14; }}
    .answer {{ color: #07c160; }}
    .q-section-content {{ white-space: pre-wrap; border-radius: 10px; background: #1a1a1a; color: #c9d1d9; padding: 12px 14px; }}
    mark {{ background: rgba(7, 193, 96, 0.28); color: #e5e7eb; }}
    .footer {{ border-top: 1px solid #222; margin-top: 30px; padding: 32px 20px; text-align: center; color: #666; }}
    @media (max-width: 640px) {{ .q-body {{ padding-left: 18px; }} .stats {{ gap: 10px; }} }}
  </style>
</head>
<body>
  <header class="header">
    <div class="header-inner">
      <h1>🤖 AI Agent <span>面试题库</span></h1>
      <div class="stats">
        <div><b id="totalCompanies">0</b> 个分类</div>
        <div><b id="totalQuestions">0</b> 道面试题</div>
        <div><b id="visibleQuestions">0</b> 道匹配</div>
        <div><a href="index.html">返回知识库首页</a></div>
      </div>
      <input class="search" id="searchInput" type="search" placeholder="搜索题目、思考逻辑或参考答案..." autocomplete="off">
    </div>
  </header>
  <div class="tags" id="tagsContainer"></div>
  <div class="controls">
    <button class="ctrl-btn" type="button" id="expandBtn">📖 展开全部</button>
    <button class="ctrl-btn" type="button" id="collapseBtn">📕 收起全部</button>
    <button class="ctrl-btn" type="button" id="resetBtn">🔄 重置筛选</button>
  </div>
  <main class="main" id="mainContainer"></main>
  <footer class="footer">共收录 {question_count} 道面试题，数据来自 <code>data.json</code>。</footer>
  <script type="application/json" id="question-data">{json_text}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('question-data').textContent);
    const activeTags = new Set();
    let searchTerm = '';
    let timer = null;

    function escapeRegExp(value) {{
      return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
    }}

    function appendHighlighted(parent, text) {{
      if (!searchTerm) {{
        parent.textContent = text;
        return;
      }}
      const parts = text.split(new RegExp(`(${{escapeRegExp(searchTerm)}})`, 'gi'));
      for (const part of parts) {{
        if (!part) continue;
        if (part.toLowerCase() === searchTerm) {{
          const mark = document.createElement('mark');
          mark.textContent = part;
          parent.appendChild(mark);
        }} else {{
          parent.appendChild(document.createTextNode(part));
        }}
      }}
    }}

    function renderTags() {{
      const tags = document.getElementById('tagsContainer');
      tags.textContent = '';
      DATA.forEach(company => {{
        const tag = document.createElement('button');
        tag.className = 'tag';
        tag.type = 'button';
        tag.textContent = `${{company.icon || '🏢'}} ${{company.company}}`;
        tag.addEventListener('click', () => {{
          if (activeTags.has(company.company)) activeTags.delete(company.company);
          else activeTags.add(company.company);
          tag.classList.toggle('active', activeTags.has(company.company));
          render();
        }});
        tags.appendChild(tag);
      }});
    }}

    function render() {{
      const main = document.getElementById('mainContainer');
      main.textContent = '';
      let visible = 0;

      DATA.forEach(company => {{
        if (activeTags.size && !activeTags.has(company.company)) return;
        const questions = company.questions.filter(item => {{
          if (!searchTerm) return true;
          return [item.question, item.thinking, item.answer].some(value => (value || '').toLowerCase().includes(searchTerm));
        }});
        if (!questions.length) return;
        visible += questions.length;

        const companyEl = document.createElement('section');
        companyEl.className = 'company open';

        const header = document.createElement('button');
        header.className = 'company-header';
        header.type = 'button';
        header.addEventListener('click', () => companyEl.classList.toggle('open'));
        const title = document.createElement('span');
        title.textContent = `${{company.icon || '🏢'}} ${{company.company}}`;
        const count = document.createElement('span');
        count.className = 'company-count';
        count.textContent = `${{questions.length}} 题`;
        header.append(title, count);

        const body = document.createElement('div');
        body.className = 'company-body';
        questions.forEach((item, index) => {{
          const question = document.createElement('article');
          question.className = 'question';

          const qHeader = document.createElement('button');
          qHeader.className = 'q-header';
          qHeader.type = 'button';
          qHeader.addEventListener('click', () => question.classList.toggle('open'));
          const num = document.createElement('span');
          num.className = 'q-num';
          num.textContent = `Q${{index + 1}}`;
          const qText = document.createElement('span');
          qText.className = 'q-text';
          appendHighlighted(qText, item.question || '');
          qHeader.append(num, qText);

          const qBody = document.createElement('div');
          qBody.className = 'q-body';
          const thinking = document.createElement('div');
          thinking.className = 'q-section';
          thinking.innerHTML = '<div class="q-section-title thinking">💡 思考逻辑</div>';
          const thinkingContent = document.createElement('div');
          thinkingContent.className = 'q-section-content';
          appendHighlighted(thinkingContent, item.thinking || '');
          thinking.appendChild(thinkingContent);

          const answer = document.createElement('div');
          answer.className = 'q-section';
          answer.innerHTML = '<div class="q-section-title answer">✅ 参考答案</div>';
          const answerContent = document.createElement('div');
          answerContent.className = 'q-section-content';
          appendHighlighted(answerContent, item.answer || '');
          answer.appendChild(answerContent);

          qBody.append(thinking, answer);
          question.append(qHeader, qBody);
          body.appendChild(question);
        }});

        companyEl.append(header, body);
        main.appendChild(companyEl);
      }});
      document.getElementById('visibleQuestions').textContent = visible;
    }}

    function resetFilters() {{
      activeTags.clear();
      searchTerm = '';
      document.getElementById('searchInput').value = '';
      document.querySelectorAll('.tag').forEach(tag => tag.classList.remove('active'));
      render();
    }}

    document.getElementById('totalCompanies').textContent = DATA.length;
    document.getElementById('totalQuestions').textContent = DATA.reduce((sum, company) => sum + company.questions.length, 0);
    document.getElementById('searchInput').addEventListener('input', event => {{
      clearTimeout(timer);
      timer = setTimeout(() => {{
        searchTerm = event.target.value.trim().toLowerCase();
        render();
      }}, 150);
    }});
    document.getElementById('expandBtn').addEventListener('click', () => document.querySelectorAll('.company,.question').forEach(el => el.classList.add('open')));
    document.getElementById('collapseBtn').addEventListener('click', () => document.querySelectorAll('.company,.question').forEach(el => el.classList.remove('open')));
    document.getElementById('resetBtn').addEventListener('click', resetFilters);
    renderTags();
    render();
  </script>
</body>
</html>
"""


def copy_legacy_pages() -> None:
    legacy_pages = ("index.html", "interview-questions.html")
    missing = [name for name in legacy_pages if not (LEGACY_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing legacy page source files: {', '.join(missing)}"
        )

    for name in legacy_pages:
        shutil.copy2(LEGACY_DIR / name, DIST / name)


def build() -> None:
    groups = collect_docs()
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    if DIAGRAMS_DIR.exists():
        shutil.copytree(DIAGRAMS_DIR, DIST / "diagrams")

    copy_data_assets()
    copy_canvas_assets()
    copy_interview_algorithm_page()
    copy_legacy_pages()

    new_index = render_index(groups).replace(
        'href="interview-questions.html"', 'href="new-interview-questions.html"'
    )
    new_questions = render_interview_questions().replace(
        'href="index.html"', 'href="new.html"'
    )
    (DIST / "new.html").write_text(new_index, encoding="utf-8")
    (DIST / "new-interview-questions.html").write_text(
        new_questions, encoding="utf-8"
    )

    doc_count = sum(len(docs) for docs in groups.values())
    question_count = sum(len(company.get("questions", [])) for company in load_questions())
    print(f"Built dist/: {doc_count} docs, {question_count} questions")


if __name__ == "__main__":
    build()
