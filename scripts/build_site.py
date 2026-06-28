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

ROOT_DOCS = ["Agent工程师学习路线图.md"]

GENERAL_ORDER = [
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
    )
    add("📄 其他文档", remaining)

    return groups


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
            return match.group(0)

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
    question_company_count = len(questions)
    company_count = sum(1 for name in COMPANY_ORDER if (ROOT / name).exists())
    diagram_count = len(list(DIAGRAMS_DIR.glob("*.svg"))) if DIAGRAMS_DIR.exists() else 0

    sections = "\n".join(render_markdown_doc(doc, md_link_map) for doc in docs)
    sidebar = render_sidebar(groups)
    diagram_gallery = render_diagram_gallery()

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Agent 面试知识库</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #f59e0b;
      --accent-2: #fbbf24;
      --green: #10b981;
      --sidebar-width: 300px;
    }}
    [data-theme="light"] {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --panel-2: #f1f5f9;
      --border: #cbd5e1;
      --text: #0f172a;
      --muted: #475569;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 16px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color: var(--accent-2); }}
    .layout {{ display: grid; grid-template-columns: var(--sidebar-width) minmax(0, 1fr); min-height: 100vh; }}
    .sidebar {{ position: sticky; top: 0; height: 100vh; overflow: auto; background: var(--panel); border-right: 1px solid var(--border); padding-bottom: 24px; }}
    .brand {{ padding: 22px 20px; border-bottom: 1px solid var(--border); }}
    .brand h1 {{ margin: 0; color: var(--accent); font-size: 20px; }}
    .brand p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    .nav-section {{ padding: 8px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.12); }}
    .nav-category-title {{ color: var(--accent); font-weight: 700; font-size: 13px; padding: 10px 20px 6px; }}
    .nav-item {{ display: block; width: 100%; border: 0; background: transparent; color: var(--muted); text-align: left; padding: 7px 20px 7px 32px; font: inherit; font-size: 13px; cursor: pointer; text-decoration: none; border-left: 3px solid transparent; }}
    .nav-item:hover, .nav-item.active {{ color: var(--accent-2); background: rgba(245, 158, 11, 0.12); border-left-color: var(--accent); }}
    .nav-item-strong {{ font-weight: 700; padding-left: 20px; color: var(--text); }}
    .nav-link {{ padding-left: 20px; }}
    .main {{ min-width: 0; }}
    .topbar {{ position: sticky; top: 0; z-index: 10; display: flex; gap: 12px; align-items: center; padding: 12px 24px; border-bottom: 1px solid var(--border); background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px); }}
    [data-theme="light"] .topbar {{ background: rgba(248, 250, 252, 0.88); }}
    .menu-btn {{ display: none; }}
    .search {{ flex: 1; max-width: 620px; border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; color: var(--text); background: var(--panel); }}
    .topbar button {{ border: 1px solid var(--border); border-radius: 10px; padding: 9px 12px; color: var(--text); background: var(--panel-2); cursor: pointer; }}
    .content {{ max-width: 1080px; margin: 0 auto; padding: 28px; }}
    .content-section {{ display: none; }}
    .content-section.active {{ display: block; }}
    .hero {{ padding: 36px; border: 1px solid var(--border); border-radius: 22px; background: linear-gradient(135deg, rgba(245, 158, 11, 0.16), rgba(59, 130, 246, 0.08)); }}
    .hero h2 {{ margin: 0 0 10px; color: var(--accent); font-size: 32px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 22px 0; }}
    .stat {{ padding: 16px; border: 1px solid var(--border); border-radius: 16px; background: var(--panel); }}
    .stat b {{ display: block; color: var(--accent-2); font-size: 28px; }}
    .quick-links {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .quick-links a, .quick-links button {{ border: 1px solid var(--border); border-radius: 999px; padding: 8px 14px; color: var(--text); background: var(--panel); text-decoration: none; cursor: pointer; }}
    .section-header {{ display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 8px; margin-bottom: 18px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }}
    .section-header h2 {{ width: 100%; margin: 0; color: var(--accent); font-size: 28px; }}
    .eyebrow {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .source-link {{ font-size: 13px; }}
    .markdown-body {{ min-width: 0; overflow-wrap: anywhere; }}
    .markdown-body h1, .markdown-body h2 {{ color: var(--accent); }}
    .markdown-body h3 {{ color: var(--accent-2); }}
    .markdown-body pre {{ overflow: auto; padding: 16px; border-radius: 12px; background: #020617; }}
    .markdown-body code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .markdown-body :not(pre) > code {{ padding: 2px 5px; border-radius: 6px; background: var(--panel-2); }}
    .markdown-body table {{ display: block; width: 100%; overflow: auto; border-collapse: collapse; }}
    .markdown-body th, .markdown-body td {{ border: 1px solid var(--border); padding: 8px 10px; vertical-align: top; }}
    .markdown-body blockquote {{ margin-left: 0; padding: 8px 16px; border-left: 4px solid var(--accent); background: rgba(245, 158, 11, 0.08); color: var(--muted); }}
    .diagram-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .diagram-card {{ display: flex; flex-direction: column; gap: 10px; padding: 14px; border: 1px solid var(--border); border-radius: 16px; background: var(--panel); text-decoration: none; color: var(--text); }}
    .diagram-card img {{ width: 100%; aspect-ratio: 16 / 10; object-fit: contain; border-radius: 12px; background: #020617; }}
    .search-result {{ width: 100%; margin: 8px 0; padding: 12px 14px; border: 1px solid var(--border); border-radius: 12px; color: var(--text); background: var(--panel); text-align: left; cursor: pointer; }}
    mark {{ background: rgba(245, 158, 11, 0.35); color: var(--text); }}
    @media (max-width: 860px) {{
      .layout {{ display: block; }}
      .sidebar {{ position: fixed; inset: 0 auto 0 0; width: min(86vw, var(--sidebar-width)); transform: translateX(-100%); transition: transform 0.2s ease; z-index: 20; }}
      .sidebar.open {{ transform: translateX(0); }}
      .menu-btn {{ display: inline-block; }}
      .topbar {{ padding: 10px 12px; }}
      .content {{ padding: 18px; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar" id="sidebar">
      <div class="brand">
        <h1>🤖 Agent Interview Hub</h1>
        <p>由 Markdown 自动构建的 AI Agent 面试知识库</p>
      </div>
      <nav>{sidebar}</nav>
    </aside>
    <main class="main">
      <div class="topbar">
        <button class="menu-btn" type="button" id="menuBtn">☰</button>
        <input class="search" id="searchInput" type="search" placeholder="搜索知识点、公司、题目..." autocomplete="off">
        <button type="button" id="expandBtn">展开</button>
        <button type="button" id="collapseBtn">折叠</button>
        <button type="button" id="themeBtn">🌙</button>
      </div>
      <div class="content">
        <section class="content-section active" id="welcome" data-title="首页">
          <div class="hero">
            <h2>AI Agent 工程师面试知识库</h2>
            <p>覆盖通用知识、项目实战、公司面经与交互式面试题库。页面由 <code>scripts/build_site.py</code> 从仓库内容自动生成。</p>
            <div class="stats">
              <div class="stat"><b>{len(docs)}</b><span>篇文档</span></div>
              <div class="stat"><b>{company_count}</b><span>家公司/类别</span></div>
              <div class="stat"><b>{question_count}</b><span>道交互题</span></div>
              <div class="stat"><b>{diagram_count}</b><span>张架构图</span></div>
            </div>
            <div class="quick-links">
              <button type="button" data-target="diagrams">查看架构图</button>
              <a href="interview-questions.html">打开交互式题库（{question_company_count} 类）</a>
              <a href="https://github.com/Zchary1106/agent-interview-hub" target="_blank" rel="noopener">GitHub 仓库</a>
            </div>
          </div>
        </section>

        <section class="content-section" id="diagrams" data-title="架构图">
          <div class="section-header">
            <p class="eyebrow">📊 Diagrams</p>
            <h2>架构图</h2>
          </div>
          <div class="diagram-grid">{diagram_gallery}</div>
        </section>

        <section class="content-section" id="search-results" data-title="搜索结果">
          <div class="section-header">
            <p class="eyebrow">🔎 Search</p>
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
    document.getElementById('menuBtn').addEventListener('click', () => document.getElementById('sidebar').classList.toggle('open'));
    document.getElementById('themeBtn').addEventListener('click', event => {{
      const light = document.body.getAttribute('data-theme') === 'light';
      if (light) {{
        document.body.removeAttribute('data-theme');
        event.currentTarget.textContent = '🌙';
      }} else {{
        document.body.setAttribute('data-theme', 'light');
        event.currentTarget.textContent = '☀️';
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


def build() -> None:
    groups = collect_docs()
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    if DIAGRAMS_DIR.exists():
        shutil.copytree(DIAGRAMS_DIR, DIST / "diagrams")

    (DIST / "index.html").write_text(render_index(groups), encoding="utf-8")
    (DIST / "interview-questions.html").write_text(render_interview_questions(), encoding="utf-8")

    doc_count = sum(len(docs) for docs in groups.values())
    question_count = sum(len(company.get("questions", [])) for company in load_questions())
    print(f"Built dist/: {doc_count} docs, {question_count} questions")


if __name__ == "__main__":
    build()
