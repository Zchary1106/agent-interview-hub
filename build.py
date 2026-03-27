#!/usr/bin/env python3
"""Build the single-file static website for agent-interview-hub."""

import os
import re
import html
import glob
import json

BASE = os.path.expanduser("~/agent-interview-hub")

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def read_svg(path):
    return read_file(path)

# Read all SVGs
svg_files = {}
for f in sorted(glob.glob(os.path.join(BASE, "diagrams", "*.svg"))):
    name = os.path.basename(f).replace('.svg', '')
    svg_files[name] = read_file(f)

# Diagram metadata
diagram_meta = {
    'agent-architecture': {'title': 'Agent 核心架构', 'section': 'Agent核心概念与设计模式'},
    'react-loop': {'title': 'ReAct 循环图', 'section': 'Agent核心概念与设计模式'},
    'multi-agent-patterns': {'title': 'Multi-Agent 协作模式', 'section': 'Agent核心概念与设计模式'},
    'rag-pipeline': {'title': 'RAG 工作流程图', 'section': 'RAG核心知识与面试题'},
    'rag-vs-finetune': {'title': 'RAG vs Fine-tuning 对比', 'section': 'RAG核心知识与面试题'},
    'langgraph-architecture': {'title': 'LangGraph 图结构示意', 'section': 'LangChain与LangGraph深度解析'},
    'framework-decision-tree': {'title': 'Agent 框架选型决策树', 'section': 'Agent框架全景'},
}

# Map section names to diagram keys
section_to_diagrams = {}
for key, meta in diagram_meta.items():
    sec = meta['section']
    if sec not in section_to_diagrams:
        section_to_diagrams[sec] = []
    section_to_diagrams[sec].append(key)

# Collect all content files
companies = ['字节跳动', '阿里巴巴', '腾讯', '百度', '美团', '小红书', '快手', '蚂蚁集团', '华为']
general_topics = [
    'Agent核心概念与设计模式',
    'Agent框架全景', 
    'LangChain与LangGraph深度解析',
    'RAG核心知识与面试题',
    '技术知识点汇总',
    '核心概念详解与参考答案',
    '高频拷打题-牛客热帖',
    '八股文题库-DataWhale开源',
    '其他公司面经-快手携程等',
]

def md_to_html(md_text):
    """Simple markdown to HTML converter."""
    lines = md_text.split('\n')
    html_parts = []
    in_code = False
    in_list = False
    in_table = False
    code_lang = ''
    code_lines = []
    table_rows = []
    
    def flush_list():
        nonlocal in_list
        if in_list:
            html_parts.append('</ul>')
            in_list = False
    
    def flush_table():
        nonlocal in_table, table_rows
        if in_table and table_rows:
            html_parts.append('<div class="table-wrap"><table>')
            for i, row in enumerate(table_rows):
                cells = [c.strip() for c in row.split('|')]
                cells = [c for c in cells if c != '']
                if i == 0:
                    html_parts.append('<thead><tr>')
                    for c in cells:
                        html_parts.append(f'<th>{process_inline(c)}</th>')
                    html_parts.append('</tr></thead><tbody>')
                elif i == 1:
                    continue  # separator row
                else:
                    html_parts.append('<tr>')
                    for c in cells:
                        html_parts.append(f'<td>{process_inline(c)}</td>')
                    html_parts.append('</tr>')
            html_parts.append('</tbody></table></div>')
            table_rows = []
            in_table = False
    
    def process_inline(text):
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # Links
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
        return text
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.strip().startswith('```'):
            if not in_code:
                flush_list()
                flush_table()
                in_code = True
                code_lang = line.strip()[3:].strip()
                code_lines = []
            else:
                in_code = False
                escaped = html.escape('\n'.join(code_lines))
                html_parts.append(f'<pre><code class="lang-{code_lang}">{escaped}</code></pre>')
            i += 1
            continue
        
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        
        stripped = line.strip()
        
        # Empty line
        if not stripped:
            flush_list()
            flush_table()
            i += 1
            continue
        
        # Table detection
        if '|' in stripped and stripped.startswith('|'):
            flush_list()
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(stripped)
            i += 1
            continue
        else:
            flush_table()
        
        # Headers
        if stripped.startswith('#'):
            flush_list()
            level = len(stripped) - len(stripped.lstrip('#'))
            text = stripped.lstrip('#').strip()
            html_parts.append(f'<h{level} class="md-h{level}">{process_inline(text)}</h{level}>')
            i += 1
            continue
        
        # Horizontal rule
        if stripped in ('---', '***', '___'):
            flush_list()
            html_parts.append('<hr/>')
            i += 1
            continue
        
        # Blockquote
        if stripped.startswith('>'):
            flush_list()
            text = stripped.lstrip('>').strip()
            html_parts.append(f'<blockquote>{process_inline(text)}</blockquote>')
            i += 1
            continue
        
        # List items
        if re.match(r'^[-*+]\s', stripped) or re.match(r'^\d+\.\s', stripped):
            if not in_list:
                in_list = True
                html_parts.append('<ul>')
            text = re.sub(r'^[-*+]\s+', '', stripped)
            text = re.sub(r'^\d+\.\s+', '', text)
            html_parts.append(f'<li>{process_inline(text)}</li>')
            i += 1
            continue
        
        # Checkbox
        if stripped.startswith('- [ ]') or stripped.startswith('- [x]'):
            if not in_list:
                in_list = True
                html_parts.append('<ul class="checklist">')
            checked = 'checked' if stripped.startswith('- [x]') else ''
            text = stripped[5:].strip()
            html_parts.append(f'<li><input type="checkbox" {checked} disabled> {process_inline(text)}</li>')
            i += 1
            continue
        
        # Regular paragraph
        flush_list()
        html_parts.append(f'<p>{process_inline(stripped)}</p>')
        i += 1
    
    flush_list()
    flush_table()
    
    return '\n'.join(html_parts)

# Build sections data
sections = []

# Companies
for company in companies:
    company_dir = os.path.join(BASE, company)
    if not os.path.isdir(company_dir):
        continue
    files = sorted(glob.glob(os.path.join(company_dir, "*.md")))
    for f in files:
        name = os.path.basename(f).replace('.md', '')
        content = read_file(f)
        section_id = f"{company}-{name}".replace(' ', '-')
        sections.append({
            'category': company,
            'type': 'company',
            'name': name,
            'title': f"{company} - {name}",
            'id': section_id,
            'content': content,
            'html': md_to_html(content),
        })

# General knowledge
for topic in general_topics:
    f = os.path.join(BASE, "通用知识", f"{topic}.md")
    if not os.path.exists(f):
        continue
    content = read_file(f)
    section_id = f"通用知识-{topic}".replace(' ', '-')
    # Check if diagrams map to this section
    diagrams_for_section = section_to_diagrams.get(topic, [])
    sections.append({
        'category': '通用知识',
        'type': 'general',
        'name': topic,
        'title': topic,
        'id': section_id,
        'content': content,
        'html': md_to_html(content),
        'diagrams': diagrams_for_section,
    })

# Build navigation structure
nav_structure = {}
for s in sections:
    cat = s['category']
    if cat not in nav_structure:
        nav_structure[cat] = []
    nav_structure[cat].append({'name': s['name'], 'id': s['id'], 'title': s['title']})

# Build the diagram gallery HTML
diagram_gallery_html = '<div class="diagram-gallery">'
for key, svg_content in svg_files.items():
    meta = diagram_meta.get(key, {'title': key})
    diagram_gallery_html += f'''
    <div class="diagram-card" onclick="openDiagramModal('{key}')">
        <div class="diagram-preview">{svg_content}</div>
        <div class="diagram-title">{meta['title']}</div>
    </div>'''
diagram_gallery_html += '</div>'

# Build diagram modals
diagram_modals_html = ''
for key, svg_content in svg_files.items():
    meta = diagram_meta.get(key, {'title': key})
    diagram_modals_html += f'''
    <div class="diagram-modal" id="modal-{key}" onclick="closeDiagramModal('{key}')">
        <div class="modal-content" onclick="event.stopPropagation()">
            <button class="modal-close" onclick="closeDiagramModal('{key}')">&times;</button>
            <h3>{meta['title']}</h3>
            <div class="modal-svg">{svg_content}</div>
        </div>
    </div>'''

# Build section content HTML
sections_html = ''
for s in sections:
    diagrams_html = ''
    if s.get('diagrams'):
        for dk in s['diagrams']:
            if dk in svg_files:
                meta = diagram_meta[dk]
                diagrams_html += f'''
                <div class="section-diagram" onclick="openDiagramModal('{dk}')">
                    <div class="section-diagram-inner">{svg_files[dk]}</div>
                    <div class="diagram-click-hint">点击查看大图</div>
                </div>'''
    
    sections_html += f'''
    <section class="content-section" id="section-{s['id']}" data-search="{html.escape(s['content'][:2000])}">
        <div class="section-header">
            <h2>{html.escape(s['title'])}</h2>
        </div>
        {diagrams_html}
        <div class="section-body">
            {s['html']}
        </div>
    </section>'''

# Build nav HTML
nav_html = '''
<div class="nav-section">
    <div class="nav-category" onclick="showSection('diagrams')">
        <span class="nav-icon">📊</span> 架构图
    </div>
</div>'''

# General knowledge first
if '通用知识' in nav_structure:
    nav_html += '<div class="nav-section"><div class="nav-category-title">📚 通用知识</div>'
    for item in nav_structure['通用知识']:
        nav_html += f'<div class="nav-item" onclick="showSection(\'{item["id"]}\')">{item["name"]}</div>'
    nav_html += '</div>'

# Companies
company_icons = {'字节跳动':'🔥','阿里巴巴':'🟠','腾讯':'💬','百度':'🔍','美团':'🟡','小红书':'📕','快手':'⚡','蚂蚁集团':'🐜','华为':'📱'}
for company in companies:
    if company in nav_structure:
        icon = company_icons.get(company, '🏢')
        nav_html += f'<div class="nav-section"><div class="nav-category-title">{icon} {company}</div>'
        for item in nav_structure[company]:
            nav_html += f'<div class="nav-item" onclick="showSection(\'{item["id"]}\')">{item["name"]}</div>'
        nav_html += '</div>'

# Build complete HTML
final_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Agent 面试知识库</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #0f172a;
  --bg2: #1e293b;
  --bg3: #334155;
  --accent: #f59e0b;
  --accent2: #fbbf24;
  --text: #f8fafc;
  --text2: #94a3b8;
  --text3: #64748b;
  --blue: #3b82f6;
  --green: #10b981;
  --red: #ef4444;
  --sidebar-w: 280px;
  --transition: 0.3s ease;
}}
[data-theme="light"] {{
  --bg: #f8fafc;
  --bg2: #ffffff;
  --bg3: #e2e8f0;
  --text: #0f172a;
  --text2: #475569;
  --text3: #94a3b8;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: 'Noto Sans SC', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.7;
  overflow-x: hidden;
}}
code, pre {{ font-family: 'JetBrains Mono', monospace; }}

/* Sidebar */
.sidebar {{
  position: fixed;
  left: 0; top: 0; bottom: 0;
  width: var(--sidebar-w);
  background: var(--bg2);
  border-right: 1px solid var(--bg3);
  overflow-y: auto;
  z-index: 100;
  transition: transform var(--transition);
  scrollbar-width: thin;
  scrollbar-color: var(--bg3) transparent;
}}
.sidebar-header {{
  padding: 20px;
  border-bottom: 1px solid var(--bg3);
  position: sticky;
  top: 0;
  background: var(--bg2);
  z-index: 1;
}}
.sidebar-header h1 {{
  font-size: 18px;
  color: var(--accent);
  margin-bottom: 4px;
}}
.sidebar-header p {{
  font-size: 12px;
  color: var(--text3);
}}
.nav-section {{
  padding: 4px 0;
}}
.nav-category-title {{
  padding: 10px 20px 6px;
  font-size: 13px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.5px;
}}
.nav-category {{
  padding: 10px 20px;
  font-size: 14px;
  font-weight: 600;
  color: var(--accent);
  cursor: pointer;
  transition: background var(--transition);
}}
.nav-category:hover {{ background: var(--bg3); }}
.nav-item {{
  padding: 7px 20px 7px 36px;
  font-size: 13px;
  color: var(--text2);
  cursor: pointer;
  transition: all var(--transition);
  border-left: 3px solid transparent;
}}
.nav-item:hover {{
  background: rgba(245,158,11,0.1);
  color: var(--accent);
  border-left-color: var(--accent);
}}
.nav-item.active {{
  background: rgba(245,158,11,0.15);
  color: var(--accent);
  border-left-color: var(--accent);
  font-weight: 500;
}}

/* Main content */
.main {{
  margin-left: var(--sidebar-w);
  min-height: 100vh;
}}
.topbar {{
  position: sticky;
  top: 0;
  background: var(--bg);
  border-bottom: 1px solid var(--bg3);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  z-index: 50;
  backdrop-filter: blur(10px);
}}
.hamburger {{
  display: none;
  background: none;
  border: none;
  color: var(--text);
  font-size: 24px;
  cursor: pointer;
}}
.search-box {{
  flex: 1;
  max-width: 500px;
  position: relative;
}}
.search-box input {{
  width: 100%;
  padding: 8px 16px 8px 40px;
  background: var(--bg2);
  border: 1px solid var(--bg3);
  border-radius: 8px;
  color: var(--text);
  font-size: 14px;
  outline: none;
  transition: border var(--transition);
}}
.search-box input:focus {{ border-color: var(--accent); }}
.search-box::before {{
  content: '🔍';
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 14px;
}}
.topbar-actions {{
  display: flex;
  gap: 8px;
  align-items: center;
}}
.topbar-actions button {{
  padding: 6px 14px;
  background: var(--bg3);
  border: none;
  border-radius: 6px;
  color: var(--text2);
  font-size: 12px;
  cursor: pointer;
  transition: all var(--transition);
  font-family: inherit;
}}
.topbar-actions button:hover {{
  background: var(--accent);
  color: var(--bg);
}}
.theme-toggle {{
  background: none !important;
  font-size: 18px !important;
  padding: 4px 8px !important;
}}

/* Content area */
.content {{
  padding: 24px;
  max-width: 1000px;
  margin: 0 auto;
}}
.content-section {{
  display: none;
  animation: fadeIn 0.3s ease;
}}
.content-section.active {{ display: block; }}
@keyframes fadeIn {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}

.section-header h2 {{
  font-size: 24px;
  color: var(--accent);
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--bg3);
}}

/* Markdown content styles */
.section-body h1 {{ font-size: 22px; color: var(--accent); margin: 24px 0 12px; }}
.section-body h2 {{ font-size: 20px; color: var(--accent2); margin: 24px 0 12px; border-bottom: 1px solid var(--bg3); padding-bottom: 8px; }}
.section-body .md-h3 {{ font-size: 17px; color: var(--text); margin: 20px 0 10px; }}
.section-body .md-h4 {{ font-size: 15px; color: var(--text2); margin: 16px 0 8px; }}
.section-body p {{ margin: 8px 0; color: var(--text2); }}
.section-body ul {{ padding-left: 24px; margin: 8px 0; }}
.section-body li {{ margin: 4px 0; color: var(--text2); }}
.section-body li strong {{ color: var(--text); }}
.section-body blockquote {{
  border-left: 3px solid var(--accent);
  padding: 8px 16px;
  margin: 12px 0;
  background: rgba(245,158,11,0.05);
  color: var(--text2);
  border-radius: 0 6px 6px 0;
}}
.section-body pre {{
  background: var(--bg);
  border: 1px solid var(--bg3);
  border-radius: 8px;
  padding: 16px;
  overflow-x: auto;
  margin: 12px 0;
  font-size: 13px;
  line-height: 1.5;
}}
.section-body code {{
  background: var(--bg3);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
  color: var(--accent2);
}}
.section-body pre code {{
  background: none;
  padding: 0;
  color: var(--text);
}}
.section-body hr {{
  border: none;
  border-top: 1px solid var(--bg3);
  margin: 20px 0;
}}
.section-body a {{
  color: var(--blue);
  text-decoration: none;
}}
.section-body a:hover {{ text-decoration: underline; }}
.section-body strong {{ color: var(--text); }}

/* Tables */
.table-wrap {{
  overflow-x: auto;
  margin: 12px 0;
}}
.section-body table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
.section-body th {{
  background: var(--bg3);
  padding: 8px 12px;
  text-align: left;
  font-weight: 600;
  color: var(--accent2);
  border: 1px solid rgba(255,255,255,0.1);
}}
.section-body td {{
  padding: 8px 12px;
  border: 1px solid var(--bg3);
  color: var(--text2);
}}
.section-body tr:hover td {{ background: rgba(245,158,11,0.05); }}

/* Diagram sections */
.section-diagram {{
  margin: 16px 0;
  padding: 16px;
  background: var(--bg2);
  border: 1px solid var(--bg3);
  border-radius: 12px;
  cursor: pointer;
  transition: all var(--transition);
  position: relative;
  overflow: hidden;
}}
.section-diagram:hover {{
  border-color: var(--accent);
  box-shadow: 0 0 20px rgba(245,158,11,0.1);
}}
.section-diagram-inner {{ max-width: 100%; }}
.section-diagram-inner svg {{ width: 100%; height: auto; }}
.diagram-click-hint {{
  position: absolute;
  bottom: 8px;
  right: 12px;
  font-size: 11px;
  color: var(--text3);
  opacity: 0;
  transition: opacity var(--transition);
}}
.section-diagram:hover .diagram-click-hint {{ opacity: 1; }}

/* Diagram gallery */
.diagram-gallery {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 20px;
}}
.diagram-card {{
  background: var(--bg2);
  border: 1px solid var(--bg3);
  border-radius: 12px;
  padding: 16px;
  cursor: pointer;
  transition: all var(--transition);
  overflow: hidden;
}}
.diagram-card:hover {{
  border-color: var(--accent);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}}
.diagram-preview svg {{ width: 100%; height: auto; }}
.diagram-title {{
  margin-top: 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--accent);
  text-align: center;
}}

/* Modal */
.diagram-modal {{
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.85);
  z-index: 1000;
  justify-content: center;
  align-items: center;
  padding: 20px;
}}
.diagram-modal.open {{ display: flex; }}
.modal-content {{
  background: var(--bg2);
  border-radius: 16px;
  padding: 24px;
  max-width: 95vw;
  max-height: 95vh;
  overflow: auto;
  position: relative;
}}
.modal-content h3 {{
  color: var(--accent);
  margin-bottom: 16px;
  font-size: 20px;
}}
.modal-close {{
  position: absolute;
  top: 12px; right: 16px;
  background: none;
  border: none;
  color: var(--text2);
  font-size: 28px;
  cursor: pointer;
}}
.modal-close:hover {{ color: var(--accent); }}
.modal-svg svg {{ width: 100%; height: auto; max-height: 80vh; }}

/* Checklist */
.checklist {{ list-style: none; padding-left: 8px; }}
.checklist li {{ display: flex; align-items: center; gap: 8px; }}
.checklist input {{ accent-color: var(--accent); }}

/* Search results */
.search-highlight {{ background: rgba(245,158,11,0.3); padding: 1px 3px; border-radius: 2px; }}

/* Mobile responsive */
@media (max-width: 768px) {{
  .sidebar {{ transform: translateX(-100%); }}
  .sidebar.open {{ transform: translateX(0); }}
  .main {{ margin-left: 0; }}
  .hamburger {{ display: block; }}
  .diagram-gallery {{ grid-template-columns: 1fr; }}
  .topbar-actions button:not(.theme-toggle) {{ display: none; }}
  .overlay {{ display: none; position: fixed; top:0;left:0;right:0;bottom:0; background:rgba(0,0,0,0.5); z-index:99; }}
  .overlay.open {{ display: block; }}
}}
</style>
</head>
<body>
<div class="overlay" id="overlay" onclick="toggleSidebar()"></div>
<aside class="sidebar" id="sidebar">
  <div class="sidebar-header">
    <h1>🤖 AI Agent 面试库</h1>
    <p>大厂面经 · 知识体系 · 架构图</p>
  </div>
  <nav id="nav">
    {nav_html}
  </nav>
</aside>

<main class="main">
  <div class="topbar">
    <button class="hamburger" onclick="toggleSidebar()">☰</button>
    <div class="search-box">
      <input type="text" id="searchInput" placeholder="搜索知识点..." oninput="handleSearch(this.value)">
    </div>
    <div class="topbar-actions">
      <button onclick="expandAll()">全部展开</button>
      <button onclick="collapseAll()">全部折叠</button>
      <button class="theme-toggle" onclick="toggleTheme()">🌙</button>
    </div>
  </div>
  
  <div class="content" id="content">
    <!-- Welcome section -->
    <section class="content-section active" id="section-welcome">
      <div class="section-header">
        <h2>👋 欢迎来到 AI Agent 面试知识库</h2>
      </div>
      <div class="section-body">
        <p>覆盖 <strong>9 家大厂</strong>（字节、阿里、腾讯、百度、美团、小红书、快手、蚂蚁、华为）的 AI Agent 岗位面试题与面经。</p>
        <p>包含 <strong>7 张架构图</strong>、通用知识体系、高频拷打题、真实面经实录。</p>
        <p>👈 从左侧导航选择内容开始学习，或使用顶部搜索快速定位。</p>
        <hr/>
        <h3 class="md-h3">📊 架构图预览</h3>
        {diagram_gallery_html}
      </div>
    </section>
    
    <!-- Diagrams section -->
    <section class="content-section" id="section-diagrams">
      <div class="section-header">
        <h2>📊 架构图集</h2>
      </div>
      <div class="section-body">
        <p>点击任意架构图查看大图</p>
        {diagram_gallery_html}
      </div>
    </section>
    
    <!-- Search results -->
    <section class="content-section" id="section-search-results">
      <div class="section-header">
        <h2>🔍 搜索结果</h2>
      </div>
      <div class="section-body" id="searchResults"></div>
    </section>
    
    {sections_html}
  </div>
</main>

{diagram_modals_html}

<script>
// State
let currentSection = 'welcome';
let allSections = document.querySelectorAll('.content-section');

function showSection(id) {{
  allSections.forEach(s => s.classList.remove('active'));
  let target = document.getElementById('section-' + id);
  if (target) target.classList.add('active');
  else {{
    // Show welcome if not found
    document.getElementById('section-welcome').classList.add('active');
  }}
  // Update nav active state
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => {{
    if (n.getAttribute('onclick')?.includes(id)) n.classList.add('active');
  }});
  // Close sidebar on mobile
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('open');
  window.scrollTo(0, 0);
  currentSection = id;
}}

function toggleSidebar() {{
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('open');
}}

function toggleTheme() {{
  let body = document.body;
  let btn = document.querySelector('.theme-toggle');
  if (body.getAttribute('data-theme') === 'light') {{
    body.removeAttribute('data-theme');
    btn.textContent = '🌙';
  }} else {{
    body.setAttribute('data-theme', 'light');
    btn.textContent = '☀️';
  }}
}}

function expandAll() {{
  // Show all sections with Q&A style - expand details if any
  document.querySelectorAll('details').forEach(d => d.open = true);
}}

function collapseAll() {{
  document.querySelectorAll('details').forEach(d => d.open = false);
}}

function openDiagramModal(key) {{
  document.getElementById('modal-' + key)?.classList.add('open');
}}

function closeDiagramModal(key) {{
  document.getElementById('modal-' + key)?.classList.remove('open');
}}

// Search
function handleSearch(query) {{
  if (!query || query.length < 2) {{
    if (currentSection === 'search-results') showSection('welcome');
    return;
  }}
  
  let results = [];
  let q = query.toLowerCase();
  
  document.querySelectorAll('.content-section[data-search]').forEach(section => {{
    let searchText = section.getAttribute('data-search').toLowerCase();
    let title = section.querySelector('.section-header h2')?.textContent || '';
    if (searchText.includes(q) || title.toLowerCase().includes(q)) {{
      let id = section.id.replace('section-', '');
      results.push({{ title: title, id: id }});
    }}
  }});
  
  let resultsHtml = results.length === 0 
    ? '<p>未找到相关内容</p>'
    : results.map(r => `<div class="nav-item" style="padding:12px 16px;font-size:15px;margin:4px 0;background:var(--bg2);border-radius:8px;" onclick="showSection('${{r.id}}')">${{r.title}}</div>`).join('');
  
  document.getElementById('searchResults').innerHTML = `<p>找到 ${{results.length}} 个结果</p>` + resultsHtml;
  showSection('search-results');
}}

// Keyboard shortcut
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    document.querySelectorAll('.diagram-modal.open').forEach(m => m.classList.remove('open'));
  }}
  if (e.key === '/' && !e.ctrlKey && !e.metaKey) {{
    let input = document.getElementById('searchInput');
    if (document.activeElement !== input) {{
      e.preventDefault();
      input.focus();
    }}
  }}
}});
</script>
</body>
</html>'''

# Write output
output_path = os.path.join(BASE, "index.html")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(final_html)

print(f"Written to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
print(f"Sections: {len(sections)}")
print(f"SVG diagrams: {len(svg_files)}")
