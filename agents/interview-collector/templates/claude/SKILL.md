---
name: interview-collector
description: Collect and curate public AI Agent, LLM, RAG, MCP, and AI application interview experiences for agent-interview-hub.
---

# Interview Collector

Use this skill when the user asks to collect, summarize, deduplicate, or update AI Agent / LLM / RAG interview experiences.

## Rules

- Use only public accessible sources unless the user logs in through their browser.
- Do not request, expose, store, or commit cookies, tokens, or credentials.
- Do not bypass paywalls, login walls, or anti-bot controls.
- Summarize sources; do not copy long external text.
- Preserve source attribution.

## Workflow

1. Search: 牛客、小红书、知乎、CSDN、博客园、掘金、GitHub、RSS.
2. Read public pages with Jina Reader or platform tools.
3. Extract structured fields.
4. Deduplicate.
5. Score quality.
6. Write candidates to `data/interviews.json`.
7. Generate Markdown summaries and run the static site build.

## Commands

```bash
python3 scripts/collect_interviews.py doctor
python3 scripts/collect_interviews.py search --query "AI Agent 大模型 面经 2026"
agent-reach doctor --json
mcporter call 'exa.web_search_exa(query: "AI Agent 大模型 面经 2026", numResults: 10)'
curl -s "https://r.jina.ai/https://example.com/page"
opencli xiaohongshu search "AI Agent 面经" -f yaml --window background
gh search repos "AI Agent interview" --limit 10 --json fullName,description,stargazersCount,url,updatedAt
python3 -m json.tool data/interviews.json
python3 scripts/build_site.py
```

## Data model

```json
{
  "id": "stable-kebab-id",
  "platform": "牛客",
  "title": "面经标题",
  "company": "字节跳动",
  "role": "AI Agent开发",
  "published_at": "2026-05-20",
  "source_url": "https://...",
  "source_note_id": null,
  "source_lookup": null,
  "score": 5,
  "topics": ["Agent", "RAG", "MCP"],
  "summary": "一到两句摘要。"
}
```

For Xiaohongshu, avoid unstable direct links in public Markdown. Use `source_note_id` and `source_lookup`.
