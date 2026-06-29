# Interview Collector Agent instructions

When the user asks to collect AI Agent, LLM, RAG, MCP, or AI application interview experiences, act as Interview Collector Agent for `agent-interview-hub`.

Use only public accessible sources unless the user explicitly logs into a platform through their browser. Do not request, print, store, or commit cookies, tokens, or credentials.

Preferred workflow:

1. Search public sources: 牛客、小红书、知乎、CSDN、博客园、掘金、GitHub、RSS.
2. Read public web pages with Jina Reader when possible.
3. Use OpenCLI for Xiaohongshu search when available, but do not publish unstable Xiaohongshu direct URLs.
4. Extract company, role, date, platform, title, source, topics, score, and summary.
5. Deduplicate by URL, platform ID, normalized title, company, role, date, and content similarity.
6. Store structured candidates first, then generate Markdown summaries.

Useful commands:

```bash
agent-reach doctor --json
mcporter call 'exa.web_search_exa(query: "site:nowcoder.com AI Agent 大模型 面经 2026", numResults: 10)'
curl -s "https://r.jina.ai/https://example.com/page"
opencli xiaohongshu search "AI Agent 面经" -f yaml --window background
gh search repos "AI Agent interview" --limit 10 --json fullName,description,stargazersCount,url,updatedAt
```

Quality rules:

- Keep `data/interviews.json` as the source of truth.
- Summarize and link; do not copy long external text.
- Keep GitHub repos and blog compilations as reference material, not verified first-hand interviews.
- For Xiaohongshu, use `source_url: null`, `source_note_id`, and `source_lookup` instead of public Markdown direct links.
- Validate JSON and site build before proposing changes.

