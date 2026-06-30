# Interview Collector Agent

Interview Collector Agent 是本仓库用于“搜索、整理、去重、沉淀 AI Agent / 大模型面经”的跨平台 Agent 规范。

它不是爬虫，也不绕过登录、付费墙或反爬限制。它的职责是把公开来源里的面经线索整理成结构化候选，再由维护者决定是否写入 `data/interviews.json`、公司面经文档或专题题库。

## 能做什么

1. 搜索公开面经来源：牛客、小红书、知乎、CSDN、博客园、掘金、GitHub、RSS 等。
2. 读取公开网页正文，提取标题、公司、岗位、时间、来源链接、考点和摘要。
3. 按 URL、标题、公司、发布时间、内容摘要做去重。
4. 输出结构化 JSON 候选，便于后续生成 Markdown 和静态站页面。
5. 避免大段搬运外部内容，只保留摘要、问题要点和来源。

## 一键安装

```bash
bash agents/interview-collector/install.sh --targets copilot,claude,cursor,generic
```

安装目标：

| 目标 | 写入位置 | 用途 |
|---|---|---|
| Copilot CLI | `~/.copilot/instructions/interview-collector.instructions.md` | 让 Copilot CLI 自动按本规范采集面经 |
| Claude Code | `~/.claude/skills/interview-collector/SKILL.md` | 安装为 Claude skill |
| Cursor | `.cursor/rules/interview-collector.mdc` | 当前仓库 Cursor 项目规则 |
| Generic | `~/.agent-interview-hub/interview-collector/AGENT.md` | 通用提示词，适合 Trae、通义灵码、豆包 MarsCode、文心快码等手动导入 |

可单独安装：

```bash
bash agents/interview-collector/install.sh --targets copilot
bash agents/interview-collector/install.sh --targets claude,cursor
```

## 推荐工具链

| 能力 | 推荐工具 |
|---|---|
| 全网搜索 | Agent-Reach + Exa (`mcporter call 'exa.web_search_exa(...)'`) |
| 网页读取 | Jina Reader (`curl -s "https://r.jina.ai/URL"`) |
| 小红书搜索 | OpenCLI (`opencli xiaohongshu search ...`) |
| GitHub 资源 | `gh search repos`, `gh search code` |
| RSS | Python `feedparser` |

## 完整采集流程

本目录既提供跨平台 Agent prompt，也提供可执行的采集辅助脚本：[`../../scripts/collect_interviews.py`](../../scripts/collect_interviews.py)。

### 1. 检查工具

```bash
python3 scripts/collect_interviews.py doctor
```

### 2. 搜索公开来源

```bash
python3 scripts/collect_interviews.py search \
  --platform nowcoder \
  --platform zhihu \
  --platform blogs \
  --platform github \
  --query "AI Agent 大模型 面经 2026" \
  --output data/interview_candidates.json \
  --report data/interview_candidates.md
```

### 3. 搜索小红书（可选）

小红书依赖本机 OpenCLI 和浏览器登录态。脚本不会把不稳定的小红书直链写进公开 Markdown，而是保留 note id 和“站内搜索原标题”。

```bash
python3 scripts/collect_interviews.py search \
  --platform xiaohongshu \
  --query "AI Agent 面经" \
  --append
```

### 4. 审核候选

脚本输出两个文件：

| 文件 | 作用 |
|---|---|
| `data/interview_candidates.json` | 原始候选，供 Agent / 维护者审核 |
| `data/interview_candidates.md` | 可读报告，方便快速筛选 |

审核后再把高质量条目整理进 `data/interviews.json`，并更新 `通用知识/最新AI-Agent面经索引.md` 或公司文档。

### 5. 验证站点

```bash
python3 -m json.tool data/interviews.json
python3 scripts/build_site.py
```

## 输出原则

- `data/interviews.json` 是结构化数据源。
- Markdown 是展示层，不是唯一事实来源。
- 小红书等登录态平台不放不稳定直链，保留“站内搜索原标题”和 note id。
- GitHub / CSDN / 知乎里的题库和二手总结标为参考资料，不混作一手真实面经。

## 工作流

```text
搜索候选
  ↓
公开网页读取 / 平台搜索结果读取
  ↓
字段抽取与摘要
  ↓
去重与评分
  ↓
写入 data/interviews.json
  ↓
生成最新面经索引 / 公司面经摘要
  ↓
构建静态站验证
```

对应脚本命令：

```bash
python3 scripts/collect_interviews.py doctor
python3 scripts/collect_interviews.py search --query "AI Agent 大模型 面经 2026"
python3 scripts/collect_interviews.py render
python3 scripts/build_site.py
```

## 质量标准

| 分数 | 含义 |
|---:|---|
| 5 | 强相关、近期、公司/岗位/问题清晰，可优先入库 |
| 4 | 相关度高但来源或字段不完整，适合补充 |
| 3 | 题库/资料型或推广味较重，只作为参考 |
| 1-2 | 不建议入库 |

## 相关文件

- Canonical prompt: [`AGENT.md`](AGENT.md)
- Copilot template: [`templates/copilot/interview-collector.instructions.md`](templates/copilot/interview-collector.instructions.md)
- Claude template: [`templates/claude/SKILL.md`](templates/claude/SKILL.md)
- Cursor template: [`templates/cursor/interview-collector.mdc`](templates/cursor/interview-collector.mdc)
- Generic template: [`templates/generic/AGENT.md`](templates/generic/AGENT.md)
