# 12 周 AI Agent 工程师进阶路线

> 面向已经会基础编程、想在 3 个月内补齐 AI Agent / 大模型应用工程能力的人。目标不是“看完资料”，而是每周交付一个可验证产物，最后形成可写进简历、可用于面试答辩的作品集。

---

## 路线总览

| 阶段 | 周期 | 目标 | 交付物 |
|---|---:|---|---|
| 基础夯实 | 第 1-2 周 | 补齐 LLM、Prompt、Tool Calling、RAG 基础 | 个人知识卡片 + 10 个核心概念答案 |
| RAG 工程化 | 第 3-4 周 | 做出可评估、可溯源的 RAG 系统 | 文档问答 Demo + 评估报告 |
| Agent 编排 | 第 5-6 周 | 掌握 ReAct、Plan-and-Execute、LangGraph 状态图 | ReAct Agent + LangGraph 多步任务 |
| 工具与协议 | 第 7-8 周 | 掌握 Function Calling、MCP、权限与沙箱 | 自建 MCP Server + 安全工具调用 |
| 生产化能力 | 第 9-10 周 | 补齐可观测性、评估、成本、部署、安全 | Agent 服务化 + trace + guardrails |
| 面试作品集 | 第 11-12 周 | 完成项目打磨、答辩材料和模拟面试 | 项目 README + 架构图 + 10 分钟答辩稿 |

---

## 第 1 周：LLM 与 Prompt 基础

**学习重点**

- Transformer、Token、Context Window、KV Cache 的基本概念
- System Prompt、Few-shot、结构化输出、JSON schema
- 幻觉、temperature、top-p、模型路由的基本取舍

**必做任务**

1. 用一页纸解释：LLM 为什么会幻觉？RAG 为什么能缓解？
2. 写 5 个结构化输出 prompt：JSON、表格、评分、拒答、引用格式。
3. 准备 10 个常见追问答案：Token、KV Cache、温度、上下文压缩、Prompt 注入。

**验收标准**

- 能不用背稿讲清楚“模型生成不是查库，而是条件概率采样”。
- 能解释结构化输出失败时如何兜底：重试、schema 校验、parser repair、函数调用。

---

## 第 2 周：Tool Calling 与基础 Agent Loop

**学习重点**

- Function Calling / Tool Use 的参数 schema、工具描述、错误处理
- ReAct：Thought → Action → Observation 循环
- 工具权限、危险操作确认、工具结果压缩

**必做任务**

1. 写一个最小 ReAct Agent：支持 `search`、`calculator`、`read_file` 三个工具。
2. 给工具加权限分级：只读、需确认、高危禁止。
3. 记录 5 个失败 case，并写出原因归因。

**验收标准**

- 能说明 Tool Description 为什么影响工具选择准确率。
- 能讲清楚工具失败后的重试、降级和停止条件。

---

## 第 3 周：RAG 最小可用系统

**学习重点**

- 文档解析、chunking、embedding、向量库、检索、生成
- 引用溯源、上下文拼接、TopK 选择

**必做任务**

1. 用 20 篇 Markdown/PDF 做一个本地知识库问答。
2. 输出答案时带引用片段和来源文件。
3. 做 30 条问题的人工测试集。

**验收标准**

- 每个答案能追溯到原文片段。
- 能解释 chunk size / overlap 对召回和幻觉的影响。

---

## 第 4 周：RAG 优化与评估

**学习重点**

- Hybrid Search：BM25 + Dense Retrieval
- RRF 融合、Rerank、Query Rewrite、HyDE
- RAGAS、Recall@K、MRR、NDCG、Faithfulness

**必做任务**

1. 加 BM25 + Dense 双路召回。
2. 加 reranker，并对比 Top5 命中率。
3. 输出一份评估报告：召回、延迟、成本、失败样例。

**验收标准**

- 能说明“召回不到”和“生成不对”如何分层排查。
- 能用指标证明优化是否有效，而不是只说“效果更好”。

---

## 第 5 周：LangGraph 状态图

**学习重点**

- State、Node、Edge、Conditional Edge、Checkpoint
- Graph vs Chain vs Agent Loop 的边界

**必做任务**

1. 用 LangGraph 改写第 2 周 ReAct Agent。
2. 加 `max_iterations`、失败重试、人工确认节点。
3. 让任务中断后可以从 checkpoint 恢复。

**验收标准**

- 能画出 StateGraph 并解释每个节点的输入输出。
- 能说明 checkpoint 恢复如何保证幂等。

---

## 第 6 周：多 Agent 协作

**学习重点**

- Supervisor / Worker、角色分工、共享状态、A2A 通信
- 多 Agent 的协调成本、循环风险、一致性问题

**必做任务**

1. 做一个三角色协作系统：Researcher、Coder、Writer。
2. Supervisor 根据状态决定下一步派谁。
3. 加成本上限、轮次上限和任务完成判断。

**验收标准**

- 能解释什么时候用多 Agent，什么时候单 Agent + 多工具更好。
- 能定位多 Agent 循环、重复工作、上下文污染问题。

---

## 第 7 周：MCP 与工具生态

**学习重点**

- MCP Host / Client / Server / Transport
- MCP 与 Function Calling、Plugin、A2A 的关系

**必做任务**

1. 自建一个 MCP Server：暴露 `search_docs`、`get_doc`、`summarize` 三个工具。
2. 写工具 schema、错误返回格式和权限说明。
3. 用 Claude Desktop / 支持 MCP 的客户端连通。

**验收标准**

- 能解释 MCP 解决的是“工具接入标准化”问题。
- 能说明 MCP Server 的安全边界和认证方式。

---

## 第 8 周：Agent 安全与权限

**学习重点**

- Prompt 注入、工具误调用、数据泄露、越权访问
- Sandbox、policy engine、audit log、human approval

**必做任务**

1. 给工具调用加 policy check。
2. 为危险操作加“先生成计划、用户确认、再执行”的流程。
3. 设计 10 个红队 prompt 并记录拦截结果。

**验收标准**

- 能讲清楚“输入过滤、工具前置检查、输出校验”三道防线。
- 能说明如何审计 Agent 的每次 tool call。

---

## 第 9 周：可观测性与评估

**学习重点**

- Trace、Run replay、日志、指标、告警
- Task success rate、tool accuracy、latency、cost、hallucination rate

**必做任务**

1. 给 Agent 加 trace id 和每步状态快照。
2. 输出一次任务执行的 timeline。
3. 构建 50 条回归测试集，统计任务成功率。

**验收标准**

- 能回答“线上 bad case 怎么定位到是哪一层的问题”。
- 能展示 trace、日志和评估数据。

---

## 第 10 周：部署与成本优化

**学习重点**

- FastAPI 服务化、Docker、队列、缓存、限流、降级
- 模型路由、语义缓存、上下文压缩、batching

**必做任务**

1. 把 Agent 包装成 HTTP API。
2. 加 Redis 语义缓存和限流。
3. 输出延迟、token 成本、缓存命中率三项指标。

**验收标准**

- 能说明如何把 demo 变成可运行服务。
- 能解释哪些请求走大模型，哪些走小模型或缓存。

---

## 第 11 周：作品集打磨

**学习重点**

- README、架构图、部署说明、评估报告、失败案例复盘

**必做任务**

1. 为项目写一份面试官友好的 README。
2. 准备 3 张图：系统架构图、数据流图、评估闭环图。
3. 写“项目亮点 / 难点 / trade-off / 下一步优化”四段。

**验收标准**

- 面试官不看代码，也能通过 README 明白你的贡献。
- 项目亮点有指标，不只是“用了 LangChain / RAG / Agent”。

---

## 第 12 周：模拟面试与补短板

**学习重点**

- 项目答辩、系统设计、八股追问、代码题

**必做任务**

1. 准备 10 分钟项目答辩稿。
2. 做 3 次模拟面试：项目深挖、系统设计、RAG/Agent 专项。
3. 根据失败题补一页“错题复盘”。

**验收标准**

- 能讲清楚你为什么这么设计，而不是只讲“我用了什么框架”。
- 能回答：效果如何评估？失败怎么办？成本多少？如何上线？

---

## 推荐每周复盘模板

```text
本周主题：
完成产物：
最难的技术点：
遇到的 bad case：
指标结果：
下周改进：
可写进简历的一句话：
```

## 最终作品集清单

- 一个 RAG 知识问答系统
- 一个 LangGraph 多 Agent 协作系统
- 一个带安全护栏和可观测性的生产级 Agent 服务
- 一份项目 README
- 一份评估报告
- 一份 10 分钟项目答辩稿

