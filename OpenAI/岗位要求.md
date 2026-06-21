# OpenAI AI Agent 相关岗位要求

> 最后更新：2026年6月（基于 OpenAI 官网 JD 及公开招聘信息整理）

---

## 一、核心岗位概览

OpenAI 目前 Agent 相关岗位主要分为三个方向：

| 岗位方向 | 典型职级 | 薪资范围（美国） |
|---|---|---|
| AI Engineer（应用/系统） | L4–L6 | $200k–$450k + 股权 |
| Research Engineer（Agent Systems） | L4–L6 | $220k–$500k + 股权 |
| ML Engineer（Deployment/Infra） | L4–L5 | $180k–$380k + 股权 |

---

## 二、AI Engineer（Agent / Assistants 方向）

### 核心职责

- 构建和维护 ChatGPT、API 产品中 Agent 相关功能（Tools、Memory、Multi-step Reasoning）
- 设计和实现 Agent 工具调用管道（Function Calling、Code Interpreter、Browser Use）
- 研发 Agents SDK 相关能力，优化 Agent orchestration 和 handoff 机制
- 与 Research 团队协作，将最新模型能力产品化（o3/o4-mini/GPT-5 等）
- 评估 Agent 系统可靠性、安全性，设计 Guardrails 和 Fallback 机制

### 技术要求

**必须具备：**
- 5年以上软件工程经验
- Python 精通（是主力语言），熟悉 TypeScript/Node.js
- 深入理解 LLM 工作原理：Transformer 架构、Attention 机制、Tokenization、Context Window 管理
- RAG 系统全链路经验：Embedding 选型、向量数据库（Pinecone/Weaviate/pgvector）、检索策略、重排序
- Agent 框架实际项目经验：LangChain/LangGraph、OpenAI Assistants API、Responses API 或自研框架
- 分布式系统设计能力：高并发、低延迟系统架构
- 扎实的数据结构与算法基础（LeetCode Medium/Hard 级别）

**加分项：**
- OpenAI API 深度使用经验（包括 Realtime API、Batch API）
- Agentic Coding 工具构建经验（如开发 Claude Code / Cursor 类工具）
- 推理模型（o-series）应用优化经验
- 开源社区贡献（Agent 框架、LLM 工具链）
- AI Safety 相关研究背景或实践经验

---

## 三、Research Engineer（Agent Systems / Coding Environment）

### 核心职责

- 构建 Agent 执行环境（Code Execution Sandbox、Tool Integration、Sandboxing）
- 开发可扩展的多步 Agent 工作流编排框架（Planning、Tool Use、Memory、Coordination）
- 设计 Agent 性能评估和 Benchmark 系统
- 通过 Prompting、数据精选、Post-training 提升 Agent 性能
- 与 Policy/Safety 团队协作，确保 Agent 系统符合安全标准

### 技术要求

**必须具备：**
- PhD 或同等研究经验（或 5年以上相关工程经验）
- LLM 系统/Pipeline/Tool-Integrated Workflow 构建经验
- 分布式系统和编排工具经验（Docker、Kubernetes、Ray、PyTorch Distributed）
- 系统设计能力：可扩展推理基础设施架构
- 强算法基础

**加分项：**
- 强化学习工作流经验（Rollout、Trajectory Collection、Policy Optimization）
- 评估框架、Prompting/Finetuning/RL 大规模实验经验
- 安全沙箱和代码执行环境构建经验

---

## 四、面试流程（2025-2026 最新）

```
1. 招聘方初筛（15-30min 电话）
   - 背景了解、岗位匹配度、薪资预期

2. 有偿 Work Trial（5-8小时，付费约$200-500）
   - 模拟真实工作任务，评估生产级代码质量
   - 典型题目：设计并实现一个 mini Agent 工具调用系统
   - 评估维度：代码质量、架构设计、错误处理、测试覆盖

3. 技术电话面（60min）
   - LeetCode 算法题（Medium/Hard）
   - ML 基础概念考察

4. Onsite Loop（4-5轮，可远程）
   - 轮次1：Coding（算法 + 数据结构）
   - 轮次2：系统设计（重点：Inference Serving、RAG Pipeline、Agent Infrastructure）
   - 轮次3：ML/AI 深度（LLM 原理、Agent 设计、Safety）
   - 轮次4：Mission Alignment（价值观、对 AI Safety 的理解）
   - 轮次5：Cross-functional（与 PM/Research 合作经验）
```

---

## 五、OpenAI 面试重点考察方向

### 1. AI Safety & Mission Alignment（必考，不可忽视）

OpenAI 的面试中，Safety 不是加分项，是**门槛**。面试官会直接问：

> "你如何看待 AI 对齐问题？"
> "如果你发现模型可能被滥用，你会怎么做？"
> "OpenAI 的使命是'确保 AGI 造福全人类'，你认为这个使命的挑战在哪里？"

准备方向：
- 了解 OpenAI 的 Safety 研究（Constitutional AI、RLHF、Superalignment）
- 有自己的观点，能清晰阐述，不需要与公司立场完全一致，但要展示深度思考

### 2. Agent Infrastructure 系统设计

典型题目：
> 设计一个支持 100 万并发用户的 AI 助手系统，支持工具调用和多轮对话

考察要点：
- LLM Inference 层：批处理、KV Cache、Token Streaming
- 工具调用层：异步执行、超时处理、错误恢复
- 状态管理层：会话记忆持久化、分布式 Context 同步
- 安全层：Rate Limiting、Prompt Injection 防御、输出过滤

### 3. 最新 API 深度使用

2025-2026 重点：
- **Responses API** vs **Chat Completions API** 的设计差异
- **Agents SDK**：Agent 编排、Handoff、Tool 定义
- **Realtime API**：语音 Agent 的延迟优化
- **Reasoning Models（o-series）**：何时用 o3 vs gpt-4o，成本/性能权衡

---

## 六、实战备考建议

1. **动手构建**：用 OpenAI Agents SDK 做一个有实际功能的 Agent（工具调用+记忆+流式输出）
2. **读源码**：研究 OpenAI Agents SDK 源码，理解 handoff 和 tracing 机制
3. **Safety 必读**：阅读 [OpenAI System Card](https://openai.com/index/openai-o3-system-card)，了解最新安全机制
4. **Work Trial 准备**：写干净的、production-ready 代码，加完善的错误处理和测试
5. **算法不能偷懒**：即使是 Applied AI 岗位，依然要刷 LeetCode（Medium 为主）
