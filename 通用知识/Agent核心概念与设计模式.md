# Agent 核心概念与设计模式面试题

## 一、Agent 核心概念

### 什么是 AI Agent？
AI Agent 是能够**感知环境、推理决策、制定计划、执行行动**的自主 AI 系统。与传统 LLM 应用的区别：
- **传统 LLM**：输入→输出，单次调用
- **AI Agent**：感知→推理→规划→执行→观察→循环，多步自主决策

### Agent 四大核心能力
1. **感知（Perception）**：接收用户输入、环境反馈、工具返回
2. **推理（Reasoning）**：分析当前状况，做出判断
3. **规划（Planning）**：制定多步行动计划
4. **行动（Action）**：调用工具、生成输出、与环境交互

### 记忆系统
- **短期记忆**：当前对话上下文（Context Window 内）
- **长期记忆**：持久化存储（向量数据库、文件系统）
- **工作记忆**：当前任务的中间状态
- **情景记忆**：过往经验和案例

### 工具使用（Tool Use）
- Agent 通过 Function Calling / Tool Calling 调用外部工具
- 工具描述（名称、参数、功能说明）帮助 LLM 选择正确工具
- 工具类型：搜索、代码执行、API 调用、数据库查询、文件操作

---

## 二、Agent 设计模式

### 1. ReAct（Reasoning + Acting）
最基础最经典。思考-行动-观察循环。

### 2. Plan-and-Execute
先制定完整计划，再逐步执行。适合复杂任务。

### 3. Reflection / Self-Critique
Agent 生成输出后自我评估，不满意则重做。

### 4. Tool-Use Pattern
Agent 动态选择和调用工具完成子任务。

### 5. Multi-Agent Patterns
- **Supervisor**：主 Agent 调度子 Agent
- **Debate**：多个 Agent 辩论，取最佳答案
- **Pipeline**：Agent 间流水线式传递
- **Peer-to-Peer**：Agent 间平等协作

### 6. Human-in-the-loop
关键决策点暂停，等待人工审批后继续。

---

## 三、Prompt Engineering 进阶

### Chain-of-Thought (CoT)
- 让 LLM 逐步推理，而非直接给答案
- "Let's think step by step"
- 变体：Zero-shot CoT、Few-shot CoT、Tree of Thoughts

### 上下文工程（Context Engineering）
- 2025 新趋势，比 Prompt Engineering 更广
- 不仅关注 Prompt 本身，还关注**提供给 LLM 的整体上下文**
- 包括：系统提示、用户输入、检索内容、工具返回、历史对话
- 核心：在有限 Context Window 内放入最有价值的信息

### 结构化输出
- 要求 LLM 以 JSON/YAML 格式输出
- OpenAI Structured Output / JSON Mode
- 使 Agent 能可靠解析 LLM 输出

---

## 四、MCP / A2A 等新兴协议

### MCP（Model Context Protocol）
- Anthropic 提出的开放协议
- 标准化 LLM 与外部工具/数据源的连接方式
- 类比"AI 的 USB 接口"
- Server 提供工具能力，Client（LLM 应用）调用

### A2A（Agent-to-Agent）
- Google 提出的 Agent 间通信协议
- 让不同框架/平台的 Agent 可以互相通信和协作

### ANP（Agent Network Protocol）
- 蚂蚁集团提出的智能体网络协议
- 面向开放网络环境的 Agent 互联

---

## 五、评估与监控

### Agent 评估维度
1. **任务完成率**：Agent 能否正确完成目标
2. **步骤效率**：完成任务所需步骤数
3. **工具调用准确率**：是否选择了正确的工具和参数
4. **幻觉率**：输出中不准确信息的比例
5. **延迟**：端到端响应时间
6. **成本**：Token 消耗和 API 调用费用

### 评测框架
- **RAGAS**：RAG 系统评测
- **TruLens**：LLM 应用评测
- **LangSmith**：LangChain 生态的追踪和评测平台
- **AgentBench**：Agent 能力基准测试

---

## 六、高频面试题

### 基础概念
1. AI Agent 和传统 LLM 应用的本质区别是什么？
2. Agent 的记忆系统如何设计？长短期记忆各怎么实现？
3. 什么是 Tool Use / Function Calling？Agent 如何决定用哪个工具？
4. 解释 ReAct 模式的工作原理
5. 单 Agent vs 多 Agent 系统，如何选择？

### 设计与架构
6. 如何设计一个 Agent 来完成「自动调研+写报告」的任务？
7. 多 Agent 系统中如何处理 Agent 间的通信和状态共享？
8. 如何实现 Human-in-the-loop？什么场景需要它？
9. Agent 的错误处理和容错机制如何设计？
10. 如何控制 Agent 的成本（Token 消耗、API 调用）？

### 工程实践
11. Agent 在生产环境中有哪些常见问题（循环、幻觉、成本失控）？
12. 如何调试一个复杂的 Agent 工作流？
13. 如何评估 Agent 的性能？用什么指标？
14. Prompt Engineering vs Context Engineering，区别是什么？
15. MCP 协议是什么？它解决什么问题？

### 场景设计
16. 设计一个客服 Agent 系统，要求：多轮对话、知识库检索、工单创建、转人工
17. 设计一个代码审查 Agent，要求：读取 PR、分析代码、给出建议
18. 如何构建一个能自主学习和改进的 Agent？
