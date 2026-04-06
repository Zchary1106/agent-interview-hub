# Agent 核心概念面试题

## 基础概念

### 1. Agentic AI vs 传统 AI vs 生成式 AI 的区别？
**考察点**：理解 Agent 的自主性、工具使用、规划能力

### 2. 什么是 ReAct 模式？与 CoT（Chain-of-Thought）有什么区别？
**考察点**：Thought-Action-Observation 循环 vs 纯推理链

### 3. Agent 的核心组件有哪些？
**参考答案**：LLM（大脑）+ Memory（记忆）+ Tools（工具）+ Planning（规划）

### 4. 什么场景不该用 Agent？
**考察点**：简单规则能搞定的、低延迟要求的、成本敏感的

### 5. 解释 Function Calling / Tool Use 的工作原理
**考察点**：LLM 如何决定调用工具、参数如何传递、结果如何回传

## 架构与模式

### 6. 解释"主管模式"和"编排模式"在多 Agent 系统中的区别
**考察点**：Supervisor vs Orchestrator，各自适用场景

### 7. 什么是 MCP（Model Context Protocol）？它解决什么问题？
**考察点**：标准化的工具接入协议

### 8. Agent 的记忆有哪些类型？
**参考**：短期（上下文窗口）、长期（向量 DB / RAG）、实体记忆、工作记忆

### 9. 如何设计 Agent 的规划能力？
**考察点**：任务分解、子目标设定、计划执行与修正

### 10. 什么是 Agentic RAG？与传统 RAG 有什么区别？
**考察点**：Agent 主动决定何时检索、检索什么、如何组合

## 工程实践

### 11. 如何评估 Agent 的性能？
**参考**：任务成功率、逻辑一致性、工具使用准确性、延迟、成本

### 12. Agent 中的幻觉问题如何缓解？
**考察点**：RAG、验证步骤、confidence scoring、human-in-the-loop

### 13. 如何处理 Agent 的错误恢复？
**考察点**：重试、回退、换路径、升级

### 14. 如何优化 Agent 系统的成本？
**参考**：模型分流、prompt 缓存、批处理、简单任务不用 Agent

### 15. 如何防范 Prompt Injection？
**考察点**：输入清洗、权限隔离、输出验证、系统提示保护

## 高级问题

### 16. 多 Agent 系统中如何处理"任务完成"的判断？
**考察点**：终止条件、质量验证、防止无限循环

### 17. 如何在生产环境中调试一个出问题的 Agent？
**考察点**：日志、trace、可观测性、replay

### 18. Agent 的自主性边界应该如何定义？
**考察点**：哪些操作需要审批、风险等级划分

### 19. 解释 Agent 的"延迟 vs 准确性"权衡
**考察点**：多次 LLM 调用提高准确性 vs 用户等待体验

### 20. 如何设计 Agent 的版本管理和灰度发布？
**考察点**：prompt 版本化、模型切换、流量分配

### 21. Agent Skill 是什么？与 Plugin 有什么区别？
**考察点**：Skill 是知识包（SKILL.md + 资源），Plugin 是代码包

### 22. 如何让 Agent 从经验中学习改进？
**考察点**：反思机制、记忆更新、few-shot 积累

### 23. 比较 LangChain / LangGraph / CrewAI / AutoGen 的适用场景
**考察点**：各框架的设计哲学和取舍
