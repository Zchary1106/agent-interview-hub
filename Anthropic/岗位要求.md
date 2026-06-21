# Anthropic AI Agent 相关岗位要求

> 最后更新：2026年6月（基于 Anthropic 官网 JD、面试经历汇总及公开信息整理）

---

## 一、公司背景与招聘特点

Anthropic 是目前最重视 AI Safety 的顶级 AI 公司，由前 OpenAI 团队创立。招聘特点：

- **门槛极高**：工程师素质要求接近 Top AI Research Lab
- **Safety First**：每个岗位都有专门的 Values Interview，Safety 理念是硬门槛
- **规模克制**：相比字节/谷歌，Anthropic 团队规模小，人均影响力极大
- **产品聚焦**：Claude 系列 + API + Claude.ai + Claude Code（Agent 系列）

---

## 二、核心岗位：Member of Technical Staff (AI Engineering)

### 核心职责

- 构建 Claude 的 Agent 能力：Tool Use、Computer Use、Claude Code 相关功能
- 开发和维护 Agents SDK（Python/TypeScript），设计 Agent 编排原语
- 设计 MCP（Model Context Protocol）服务端和客户端实现
- 与 Interpretability 和 Alignment 团队合作，确保 Agent 行为符合安全标准
- Context Engineering：设计 Agent 的上下文组装策略，优化信息利用效率
- 构建 Agent 评估体系（Evals），覆盖能力、安全、可靠性多维度

### 技术要求

**必须具备：**
- Python 精通（主力语言，需生产级代码质量）
- 深入理解 Claude API / LLM 工作原理（不要求非 Claude，理解通用 LLM 机制即可）
- **Context Engineering** 能力：上下文窗口管理、信息优先级设计、Token 预算分配
- **Agentic System Design**：Planning 模块、Tool Use、State Management、持久化记忆、Human-in-the-loop
- 并发/异步编程：async Python、队列设计、缓存策略
- 系统设计：能设计模块化、可扩展的 Agent 架构
- **MCP 协议**：理解 MCP 架构（Host/Client/Server）、JSON-RPC 2.0、工具暴露方式

**加分项：**
- Claude API 及 Claude Code SDK 实际项目经验
- 开源 MCP Server 开发经验（anthropic-tools/mcp 生态贡献）
- AI Safety 相关学术或工程背景
- LLM 微调 / RLHF 经验
- Rust/Go 系统编程能力

---

## 三、ML Engineer（Research Infrastructure）

### 核心职责

- 构建训练和评估基础设施，支撑 Claude 模型迭代
- 开发高效的 Agent 评估 Pipeline（Evals at Scale）
- 优化推理服务：KV Cache、批处理、低延迟流式输出
- 构建红队测试（Red Teaming）工具和自动化安全测试系统

### 技术要求

- PyTorch 深度使用经验（不仅会用，要理解底层）
- 分布式训练/推理系统经验（多卡、多机）
- 高性能 Python + 熟悉 Rust/C++ 者优先
- 理解 Transformer 架构细节（Attention、KV Cache、Speculative Decoding）

---

## 四、面试流程（2025-2026 最新）

```
Stage 1: 招聘方初筛（30min）
   - 背景介绍，岗位了解，薪资期望
   
Stage 2: Coding Assessment（CodeSignal 在线题或 Take-home）
   - CodeSignal：4道题，75分钟（主要考算法+Python实现）
   - Take-home：3-5小时开放项目（考察真实代码质量）
   
Stage 3: Hiring Manager Interview（45-60min）
   - 技术深度：深挖过往项目
   - 系统设计初探：如何设计 Agent 相关系统
   
Stage 4: Technical Loop（3-4轮，每轮 45-60min）
   - 轮次1：Coding（算法题，Python）
   - 轮次2：系统设计（Agent Infrastructure 或 LLM 系统）
   - 轮次3：Context Engineering 专题（上下文设计方法论）
   - 轮次4：Past Work Deep Dive（精选一个最复杂的项目拆解）
   
Stage 5: Values Interview（45-60min，专门独立轮次）⚠️
   - 考察对 AI Safety 和 Anthropic 使命的理解
   - 讨论你在实际工程中如何权衡能力与安全
```

---

## 五、Anthropic 面试重点考察方向

### 1. AI Safety & Values（最重要，专门单独一轮）

Anthropic 的 Safety 面试是独立 1 小时，不可轻视。典型问题：

> "Claude 的 Constitutional AI 方法论是什么？你怎么理解它的核心思路？"
> "如果你在实现一个 Agent 功能时发现它可能被滥用，你会如何处理？"
> "自主 Agent 和人类控制之间如何平衡？你有什么设计原则？"
> "你认为 AI 对齐最难的问题是什么？"

准备建议：
- 阅读 [Anthropic's Core Views](https://www.anthropic.com/company)
- 了解 Constitutional AI 论文核心思路
- 有真实的、自己的观点，展示认真思考过（不需要与官方完全一致）

### 2. Context Engineering 专题（区别于其他公司）

Anthropic 把 Context Engineering 作为独立考察方向，典型问题：

> "你如何设计一个有 20 个工具的 Agent 的 System Prompt？"
> "当 Context Window 快满时，你的 Agent 会怎么处理？"
> "如何平衡检索内容的完整性和 Token 预算？"

核心考点：
- **信息优先级**：重要信息放首位还是尾部（实验表明 LLM 对首尾关注度更高）
- **动态组装**：System Prompt 不是静态模板，要根据任务动态注入相关工具和知识
- **Token Budget 管理**：如何给 Reasoning、Tools、History、RAG 分配 Token
- **上下文衰减**：长对话中早期信息如何压缩和保留

### 3. Agentic System Design

典型题目：
> 设计 Claude Code：一个能自主完成整个开发任务的 Coding Agent

考察要点：
- Agent Loop 设计：Understand → Plan → Execute → Verify → Iterate
- 工具集定义：文件读写、命令执行、测试运行、Git 操作
- 安全模型：权限分层（只读/读写/执行），危险操作确认机制
- 上下文管理：项目文件如何高效加入上下文（按需加载 vs 预加载）
- 错误恢复：Agent 卡死/循环如何检测和处理

### 4. Claude API / MCP 深度

- Claude 系列模型的能力差异（Haiku/Sonnet/Opus 的成本-性能权衡）
- Tool Use 的实现原理（比 Function Calling 更灵活的 XML 格式）
- Extended Thinking 机制：何时启用，如何影响 Agent 推理质量
- MCP 协议全流程：从 Server 定义到 Client 消费

---

## 六、实战备考建议

1. **Claude API 实操**：至少完整做过一个使用 Tool Use + Streaming 的 Agent 项目
2. **读 Anthropic 博客**：[Anthropic Research](https://www.anthropic.com/research) 要通读最新文章
3. **Constitutional AI 论文**：至少了解核心思路，面试可能直接问
4. **Context Engineering 练习**：设计 System Prompt 时有意识地考虑信息架构
5. **Values 不能临时抱佛脚**：安全 AI 的思考要在日常积累，临时背稿子效果很差
6. **算法同样重要**：Python 算法题，复杂度分析，不输 FAANG 标准
