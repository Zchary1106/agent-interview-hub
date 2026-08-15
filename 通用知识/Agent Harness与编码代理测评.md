# Agent Harness 与编码代理深度测评

> 面试必备：市面主流 Agent Harness / 编码代理（OpenAI Codex CLI、Claude Code、LangGraph、OpenHands、CrewAI、AutoGen）的定位、架构、安全边界、源码入口与动手练习。
> 每个知识点按「要求 → 知识 → 源码入口 → 动手练习 → 掌握标准」组织，源码路径以 2026-08 各仓库 main 分支为准，版本迭代后可能漂移，请以实际仓库为准。

---

## 一、为什么面试要研究 Harness

面试考察的不只是模型能力，还有工具调用、上下文管理、记忆、评估与安全边界——这些能力全部"长在" Harness 里。模型决定"能不能"，Harness 决定"是否可用、可控、可审计"。

| 能力要求 | 对应知识 | 掌握标准 |
|---------|---------|---------|
| 讲清 agent 如何自主完成多步任务 | agent loop | 能画出「规划 → 工具调用 → 观察 → 修正」循环，并指出任一框架的实现位置 |
| 讲清工具如何被模型调用 | 工具 schema、function calling、MCP | 能对比「原生工具 / MCP 工具」接入方式与校验流程 |
| 讲清长任务如何不丢失上下文 | 上下文压缩、摘要、token budget | 能解释压缩触发时机与信息损失权衡 |
| 讲清 agent 如何记住状态 | 会话记忆、checkpoint、持久记忆 | 能区分会话级与持久级记忆，并说出实现机制 |
| 讲清 agent 的安全边界 | 审批、沙箱、命令白名单、审计 | 能区分「信任标记」与「真沙箱」，并给出攻击面 |
| 讲清如何衡量 agent 好坏 | benchmark、评估集、可观测性 | 能说出至少 2 个 benchmark 与 2 种轨迹观测手段 |

---

## 二、共同考察点（七个必考维度）

### 1. Agent Loop（agent 循环）

- **要求**：解释一个 agent 为什么能自主完成多步任务，而不是单次问答。
- **知识**：循环 = 规划（把任务拆成步骤）→ 工具调用（模型输出结构化工具请求）→ 观察（执行结果回填）→ 修正（依据结果调整下一步），直到满足终止条件。工程上还包含：重试与错误恢复、最大迭代上限、循环守卫（防止死循环）。
- **源码入口**：
  - OpenAI Codex CLI：`codex-rs/core/src/agent/`（turn 控制、agent registry）
  - Claude Code：`src/Task.ts`（任务态）+ `src/query/`（每轮查询与 token 预算）
  - LangGraph：`langgraph/pregel/`（superstep 图执行运行时）
- **动手练习**：跑一个 Codex CLI 会话，记录一次任务里"模型响应 → 工具执行 → 结果回填"发生了多少次、失败后如何重试。
- **掌握标准**：能讲清"一次 turn 内模型响应 → 工具执行 → 观察结果 → 下一轮"的完整链路，并能指出至少一个循环守卫（迭代上限 / 退出条件）。

### 2. 工具调用（Tool Calling）

- **要求**：讲清模型如何"调用"工具，以及工具失败时 agent 如何恢复。
- **知识**：模型输出结构化工具请求（JSON schema 对齐），运行时校验参数、执行、返回结果；工具注册表统一 schema；错误可回灌给模型让它修正参数或换工具。MCP（Model Context Protocol）把工具做成标准化 server，`tools/list` / `tools/call` 两个原语。
- **源码入口**：
  - OpenAI Codex CLI：`codex-rs/core/src/function_tool.rs`、`codex-rs/core/src/mcp.rs`
  - Claude Code：`src/tools.ts`、`src/tools/`（BashTool / FileEditTool / WebFetchTool / MCPTool 等 40+ 工具）
  - AutoGen：`autogen_core` 工具协议 + agentchat 内工具注册
- **动手练习**：用 Codex CLI 或 Claude Code 观察一次参数校验失败（如命令不存在），记录错误如何回灌给模型。
- **掌握标准**：能说出工具调用的完整生命周期：schema 声明 → 参数校验 → 执行 → 结果/错误回填。

### 3. 上下文管理（Context Management）

- **要求**：长任务下模型上下文放不下时怎么办。
- **知识**：手段包括——截断（丢弃最早消息）、摘要（LLM 压缩历史）、结构化检索（把对话转成可查询索引）、token budget（按预算自动触发压缩/警告）、子代理隔离上下文（把子任务上下文与主线程隔离）。
- **源码入口**：
  - OpenAI Codex CLI：`codex-rs/core/src/compact*.rs`（压缩与 token 预算）
  - Claude Code：`src/services/compact/`、`src/query/tokenBudget.ts`、`src/services/AgentSummary/`（子代理摘要）
- **动手练习**：跑一个长会话，观察压缩何时触发、压缩后哪些信息保留。
- **掌握标准**：能对比「截断 vs 摘要 vs 检索」的取舍，并说出至少一个 token budget 触发点。

### 4. 记忆（Memory）

- **要求**：agent 如何在会话之间记住信息。
- **知识**：会话级记忆（对话历史、checkpoint 状态快照）与持久级记忆（记忆文件、向量库、用户画像）。LangGraph 用 Checkpointer 持久化每个节点的状态，支持断点恢复与"时间旅行"；Claude Code 有 SessionMemory 与记忆提取（autoDream/extractMemories）。
- **源码入口**：
  - LangGraph：`langgraph/checkpoint/`（BaseCheckpointSaver 及 memory/sqlite/postgres 实现）
  - Claude Code：`src/services/SessionMemory/`、`src/services/extractMemories/`
- **动手练习**：用 LangGraph 跑一个带 checkpointer 的图，中断后恢复，观察状态如何回放。
- **掌握标准**：能区分会话记忆与持久记忆，并解释 checkpoint 为什么能实现"断点续跑"。

### 5. 安全边界（Security Boundary）

- **要求**：agent 能执行命令时，如何防止它做不该做的事。
- **知识**：安全层次——①审批（人在环上决定是否放行）；②策略（命令白名单/黑名单、敏感路径保护）；③沙箱（操作系统级隔离：容器、Landlock、seccomp）；④审计（记录每条命令与理由）。关键区分：**"信任标记"不是沙箱**，只有执行层隔离才是真边界。
- **源码入口**：
  - OpenAI Codex CLI：`codex-rs/core/src/exec_policy.rs`（审批策略）、`codex-rs/core/src/sandboxing/` + `codex-rs/linux-sandbox`（沙箱）
  - Claude Code：`src/tools/BashTool/bashPermissions.ts`、`src/tools/BashTool/bashSecurity.ts`
  - OpenHands：Docker 沙箱运行时（Agent Server / software-agent-sdk）
- **动手练习**：对照 Codex CLI 与 Claude Code 的权限模型，整理一份「终端 Harness 权限设计」对比表。
- **掌握标准**：能对任意框架回答三问——命令在谁的权限下执行？是否需要审批？有无操作系统级隔离？

### 6. 评估（Evaluation）

- **要求**：如何衡量一个 agent / Harness 的真实能力。
- **知识**：代码类 benchmark（SWE-bench、Terminal-Bench、Aider 基准）、通用 agent 基准（GAIA、WebArena）、任务回放与人工评分；评估维度包括完成率、成本、延迟、安全违规次数。
- **源码入口**：
  - OpenHands 以 SWE-bench 闻名（经典版论文与评估集），现评估能力沉淀到 software-agent-sdk
  - OpenAI Codex CLI：`evaluation/` 目录（离线评估集）
- **动手练习**：用任一公开 benchmark 的 sample 集跑一个框架，记录 pass@1 与成本。
- **掌握标准**：能说出 2 个 benchmark 名称、它们测什么、为什么不能只看准确率。

### 7. 可观测性（Observability）

- **要求**：agent 出错时如何定位、回放、审计。
- **知识**：事件流（把动作、观察、状态变化都记为事件，天然可回放）、轨迹导出（Trace/Replay）、结构化日志与 OpenTelemetry、LLM 调用成本统计。
- **源码入口**：
  - OpenHands 经典架构：EventStream（一切皆事件，可回放审计）
  - OpenAI Codex CLI：`codex-rs/core/src/otel_init.rs`（OpenTelemetry 初始化）
- **动手练习**：画出一条「动作 → 沙箱执行 → 观察事件」的链路，说明每个环节如何被记录。
- **掌握标准**：能解释"事件溯源"为什么比"日志行"更适合 agent 调试。

---

## 三、横向对比矩阵

| 框架 | 形态 | 定位 | 核心抽象 | 执行边界 | 主要语言 | 源码入口 | 适合场景 |
|---|---|---|---|---|---|---|---|
| OpenAI Codex CLI | 终端编码代理 | 单代理真实工程任务 | Agent loop + 审批策略 | 本地命令 + 可选沙箱 | Rust / TS | openai/codex | 日常开发、PR 自动化、终端自动化 |
| Claude Code | 终端编码代理 | 长任务、仓库深度理解 | 会话代理 + 权限提示 | 本地 + 分级审批 | TypeScript（闭源） | anthropics/claude-code | 长任务、子代理、与 IDE 协作 |
| LangGraph | 图编排框架 | 有状态复杂工作流 | StateGraph + Pregel + Checkpointer | 由应用层决定 | Python | langchain-ai/langgraph | 复杂流程、人工介入（HITL） |
| OpenHands | 自主 SWE 代理 → Agent 控制台 | 端到端代码任务与多代理管控 | EventStream（经典）/ Agent Server + ACP（现状） | Docker 沙箱 | Python / TS | All-Hands-AI/OpenHands | SWE-bench、自主开发、多 Harness 统一管理 |
| CrewAI | 多代理团队框架 | 角色化协作流程 | Agent + Task + Crew + Process | 无内置沙箱 | Python | crewAIInc/crewAI | 团队式流程、内容生产管线 |
| AutoGen | 多代理对话框架 | 对话驱动的代理协作 | ConversableAgent + GroupChat | 代码执行器可选隔离 | Python | microsoft/autogen | 多代理研究、对话式协作、群聊调度 |

---

## 四、逐个深度测评

### 1. OpenAI Codex CLI

**定位**：OpenAI 开源的终端编码代理，自然语言驱动真实工程任务——读仓库、跑命令、改代码、提交 PR。

**核心架构**：
- **Agent loop**：任务规划 → 工具调用（shell / 文件 / grep / apply_patch / web / MCP）→ 观察 → 重试，直到满足终止条件。
- **审批策略**：`AskForApproval` 枚举（`codex-rs/core/src/exec_policy.rs`）——`Never`（禁止自动批准）、`OnRequest`（请求确认后执行）、`UnlessTrusted`（受信任环境自动放行）、`Granular`（规则化细粒度策略，可叠加沙箱审批）。
- **多代理/会话**：支持子代理（subagent）、多会话线程（thread）与历史压缩，架构已演化为 turn 控制器 + agent registry + 沙箱模块。

**源码入口**（openai/codex，2026-08）：
- `codex-rs/core/src/agent/`：agent 注册表、turn 控制（control.rs）、状态机
- `codex-rs/core/src/exec_policy.rs`：执行与审批策略判定
- `codex-rs/core/src/sandboxing/` + `codex-rs/linux-sandbox`：沙箱实现
- `codex-rs/core/src/compact*.rs`：上下文压缩与 token 预算
- `codex-rs/core/src/mcp.rs`：MCP 工具接入

**安全边界**：默认在本地以当前用户权限执行命令；Project Trust 是"信任标记"，不是沙箱。真正边界 = 审批策略（Never/OnRequest/UnlessTrusted/Granular）+ 操作系统级沙箱（Landlock/seccomp 等）。面试时务必说清：**信任 ≠ 隔离**。

**动手练习**：
1. 跑一个 Codex CLI 会话，观察它如何拆解任务、请求审批、失败后重试。
2. 读 `agent/control.rs` 与 `exec_policy.rs`，确认 turn 循环与审批判定顺序。
3. 配一个 `Granular` 策略（如禁止 `rm -rf`），观察命中策略时命令被拦截。

**面试要点**：
- Q：Codex CLI 的安全边界是什么？A：本地以用户权限执行 + 审批策略 + 可选沙箱；Project Trust 只是信任标记。
- Q：审批策略有哪几种？A：Never / OnRequest / UnlessTrusted / Granular（2026 版，见 `AskForApproval`）。
- Q：agent loop 在源码哪里？A：`codex-rs/core/src/agent/` 的 turn 控制。

---

### 2. Claude Code

**定位**：Anthropic 的终端代理式编码助手，强调长任务、仓库理解与权限提示（Permission Prompt）。

**核心架构**：
- **会话代理循环**：`Task.ts` 管理任务态，`query/` 驱动每轮查询与 token 预算；工具集 40+，覆盖文件读写、Bash、Grep、WebFetch、MCP、子代理（AgentTool）、TodoWrite、Skills。
- **权限提示分级**：工具执行前按风险分级请求确认；敏感目录与破坏性命令（如 `rm -rf`）有额外检查（`destructiveCommandWarning`）。
- **hooks**：配置级可编程拦截（PreToolUse / PostToolUse 等），可用于强制安全策略。
- **记忆**：SessionMemory + 压缩（compact）+ 记忆提取（extractMemories）。

**源码入口**（npm 包 v2.1.88 源码映射恢复，本机 `~/MyWork/claudecodesource/restored-src`）：
- `src/Task.ts`、`src/query/`：任务态与查询循环
- `src/tools/BashTool/bashPermissions.ts`、`src/utils/permissions/`、`src/types/permissions.ts`：权限模型
- `src/tools/BashTool/bashSecurity.ts`：命令安全检查
- `src/tools/AgentTool/`：子代理工具
- `src/services/SessionMemory/`、`src/services/compact/`：记忆与压缩
- `src/tools/MCPTool/`、`src/services/mcp/`：MCP 客户端

**安全边界**：Bash 工具默认需要确认，敏感目录/命令有额外防护；hooks 可编程拦截；最终边界由用户审批决定，**非容器隔离**。

**动手练习**：
1. 配置一个 PreToolUse hook 拦截危险命令，验证拦截生效。
2. 对照 Codex CLI 的审批模式，整理一份「终端 Harness 权限设计」对比表（提示时机、可配置性、审计能力）。

**面试要点**：
- Q：Claude Code 与 Codex CLI 的权限模型差异？A：两者都是"本地执行 + 审批"，Claude Code 以权限提示分级 + hooks 见长，Codex CLI 以策略枚举（Never/OnRequest/…）+ 沙箱见长。
- Q：Claude Code 为什么不是沙箱？A：Bash 在用户环境直接执行，靠审批与检查兜底；无容器级隔离。

---

### 3. LangGraph

**定位**：LangChain 生态的图式 Agent 编排框架，把代理建模成**有状态状态机**。

**核心架构**：
- **StateGraph**：节点（Node）+ 边（Edge）+ 共享状态（State，TypedDict）+ 条件路由（conditional edges）。
- **Pregel 运行时**：以 superstep 方式执行图，每个节点读写共享 state 的增量。
- **Checkpointer**：持久化每次状态快照，支持断点恢复、时间旅行、人工回退。
- **HITL（Human-in-the-loop）**：`interrupt_before / interrupt_after` 挂起执行等待人工输入。

**源码入口**（langchain-ai/langgraph，2026-08）：
- `libs/langgraph/langgraph/graph/state.py`：StateGraph 定义
- `libs/langgraph/langgraph/pregel/`：Pregel 执行运行时（loop / algo / types）
- `libs/langgraph/langgraph/checkpoint/`：BaseCheckpointSaver（memory / sqlite / postgres）

**安全边界**：框架本身不提供运行时沙箱；工具执行权限由你的应用层控制，敏感工具必须自己加授权与审计。

**动手练习**：用一个 3 节点图（规划 → 工具 → 总结）实现带循环重试的 agent，加上 checkpointer 后中断/恢复，验证状态如何在节点间流动。

**面试要点**：
- Q：checkpointer 解决什么问题？A：状态持久化与断点恢复，支持"时间旅行"和人工回退。
- Q：StateGraph 与普通链式调用（LCEL）的区别？A：图支持循环、分支、共享状态；链是单向管道。

---

### 4. OpenHands

**定位**：开源自主软件工程代理。**2026 年已转型**：仓库主体从"单一代理"演化为 **Agent Canvas**——自托管的多 Harness 控制台。

**核心架构**：
- **经典架构（2023-2025，面试常考）**：EventStream——代理动作、观察、状态变化都是事件；动作在 Docker 沙箱中执行（ActionExecution），天然可回放、可审计；以 SWE-bench 闻名。
- **现状（2026）**：Agent Canvas（React/TS 前端）+ OpenHands Agent Server（`OpenHands/software-agent-sdk`）+ **ACP**（Agent Client Protocol，JSON-RPC over stdio）——以子进程方式拉起 Claude Code、Codex、Gemini CLI，或内置 OpenHands agent；一个控制台统一管理多个 Harness。

**源码入口**（2026-08）：
- `All-Hands-AI/OpenHands`（Agent Canvas）：`src/api/`（服务适配器）、`src/components/`、`src/stores/`（Zustand 状态）、`docs/ACP_AGENTS.md`、`docs/architecture.md`
- `OpenHands/software-agent-sdk`（OpenHands V1）：Agent Server 实现（`openhands-agent-server/`）
- 经典 EventStream 实现见历史版本源码

**安全边界**：Agent Server 拥有子进程与凭据，Canvas 不直接执行动作；Docker 沙箱隔离代理执行；本地 `npm run dev` 模式下 agent 有宿主机文件系统访问，**仅限可信环境**。

**动手练习**：
1. 读 `docs/ACP_AGENTS.md`，画出一条「Canvas → Agent Server → ACP 子进程 → LLM」的完整链路。
2. 本地启动 dev，切换 Codex / Claude Code 两种后端，观察凭据归属与沙箱差异。
3. 整理「专用 SWE agent → 通用控制平面 + ACP」的演进谈资，这是 2026 面试加分点。

**面试要点**：
- Q：OpenHands 的事件流为什么可审计？A：动作与观察都是事件，状态变更可全量回放。
- Q：ACP 是什么？A：Agent Client Protocol，通过 JSON-RPC over stdio 标准化"客户端 ↔ 编码代理"交互，让控制台可以拉起任意兼容代理。
- Q：Agent Canvas 与经典 OpenHands 的关系？A：Canvas 是控制平面，执行仍交给 Agent Server / ACP 代理，Docker 沙箱保留。

---

### 5. CrewAI

**定位**：多角色协作框架，把任务拆给不同专业代理，按流程并行/串行推进。

**核心架构**：
- **Agent**：Role / Goal / Backstory 定义角色与行为约束。
- **Task**：定义产出（expected output）、工具与上下文。
- **Crew**：定义协作流程（Process）与代理集合。
- **Process**：sequential（顺序执行）/ hierarchical（层级委派，manager 代理协调）。
- **Flow**：事件驱动的无向图流程（`@start` / `@listen` 装饰器），适合把 Crew 串成管线。

**源码入口**（crewAIInc/crewAI，2026-08，monorepo）：
- `lib/crewai/src/crewai/crew.py`：Crew 编排
- `lib/crewai/src/crewai/agent.py`、`task.py`：角色与任务
- `lib/crewai/src/crewai/process.py`：顺序 / 层级流程
- `lib/crewai/src/crewai/flow/flow.py`：事件驱动 Flow

**安全边界**：不内置沙箱；代理的工具（含任意代码执行）以应用进程身份运行，权限完全取决于你注册的工具实现。

**动手练习**：搭一个「调研员 → 评审员 → 总结员」三角色 Crew，观察任务产物如何在代理间流转；再用 Flow 把两个 Crew 串成管线。

**面试要点**：
- Q：CrewAI 的 Process 有哪几种？A：sequential 与 hierarchical（manager 委派）。
- Q：CrewAI 如何保证安全？A：框架不兜底，必须自己控制工具权限与审计。

---

### 6. AutoGen

**定位**：微软的多代理对话框架，以对话式消息驱动代理协作与工具执行。

**核心架构**：
- **ConversableAgent**：代理之间通过消息对话完成任务；每次消息触发 reply 逻辑（LLM / 工具 / 人工）。
- **GroupChat**：群聊调度——RoundRobinGroupChat（轮转）、SelectorGroupChat（LLM 选择发言者）、MagenticOne 等编排。
- **代码执行器**：`DockerCommandLineCodeExecutor` / `LocalCommandLineCodeExecutor` 等，隔离程度可选。
- **v0.4+ 架构分层**：`autogen-core`（Actor 运行时、消息路由）与 `autogen-agentchat`（agent/team 高层 API）。

**源码入口**（microsoft/autogen，2026-08）：
- `python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py`：AssistantAgent 主循环
- `python/packages/autogen-agentchat/src/autogen_agentchat/agents/_user_proxy_agent.py`：UserProxy（人工/代码执行代理）
- `python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/`：群聊团队（轮转/选择器/MagenticOne）
- `python/packages/autogen-core/src/autogen_core/`：Actor 运行时与消息协议

**安全边界**：代码执行器可选 Docker / Jupyter / 本地；**默认本地执行有风险，生产应强制 Docker 隔离**。

**动手练习**：跑一个两代理对话（UserProxy + Assistant），观察工具调用如何通过消息往返完成；再把执行器切到 Docker，对比隔离差异。

**面试要点**：
- Q：AutoGen 多代理如何"对话"？A：消息在代理间传递，每次消息触发 reply；GroupChatManager 负责发言调度。
- Q：AutoGen 的代码执行安全？A：执行器可选，生产必须 Docker/Jupyter 隔离，不能默认本地。

---

## 五、选型与学习路线

### 如何选型

| 诉求 | 推荐 | 理由 |
|---|---|---|
| 日常开发/终端自动化 | Codex CLI、Claude Code | 开箱即用，审批流成熟 |
| 复杂有状态工作流、需人工介入 | LangGraph | 图编排 + checkpoint + HITL 最完整 |
| 端到端自主代码任务、要强隔离 | OpenHands | Docker 沙箱 + 事件流可审计 |
| 团队式/角色化内容流程 | CrewAI | 角色抽象直接，落地快 |
| 多代理对话研究/群聊调度 | AutoGen | 对话原语最灵活，生态研究向 |
| 统一管理多种 Harness | OpenHands Agent Canvas | ACP 拉起 Codex/Claude Code/Gemini CLI |

### 学习路线

1. **先精读 1–2 个**：推荐 Codex CLI 的 `agent/control.rs` + `exec_policy.rs`（理解 loop 与审批），再选 LangGraph 的 `pregel/` 或 OpenHands 的 ACP 链路（理解状态与沙箱）。
2. **横向对比安全边界**：对每个框架回答三问——命令在谁的权限下执行？是否需要审批？有无操作系统级隔离？整理成对比表。
3. **整理自己的面经**：把每个框架按「定位 → 架构 → 安全边界 → 源码入口 → 练习」固定结构讲一遍，面试时按同一结构输出。
4. **动手验证**：每个框架至少跑一个最小示例（本文件「动手练习」均已给出），把源码路径记进自己的笔记。

---

## 六、自测题（10 道，附参考答案）

**1. 什么是 agent loop？请以 Codex CLI 为例画出一条实现链路。**

> agent loop 是「规划 → 工具调用 → 观察 → 修正」的循环。Codex CLI 中，turn 控制器（`codex-rs/core/src/agent/`）把用户请求交给模型，模型输出工具请求，`exec_policy.rs` 判定审批与执行策略，工具执行后结果回填，模型继续下一轮直到终止条件。

**2. 对比 Codex CLI 与 Claude Code 的权限模型。**

> 两者都是"本地执行 + 审批"。Codex CLI 用策略枚举（Never / OnRequest / UnlessTrusted / Granular，`exec_policy.rs`）+ 可选沙箱（linux-sandbox）；Claude Code 用权限提示分级 + Bash 命令检查（`bashPermissions.ts` / `bashSecurity.ts`）+ hooks 可编程拦截。边界都是用户审批，非默认容器隔离。

**3. "Project Trust" 是沙箱吗？为什么面试官爱问这个？**

> 不是。Project Trust 是信任标记，决定是否减少提示；真正的安全边界是审批策略 + 沙箱执行层。面试官用这个问题考察你能否区分"信任"与"隔离"。

**4. LangGraph 的 Checkpointer 解决什么问题？interrupt 呢？**

> Checkpointer 持久化每个节点执行后的状态快照，实现断点恢复、时间旅行与人工回退；interrupt（`interrupt_before/after`）在执行中挂起等待人工输入，是 HITL 的基础。

**5. OpenHands 经典架构为什么可审计？**

> 事件流架构（EventStream）把代理动作、观察、状态变化都记为事件，执行在 Docker 沙箱中发生；任何一步都可以全量回放，天然满足审计与调试需求。

**6. ACP 是什么？Agent Canvas 如何用 ACP 拉起第三方代理？**

> ACP（Agent Client Protocol）是客户端与编码代理之间基于 JSON-RPC over stdio 的标准协议。Agent Canvas 的 Agent Server 把 Codex / Claude Code / Gemini CLI 等作为子进程拉起，通过 JSON-RPC 转发会话消息；代理自己管理 LLM、工具与执行，Canvas 只渲染与管控。

**7. CrewAI 的 Process 有哪几种？**

> 两种：sequential（顺序执行）与 hierarchical（manager 代理层级委派）；另可用 Flow（事件驱动）把多个 Crew 串成更复杂的管线。

**8. AutoGen 的代码执行器为什么默认本地有风险？生产如何隔离？**

> 本地执行器直接以进程身份运行任意代码，等于把宿主机暴露给模型输出。生产应使用 `DockerCommandLineCodeExecutor` / Jupyter 等隔离执行环境，并配合网络与资源限制。

**9. 长任务上下文放不下时有哪些手段？各自取舍？**

> 截断（简单但丢信息）、摘要（保留语义但可能失真）、结构化检索（可查询但工程复杂）、token budget（按预算触发压缩/告警）、子代理隔离（把子任务上下文移出主线程）。选型取决于任务对细节 vs 语义的依赖程度。

**10. 如何评估一个 Harness？给出 benchmark 与可观测性手段。**

> 能力侧用 SWE-bench（代码任务）、Terminal-Bench（终端命令）、GAIA / WebArena（通用 agent）等；工程侧看完成率、成本、延迟与安全违规数；可观测性用事件流/轨迹回放（OpenHands EventStream）、结构化日志与 OpenTelemetry（Codex CLI `otel_init.rs`）、LLM 成本统计。

---

## 附：关联文档

- [Agent 框架全景](Agent框架全景.md)：LangChain、LangGraph、AutoGen、CrewAI 等框架横向概述
- [Agentic Coding 与 AI 编程工具](Agentic%20Coding与AI编程工具.md)：Claude Code / Cursor / Copilot 对比与 Hooks
- [Agent 安全与评估体系](Agent安全与评估体系.md)：Prompt 注入防御、幻觉检测、评估框架
- [MCP 与工具生态](MCP与工具生态.md)：Model Context Protocol 架构与开发实战
