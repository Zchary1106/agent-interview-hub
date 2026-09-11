# Coding Agent 源码架构对比

> 更新日期：2026-09-11。本文只把官方仓库、官方文档或论文中能定位的内容写成“事实”；跨项目优劣判断标为“工程推断”。面试时不要只背产品功能，要能沿着入口、循环、上下文、工具、执行环境、会话和协议追到源码边界。

## 一、先建立统一分析框架

分析 Coding Agent，至少回答十个问题：

1. **入口与 Bootstrap**：CLI、IDE、Web 或 App 如何创建一次会话？
2. **Agent Loop**：谁负责 `model → tool → observation → model` 循环和停止条件？
3. **Tool Registry**：工具如何注册、描述、筛选、调用和返回结构化错误？
4. **Context Assembly**：系统指令、仓库规则、历史、工具结果如何拼装与压缩？
5. **Session**：会话如何持久化、恢复、分叉和归档？
6. **Execution**：命令和文件编辑在哪里执行，如何限制资源和副作用？
7. **Permission**：谁决定“允许做什么”，与 Sandbox 的“即使做了也能被隔离”有何不同？
8. **Extension**：Skill、Plugin、Hook、MCP 各自扩展哪一层？
9. **Client Protocol**：前端如何订阅流式事件、审批和文件变更？
10. **Trace**：能否重放决策轨迹并定位失败责任？

```text
Client/CLI
   ↓ protocol
Session / Turn / Event
   ↓
Context Builder → Model Adapter
   ↑                 ↓ tool call
Checkpoint ← Agent Loop → Tool Router → Policy → Runtime/Sandbox
   ↓                                           ↓
Trace / Artifact / Eval                       Observation
```

## 二、四种代表性架构

| 项目 | 主要架构取向 | 源码阅读重点 | 最适合回答的面试问题 |
|---|---|---|---|
| OpenAI Codex | Rust 核心 + App Server 协议 + Thread/Turn/Item 生命周期 | `codex-rs`、app-server protocol、审批与沙箱 | 如何让同一 Agent 支撑 CLI/IDE/桌面客户端 |
| OpenHands | 事件流 + Action/Observation + 独立 Runtime | event stream、controller、runtime client/server | 如何隔离任意代码执行并支持多种 Runtime |
| SWE-agent | Agent-Computer Interface（ACI）+ 配置化工具/环境 | agent、environment、tools、history processor | 工具界面设计为什么会显著影响解题成功率 |
| DeepSeek Harness | Cordis 微内核，“一切皆插件” | Context、插件挂载/卸载、依赖、profile/patch | 如何做到模型、循环、工具、会话和 UI 可替换 |

> 这不是排行榜。Codex 强在产品协议和会话生命周期；OpenHands 强在 Action/Observation 与 Runtime 边界；SWE-agent 强调 ACI 对模型行为的影响；DeepSeek Harness 把可组合插件提升为内核原则。

## 三、OpenAI Codex：协议驱动的多客户端 Agent

### 3.1 可验证结构

官方 `codex app-server` 文档把交互抽象为：

```text
Thread → Turn → Item
```

- Thread：持续会话；
- Turn：一次用户输入到 Agent 响应；
- Item：消息、推理、命令、文件编辑等可持久化单元；
- 客户端可 start、resume、fork Thread，并通过通知订阅 Turn 进度；
- App Server 使用双向 JSON-RPC 风格协议，支持 stdio JSONL，并能生成匹配当前版本的 TypeScript/JSON Schema。

这意味着客户端不应解析终端文本来猜状态，而应消费带版本的协议事件。

### 3.2 面试可讲的设计点

1. **核心与 UI 解耦**：CLI、IDE、桌面端可以共享生命周期语义。
2. **审批不是沙箱**：审批解决用户意图与授权，沙箱限制进程实际能力，两者需要同时存在。
3. **恢复与分叉是一等能力**：`resume` 继续原历史，`fork` 复制历史后形成新实验分支。
4. **背压是协议问题**：事件队列有界时，客户端必须识别可重试错误并退避，不能无限堆积。

### 3.3 深挖追问

- 如果文件已经修改，但客户端在收到事件前断线，恢复后如何避免重复执行？
- Thread 历史与工作区 Git 状态不一致时，以谁为准？
- 协议版本升级时，旧客户端如何兼容新 Item 类型？

## 四、OpenHands：事件驱动与 Runtime 隔离

### 4.1 可验证结构

OpenHands 用 Action/Observation 表达 Agent 与环境的交互，并把任意代码执行放入 Runtime。官方 Runtime 文档描述了后端与容器内 Action Execution Server 的 client-server 通信：后端发送动作，Runtime 执行后返回 observation。

```text
Agent/Controller
      ↓ Action
Event Stream → Runtime Client → Action Executor in Sandbox
      ↑ Observation
```

### 4.2 面试可讲的设计点

- **事件是协作协议**：模型动作、用户输入、工具结果可以进入统一时间线。
- **Runtime 是安全与可复现边界**：容器、资源控制、工作目录和插件加载不应散落在 Agent Loop。
- **替换 Runtime 不应改 Agent 策略**：本地、Docker 或远端执行环境应实现相同动作接口。

### 4.3 风险

- Event Stream 不是自动等于 Event Sourcing；若事件不完整、不可重放或副作用无幂等键，仍无法可靠恢复。
- 容器不是完整安全方案，还要限制挂载、网络、凭证、内核能力和资源。

## 五、SWE-agent：Agent-Computer Interface 决定上限

SWE-agent 的核心观点是：模型面对仓库时使用什么命令、观察到什么输出、历史如何裁剪，会显著影响表现。工具签名、文档、输出格式和上下文处理都属于 ACI，而不是“外围胶水”。

### 源码阅读路径

```text
Config
 ├─ Agent: prompt、history processor、forward/stop
 ├─ Tools/ACI: command schema、parser、error handling
 └─ Environment: repository、process、observation
```

### 面试结论

- 给模型一个通用 Shell，不一定优于少量语义明确的工具；
- 工具输出过长会挤压历史，过短则丢失诊断线索；
- Benchmark 提升可能来自模型，也可能来自 ACI、Prompt、检索或环境修复，必须做消融实验。

## 六、DeepSeek Harness：插件化微内核

官方 Cordis 教程说明，模型适配器、工具、文件访问乃至 Agent Loop 都是挂载到共享 Context 的插件。插件具有挂载、卸载与依赖关系，配置决定最终系统组合。

```text
Cordis Context
 ├─ model plugin
 ├─ loop plugin
 ├─ tool / MCP plugin
 ├─ session / storage plugin
 ├─ sandbox plugin
 ├─ scheduler plugin
 └─ UI plugin
```

### 优势（工程推断）

- 能以统一生命周期替换横切组件；
- Profile/配置组合适合构建不同产品形态；
- 插件可独立测试和按需装载。

### 代价（工程推断）

- 动态依赖和生命周期顺序更难调试；
- “都可替换”会扩大兼容矩阵；
- 第三方插件必须有来源、权限、版本、签名、隔离和撤销机制；
- 微内核很薄不代表系统简单，复杂度可能转移到配置与依赖图。

## 七、横向对比：不要混淆这些边界

| 维度 | 常见错误 | 更成熟的回答 |
|---|---|---|
| Loop | 一个 while 循环就是 Agent | 包含停止、预算、取消、重试、状态与恢复 |
| Tool | JSON Schema 能调用就行 | 还需权限、超时、幂等、错误类型、结果截断和版本 |
| Sandbox | Docker 就安全 | 还需网络、挂载、凭证、资源、系统调用和审计边界 |
| Permission | 弹确认框就安全 | 必须绑定用户、资源、动作、时效和实际执行身份 |
| Context | 历史全塞给模型 | 需要选择、压缩、来源、缓存、丢失检测和回滚 |
| Session | 保存聊天文本 | 还要保存执行状态、工作区版本、Artifact 和副作用证据 |
| Plugin | 能动态 import | 还要管理依赖、生命周期、冲突、能力声明和供应链风险 |
| Trace | 打印日志 | 应能关联输入、模型版本、工具参数、状态变化与最终证据 |

## 八、源码审计 Checklist

```text
[ ] 锁定仓库 commit/tag 与阅读日期
[ ] 找到真实入口，不从 README 架构图反推源码
[ ] 画出一次 turn 的调用链
[ ] 列出状态的唯一事实来源
[ ] 区分 permission、policy、sandbox、approval
[ ] 找到 context 构建和压缩发生的位置
[ ] 找到工具注册、选择、执行、错误回传四个位置
[ ] 验证 cancel/resume/fork 是否真的贯穿执行层
[ ] 检查副作用工具是否支持幂等和补偿
[ ] 检查客户端协议、事件顺序与背压
[ ] 用一次真实失败 trace 验证架构图
```

## 九、高频面试题

1. Coding Agent 与普通 Chat Agent 的架构差异是什么？
2. 为什么 Permission 和 Sandbox 不能二选一？
3. Codex 的 Thread/Turn/Item 抽象解决了什么问题？
4. OpenHands 为什么将 Runtime 做成独立 client-server 边界？
5. ACI 为什么可能比“增加一个工具”更重要？
6. 插件化 Agent Loop 会带来哪些测试和治理成本？
7. 如何设计可恢复的文件编辑与命令执行？
8. 如何证明一次 Coding Agent 任务真的完成，而不是模型自述完成？
9. 你会如何对四个系统做公平 Benchmark？
10. 如果让你从零实现最小 Coding Agent，哪些模块先做，哪些暂缓？

## 十、参考资料（官方/论文）

- [OpenAI Codex repository](https://github.com/openai/codex)
- [Codex App Server protocol](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
- [OpenHands repository](https://github.com/All-Hands-AI/OpenHands)
- [OpenHands Runtime architecture](https://docs.openhands.dev/openhands/usage/architecture/runtime)
- [SWE-agent repository](https://github.com/SWE-agent/SWE-agent)
- [SWE-agent paper](https://arxiv.org/abs/2405.15793)
- [DeepSeek Harness repository](https://github.com/deepseek-ai/deepseek-harness)
- [Cordis tutorial](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-tutorial/index.md)
