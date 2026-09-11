# 2026 AI Agent 工程化新考点：Runtime、可靠性与规模化执行

> 更新时间：2026-09-11
> 定位：补充本仓库已有 RAG、MCP、Context Engineering、Agent Harness 内容中尚未系统展开的生产级考点。
> 使用方式：先读概念和设计框架，再用文末问题进行项目追问演练。

## 一、为什么这些内容成为新考点

早期 Agent 面试更关注 ReAct、RAG、Function Calling 和框架 API。2026 年的大厂面试开始把重点放到一个更难的问题：

> 当 Agent 需要运行几十分钟甚至数小时、调用多个有副作用的工具、并发修改共享资源时，怎样保证它可恢复、可审计、可评估，而且不会重复执行危险操作？

因此，面试官不只问“Agent 能不能做”，还会继续追问：

- 为什么需要独立的 Runtime 或 Harness？
- 模型、编排器、状态存储和工具执行器分别负责什么？
- Agent 中断后从哪里恢复？恢复后会不会重复发邮件、扣款或提交代码？
- 如何发现路径震荡、无效重试和工具调用死循环？
- 当工具数量从 10 个增长到 1000 个时，如何控制上下文和选错工具的概率？
- 如何证明系统真的变好，而不是 Demo 看起来更流畅？

---

## 二、Agent、Orchestrator、Harness 与 Runtime 的边界

### 2.1 四个概念不要混用

| 概念 | 核心职责 | 不应该负责 |
|---|---|---|
| Agent | 根据目标和上下文做决策 | 直接承担持久化、资源隔离和审计 |
| Orchestrator | 拆分任务、分派执行单元、处理依赖关系 | 替模型完成所有语义决策 |
| Harness | 包装模型与工具循环，提供审批、上下文、工具和执行约束 | 替代业务工作流引擎 |
| Runtime | 管理任务生命周期、状态、队列、恢复、资源和事件 | 把所有业务规则硬编码进基础设施 |

可以把它们理解为：

```text
用户目标
  ↓
Orchestrator：任务图、依赖、分派
  ↓
Agent / Harness：推理、工具选择、上下文与策略
  ↓
Runtime：调度、状态、超时、重试、恢复、资源隔离
  ↓
Tool Executor：真正产生外部副作用
```

### 2.2 为什么不能只靠 LangGraph 或一个 while 循环

一个简单循环可以完成短任务：

```python
while not done:
    action = model(messages, tools)
    observation = execute(action)
    messages.append(observation)
```

但生产环境还必须解决：

1. 进程重启后任务状态是否还在；
2. 工具调用超时时，究竟是“没有执行”还是“执行成功但响应丢失”；
3. 同一个任务被两个 Worker 消费时如何去重；
4. 人工审批等待数小时后如何继续；
5. 模型版本、Prompt、工具 Schema 变化后如何回放；
6. 子 Agent 部分成功、部分失败时如何补偿。

因此，框架解决“怎样表达流程”，Runtime 解决“怎样可靠地运行流程”。

---

## 三、Durable Execution：可恢复的长任务执行

### 3.1 最小持久化状态

不要只保存聊天记录。一个可恢复任务至少需要：

```json
{
  "task_id": "task-123",
  "run_id": "run-7",
  "status": "WAITING_APPROVAL",
  "current_step": "submit_payment",
  "context_version": 12,
  "plan_version": 3,
  "tool_calls": [],
  "artifacts": [],
  "checkpoint_id": "cp-42",
  "lease_owner": "worker-5",
  "lease_expires_at": "2026-09-11T10:30:00Z"
}
```

关键点：

- `task_id` 表示用户任务，`run_id` 表示一次执行尝试；
- `checkpoint_id` 必须对应一个可重放的稳定边界；
- `lease` 防止多个 Worker 同时推进同一任务；
- Artifact 应外置保存，不要全部塞进消息历史；
- 所有状态转换都应产生事件，而不是只覆盖最终状态。

### 3.2 Checkpoint 不等于保存 messages

合格的 Checkpoint 应同时记录：

- 当前状态机节点；
- 已完成步骤及其输出摘要；
- 外部 Artifact 的引用和版本；
- 已执行工具调用的幂等键；
- 下一步允许执行的动作；
- 当时使用的模型、Prompt、工具 Schema 版本。

只保存 messages 会遇到两个问题：

1. 无法确定哪些副作用已经发生；
2. 恢复时只能让模型“猜”接下来该做什么。

### 3.3 Exactly-once 通常做不到

分布式系统中很难保证工具调用真正 Exactly-once。更实际的设计是：

```text
At-least-once delivery
        +
Idempotency key
        +
Result journal
        +
Compensation
```

例如支付工具：

1. Runtime 生成 `tool_call_id`；
2. 调用支付服务时同时发送该 ID；
3. 支付服务保存 ID 和执行结果；
4. 超时重试时返回第一次结果；
5. 如果操作不能幂等，则提供撤销或补偿动作。

### 3.4 推荐状态机

```text
PENDING
  → RUNNING
  → WAITING_TOOL
  → WAITING_APPROVAL
  → RUNNING
  → VERIFYING
  → COMPLETED

异常分支：
RUNNING → RETRYABLE_FAILED → RUNNING
RUNNING → BLOCKED
RUNNING → COMPENSATING → FAILED
```

面试时应强调：`FAILED`、`BLOCKED` 和 `WAITING_APPROVAL` 是三种不同状态，不能统统记成失败。

---

## 四、路径震荡、死循环与无效重试

### 4.1 什么是路径震荡

路径震荡指 Agent 在两个或多个策略之间反复切换，却没有产生新的有效信息。例如：

```text
搜索代码 → 尝试修改 → 测试失败
→ 回退修改 → 再次搜索相同文件
→ 重复之前的修改 → 再次失败
```

它与普通重试的区别是：

- 普通重试通常因为瞬时错误；
- 路径震荡是决策层没有吸收失败证据。

### 4.2 检测方法

可以组合使用：

- 相同工具与参数的重复率；
- 连续状态摘要的相似度；
- Artifact 是否发生有效变化；
- 相同错误指纹是否重复出现；
- 单位 Token 带来的进展；
- 计划节点是否在有限集合内来回跳转。

一个简单的震荡分数：

```text
oscillation_score =
  0.35 × repeated_action_ratio
  + 0.25 × state_similarity
  + 0.20 × repeated_error_ratio
  + 0.20 × no_progress_ratio
```

超过阈值后，不应只是继续增加 `max_iterations`，而应触发：

1. 强制总结已有证据；
2. 禁止重复最近动作；
3. 请求替代计划；
4. 切换模型或子 Agent；
5. 必要时转人工处理。

### 4.3 失败预算

| 预算 | 示例 |
|---|---|
| 总迭代次数 | 最多 30 轮 |
| 相同动作次数 | 同工具同参数最多 2 次 |
| 连续无进展次数 | 最多 3 次 |
| Token | 200k |
| 金额 | 5 美元 |
| 墙钟时间 | 45 分钟 |
| 高风险工具调用 | 最多 1 次且必须审批 |

预算耗尽后应生成失败报告和恢复建议，而不是只返回“达到最大轮次”。

---

## 五、从 10 个工具扩展到 1000 个工具

### 5.1 为什么不能把所有工具定义都放进 Prompt

工具数量增长会带来：

- Prompt Token 成本快速增加；
- 相似工具之间更容易误选；
- Schema 变更污染缓存；
- 模型注意力被无关工具分散；
- 权限控制难以解释。

### 5.2 分层工具路由

推荐四层结构：

```text
用户意图
  ↓
领域路由：代码 / 数据 / CRM / 财务
  ↓
候选工具检索：关键词 + Embedding + 权限过滤
  ↓
Schema 精排：参数匹配、风险、成本、历史成功率
  ↓
模型从 Top-K 中选择并调用
```

注意顺序：**先做权限过滤，再把工具暴露给模型**。不能让模型看到无权调用的工具，然后依赖 Prompt 要求它“不要使用”。

### 5.3 Skill 的渐进式披露

Skill 不应等于一段超长 Prompt。可以拆成三层：

1. **Manifest**：名称、用途、适用条件、风险等级；
2. **Instructions**：被选中后才加载详细步骤；
3. **Resources**：示例、脚本、模板按需加载。

这既减少上下文，也能让版本、权限和审核边界更清楚。

### 5.4 工具 Schema 版本化

至少记录：

- `tool_name`
- `schema_version`
- `risk_level`
- `required_scopes`
- `timeout`
- `retry_policy`
- `idempotency_support`

工具升级时要支持旧任务恢复，不能假设所有运行中的任务都会立刻使用新 Schema。

---

## 六、Agent 可观测性：从日志升级为 Trace

### 6.1 一条 Trace 应该包含什么

```text
Task
 ├─ Run
 │   ├─ Model Span
 │   ├─ Retrieval Span
 │   ├─ Tool Span
 │   ├─ Approval Span
 │   └─ Verification Span
 └─ Artifacts / Events / Checkpoints
```

每个 Span 至少记录：

- 输入输出摘要和哈希；
- 模型、Prompt、工具版本；
- Token、延迟、成本；
- 重试次数和错误类型；
- 决策原因或路由标签；
- 是否产生副作用；
- 关联的 Artifact 和 Checkpoint。

敏感输入不应原样进入日志，应进行脱敏、分级存储和访问审计。

### 6.2 线上排障顺序

当任务失败时，按层定位：

1. **输入层**：目标是否清晰、权限是否充足；
2. **规划层**：任务拆分是否错误；
3. **上下文层**：关键信息是否被截断或污染；
4. **检索层**：是否召回错误、过期或相互冲突的资料；
5. **工具层**：参数、权限、超时、Schema 是否正确；
6. **模型层**：是否理解错误或违反输出约束；
7. **验证层**：成功条件是否太弱，导致假完成；
8. **Runtime 层**：是否发生重复消费、恢复错误或状态竞争。

---

## 七、2026 年应掌握的 Agent 评测指标

### 7.1 任务结果

- Task Success Rate：最终完成率；
- First-pass Success Rate：不重试一次完成率；
- Verified Success Rate：经过独立检查后确认的完成率；
- Partial Completion Rate：部分交付比例。

### 7.2 过程质量

- 平均计划修改次数；
- 工具选择准确率；
- 无效工具调用率；
- 路径震荡率；
- 人工接管率；
- Recovery Success Rate。

### 7.3 成本和性能

- 单任务 Token 和金额；
- P50/P95 完成时间；
- 模型调用与工具调用占比；
- 上下文压缩率；
- 缓存命中率。

### 7.4 安全

- 未授权工具调用数；
- Prompt Injection 攻击成功率；
- 敏感数据泄露率；
- 高风险操作审批覆盖率；
- 沙箱逃逸或策略违规数。

### 7.5 评测集构建

评测集至少包含：

- 正常路径；
- 缺少必要信息；
- 工具超时和部分失败；
- 工具返回冲突数据；
- 上下文超长；
- Prompt Injection；
- 需要人工审批；
- 中途重启和恢复；
- 同一任务并发执行。

要保留失败案例，不要只用成功 Demo 构建评测集。

---

## 八、Coding Agent 的并发修改冲突

### 8.1 常见冲突

- 两个 Agent 同时修改同一文件；
- 一个 Agent 修改接口，另一个仍按旧接口实现；
- 测试 Agent 在代码尚未稳定时读取中间状态；
- 多个 Agent 共享同一个未提交工作区；
- 主 Agent 合并结果时丢失子 Agent 的约束和失败证据。

### 8.2 推荐隔离策略

```text
Task Contract
  ↓
为每个子任务创建独立工作区或 worktree
  ↓
限制允许修改的路径
  ↓
每个子任务产出补丁 + 验证证据
  ↓
由集成者按依赖顺序合并
  ↓
重新运行全量验证
```

关键原则：

- 共享 Artifact，不共享可变工作区；
- 子 Agent 返回结构化结果，而不只是自然语言“完成了”；
- 合并成功不等于功能验证成功；
- 冲突解决必须保留原始需求和测试证据。

---

## 九、近期大厂面试新增信号

> 下列内容来自公开面经的题目归纳，只作为备考线索，不代表公司固定题库。

### 字节跳动：Harness、长任务与 Coding Agent

近期公开面经新增关注：

- Harness 和 Orchestrator 的边界；
- Agent 路径震荡和失败归因；
- 上下文超过窗口后的分层压缩；
- 大量工具下的检索和路由；
- 多 Agent 同时修改代码的冲突；
- Coding Agent 的单测、覆盖率与质量门禁；
- 高并发 Agent 服务的容量和降级设计。

准备时不能只背框架 API，应能画出任务、Run、Checkpoint、Tool Call 和 Artifact 的数据关系。

### 阿里 / 淘天：MCP 评测、Trace 与 Agent Infra

近期公开题目新增关注：

- 一个 MCP Server 开发完成后怎样评测；
- Agent 链路追踪和对话历史如何保存；
- LangGraph State、Memory、Checkpoint 的边界；
- Max Tokens 中断后如何续跑；
- 评测集如何产生并防止数据污染；
- Skill 如何渐进式加载；
- Agent 应用与推理服务的延迟、KV Cache 和批处理。

准备时要把“应用层 Agent”和“模型推理基础设施”串起来回答。

### 腾讯：失败恢复、幂等和金融安全

近期公开题目重点包括：

- Agent 中断后如何恢复；
- 工具超时后怎样判断是否已经成功执行；
- 重试如何避免重复扣款或重复提交；
- 多 Agent 风控流程如何设置权限和人工审批；
- RAG 知识库如何不停机更新；
- 如何设计可回滚、可补偿的执行链。

这类题目的核心不是“多试几次”，而是副作用、审计和责任边界。

### 美团：框架取舍、评测闭环和沙箱

近期公开题目重点包括：

- 什么场景不应该使用 LangGraph；
- Workflow 与自主 Agent 的边界；
- 为什么自研，而不是直接使用现成 Coding Agent；
- 怎样构建离线评测集；
- 如何采集 Trace 并定位 bad case；
- 工具执行为什么需要沙箱；
- 如何同时优化完成率、延迟和 Token 成本。

准备时应给出基线、失败案例、改动和指标，而不是只讲系统架构图。

---

## 十、面试回答模板

回答生产级 Agent 系统设计题时，可以使用下面的顺序：

```text
1. 目标：Agent 要完成什么，成功条件是什么
2. 约束：时延、成本、安全、权限、并发、数据边界
3. 架构：Agent / Orchestrator / Runtime / Tool / State
4. 状态：Task、Run、Event、Checkpoint、Artifact
5. 可靠性：超时、重试、幂等、补偿、恢复
6. 安全：权限、审批、沙箱、审计
7. 评测：离线集、线上指标、失败案例
8. 演进：先做单 Agent 基线，再证明是否需要多 Agent
```

避免只回答“我们用了 LangGraph、MCP 和向量数据库”。面试官真正想知道的是：为什么这样选、失败时发生什么、怎样证明它有效。

---

## 十一、高频自测题

1. Agent、Orchestrator、Harness 和 Runtime 有什么区别？
2. 为什么 Checkpoint 不能只保存消息历史？
3. 工具调用超时后，怎样判断是否可以安全重试？
4. 为什么 Exactly-once 很难？Agent 系统通常怎样近似实现？
5. 如何发现 Agent 正在路径震荡而不是正常探索？
6. `max_iterations` 为什么不能彻底解决死循环？
7. 1000 个工具为什么不能全部放进 Prompt？
8. 工具检索为什么必须先做权限过滤？
9. Skill 的 Manifest、Instructions 和 Resources 怎样按需加载？
10. 工具 Schema 升级后，旧任务怎样恢复？
11. 一条 Agent Trace 应包含哪些 Span？
12. Task Success 和 Verified Success 有什么区别？
13. 怎样构建包含恢复、越权和注入攻击的评测集？
14. 多个 Coding Agent 修改同一仓库时怎样避免冲突？
15. 为什么应当先证明单 Agent 的不足，再引入多 Agent？

---

## 十二、公开面经来源

- [字节跳动 Agent 开发近期面经汇总](https://www.nowcoder.com/discuss/922659050167226368)
- [阿里 Agent 开发近期面经汇总](https://www.nowcoder.com/discuss/923740641878646784)
- [腾讯 / 百度大模型与 Agent 面经总结](https://www.nowcoder.com/discuss/878600528970735616)
- [美团 Agent 开发面经与工程化追问](https://www.nowcoder.com/discuss/881209147377664000)

> 来源分级建议：带明确轮次、日期和个人项目过程的一手复盘优先级最高；汇总型文章只用于发现题目趋势；带课程销售和“标准满分答案”的内容不可作为唯一证据。

## 关联阅读

- [Agent Harness 与编码代理深度测评](Agent%20Harness与编码代理测评.md)
- [Context Engineering 上下文工程](Context%20Engineering上下文工程.md)
- [Agent 安全与评估体系](Agent安全与评估体系.md)
- [MCP 与工具生态](MCP与工具生态.md)
- [最新 AI Agent / 大模型面经索引](最新AI-Agent面经索引.md)
