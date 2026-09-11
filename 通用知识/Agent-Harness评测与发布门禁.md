# Agent Harness 评测与发布门禁

## 一、Harness 不只是跑 Benchmark 的脚本

Agent Harness 的职责是把任务、环境、执行、轨迹、判分和发布决策连接成可重复实验：

```text
Task Spec + Environment Spec
          ↓
       Agent Runner
          ↓
 Trace + Final State + Artifacts
          ↓
 Rule/Test/LLM/Human Graders
          ↓
 Regression Report
          ↓
 Release Gate
```

## 二、三类 Spec

### Task Spec

```yaml
id: customer-refund-001
objective: 核验订单并在符合规则时发起退款
initial_state: fixtures/refund/order-001.json
allowed_tools: [get_order, read_policy, request_approval, refund]
timeout_seconds: 180
cost_budget_usd: 0.5
required_artifacts: [decision.json, trace.json]
```

### Environment Spec

必须固定：

- 工具与 Schema 版本；
- 数据快照；
- 时间和随机种子；
- 网络是否开放；
- 用户身份和权限；
- 沙箱资源；
- 故障注入规则。

### Expected Outcome

不能只写一段参考答案，应同时定义：

- 最终业务状态；
- 必须产生的 Artifact；
- 禁止发生的副作用；
- 允许的中间路径；
- 成本和延迟上限；
- 是否必须人工审批。

## 三、三层 Grading

### Outcome Grading

检查最终结果：文件是否生成、订单是否更新、测试是否通过、回答是否有正确证据。

### Trajectory Grading

检查过程：是否调用正确工具、是否重复尝试、是否跳过审批、是否使用不必要的高成本模型。

### State Grading

检查环境最终状态：数据库、文件系统、Git、远端资源是否处于期望状态，以及是否留下未完成副作用。

一个任务可能 Outcome 正确但 Trajectory 不安全，也可能答案看起来正确但 State 没有真正改变。

## 四、Grader 组合

| Grader | 适合 | 风险 |
|---|---|---|
| 规则 | Schema、权限、次数、格式 | 难判断语义质量 |
| 单元/E2E 测试 | 代码和确定性状态 | 覆盖不到主观质量 |
| 环境检查 | 数据库、文件、浏览器状态 | 搭建成本较高 |
| LLM-as-Judge | 开放式文本和解释 | 偏差、漂移、自洽性问题 |
| 人工评审 | 高风险和模糊任务 | 成本高、速度慢 |

推荐：规则和测试先判硬约束，LLM Judge 只判开放部分，人审抽检并校准 Judge。

## 五、评测集分层

```text
Smoke：少量关键路径，分钟级
Regression：历史失败和核心功能
Safety：注入、越权、泄露、危险操作
Recovery：超时、断网、重启、重复消费
Canary：接近真实流量的小比例验证
```

数据集需要版本化，并记录：来源、创建原因、适用版本、是否可能进入训练数据。

## 六、Benchmark 选择矩阵

| 场景 | 可参考 Benchmark | 重点能力 | 不能单独证明 |
|---|---|---|---|
| Coding Agent | SWE-bench Verified、SWE-Lancer | 修改真实仓库、测试和交付 | 企业权限、长期运行 |
| Web Agent | WebArena、BrowserGym | 浏览器导航与网页操作 | OS 级操作和真实账号安全 |
| Computer-use | OSWorld | 跨应用桌面操作 | 企业业务正确性 |
| 工具与用户交互 | τ-bench、AppWorld | 工具调用、状态变化、多轮一致性 | 自有业务规则 |
| 通用 Agent | GAIA、AgentBench | 综合推理和工具使用 | 特定场景 SLA |
| 安全 | AgentDojo、内部红队集 | 间接注入、越权、数据泄露 | 全部生产攻击 |
| 科研 Agent | PaperBench | 长程研究和复现 | 普通业务 Agent |

选择 Benchmark 时要问：

1. 环境是否可重复；
2. 最终状态是否能自动验证；
3. 是否测多次稳定性而非单次成功；
4. 是否容易通过特殊提示或数据污染刷分；
5. 与真实岗位任务的距离有多大。

## 七、失败分类

建议将失败统一映射到以下类别：

| 类别 | 示例 |
|---|---|
| 任务理解 | 错解目标、遗漏约束 |
| 规划 | 拆分错误、依赖顺序错误 |
| 上下文 | 关键事实丢失、压缩失真 |
| 检索 | 召回缺失、证据冲突 |
| 工具 | 选错工具、参数错误、超时 |
| 协作 | Agent 角色重叠、消息遗漏、冲突未解决 |
| Runtime | 重复消费、恢复错误、状态竞争 |
| 验证 | 假完成、测试不足、Judge 误判 |
| 安全 | 越权、注入、敏感信息泄露 |

多 Agent 系统应额外记录失败传播链：哪个 Agent 首先产生错误，哪个环节没有拦截，最终由谁把错误转成外部副作用。

## 八、发布门禁

建议配置示例：

```yaml
release_gate:
  verified_success_rate: ">= 0.90"
  first_pass_success_rate: ">= baseline - 0.02"
  critical_safety_violations: 0
  recovery_success_rate: ">= 0.95"
  p95_latency: "<= baseline * 1.15"
  cost_per_success: "<= baseline * 1.10"
```

不能只看平均总分。关键安全场景应使用“一票否决”。

## 九、Shadow 与 Canary

### Shadow Mode

新版本接收生产请求副本，但不产生真实副作用。适合比较路由、计划和工具选择。

注意：即使不执行工具，把真实数据发送给新模型也可能构成数据泄露，因此仍需权限和脱敏。

### Canary

只让小比例真实任务进入新版本，并设置：

- 用户和场景白名单；
- 更低预算；
- 强制人工审批；
- 自动回滚阈值；
- 与旧版本并行统计。

## 十、失败样本回流

```text
线上失败
→ 归因到输入/规划/上下文/检索/工具/模型/验证/Runtime
→ 最小化成可复现 Task Spec
→ 加入 Regression 或 Safety Eval
→ 修复
→ 新旧版本对比
→ 通过门禁后发布
```

不能把完整生产数据无审查地写回训练集或长期记忆。

## 十一、Harness 平台最小数据模型

```text
Dataset → Task → Run → Step/Span → Artifact
                         ↓
                       Grade
                         ↓
                  Evaluation Report
                         ↓
                    Release Decision
```

每次 Run 应固定：代码提交、模型版本、Prompt 版本、工具版本、数据快照和环境镜像。

## 十二、高频面试题

1. Harness、Benchmark、Eval 和 AgentOps 有什么区别？
2. Outcome、Trajectory、State Grading 为什么要分开？
3. LLM-as-Judge 为什么不能单独作为发布门禁？
4. 如何保证一次评测可复现？
5. 失败样本如何回流但不污染评测集？
6. Shadow Mode 如何避免副作用？
7. Canary 阶段哪些指标应一票否决？
8. 如何评估需要人工审批的任务？
9. 多 Agent Harness 如何判定责任和失败传播？
10. 模型升级后为什么必须重新跑安全与恢复评测？
