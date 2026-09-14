# Agent Benchmark 与失败分类

> 更新日期：2026-09-11。Benchmark 分数回答的是“某个固定 Agent-模型-环境组合在某套任务上表现如何”，不能单独证明企业生产可用性。

## 一、先问 Benchmark 测了什么

任何榜单都要拆成六部分：

```text
Task × Environment × Agent/Harness × Model × Grader × Budget
```

- Task：任务分布是否接近真实工作？
- Environment：网页、OS、代码仓库、API 是否可复现？
- Agent/Harness：工具、Prompt、上下文和重试策略是什么？
- Model：版本是否固定，是否存在不可见更新？
- Grader：看最终状态、轨迹、测试、人工评分还是模型裁判？
- Budget：Token、时间、步骤、重试次数是否一致？

不控制 Harness 和预算，模型分数通常不可直接横比。

## 二、主流 Benchmark 地图

| Benchmark | 环境/任务 | 主要判定 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| SWE-bench Verified | 真实 GitHub issue + 仓库 | Patch 是否通过 fail-to-pass / pass-to-pass tests | 修复软件问题的端到端能力 | 新需求设计、长期维护、生产安全 |
| SWE-Lancer | 真实自由职业软件任务 | 任务级自动/人工评分与经济价值 | 更贴近真实委托的工程执行 | 企业私有仓、持续协作和上线责任 |
| WebArena / BrowserGym | 可复现网站与浏览器交互 | 最终网页状态、任务成功 | Web 导航、表单与跨页面操作 | 真实站点漂移、账号权限和高风险交易 |
| OSWorld | 桌面 OS 与应用 | 基于状态的成功检查 | GUI 感知和跨应用操作 | 长期个性化、真实企业设备治理 |
| τ-bench | 用户、工具和数据库状态的多轮交互 | Policy 合规 + 数据库最终状态；强调多次运行一致性 | 多轮对话、工具调用与规则遵循 | 开放世界研究或任意代码执行 |
| AppWorld | 多 App API 世界 | 程序化检查最终状态 | 组合 API、长链工具任务 | UI 感知、真实第三方 API 漂移 |
| GAIA | 现实世界问答，常需搜索/工具 | 最终答案 | 通用助理的检索与推理 | 企业写操作、安全与长期任务 |
| AgentBench | 多类交互环境 | 环境任务成功 | 跨环境 Agent 基线 | 单一生产场景的深度可靠性 |
| AgentDojo | 带工具的动态任务和 Prompt Injection | 任务效用与安全结果 | 间接注入下的效用-安全权衡 | 完整供应链、内部人和基础设施攻击 |
| PaperBench | 复现 AI 研究论文 | 分层 rubric 与独立评分 | 长周期研究工程和实验复现 | 普通业务 Agent 的成本收益 |

### 一个重要指标：稳定成功

单次成功率会掩盖随机性。若单次成功概率为 `p`，要求连续 `k` 次都成功，其理想化概率为：

```text
pass^k = p^k
```

例如 `p=0.8`，连续 4 次都成功只有 `0.4096`。企业自动化更关心“重复执行是否稳定”，而不只是“最好的一次能否成功”。真实评测应直接重复 rollout，而不是只用独立同分布假设计算。

## 三、统一失败分类

### L1：任务理解失败

- 漏掉约束、交付物或隐含验收标准；
- 把“分析”误当“修改”，把“创建 PR”误当“提交 commit”；
- 未识别用户身份、资源范围和风险等级。

**证据**：第一版计划已偏离；最终产物缺少明确要求。

### L2：规划与控制失败

- 步骤顺序错误；
- 无停止条件、预算或取消点；
- 反复试错、路径震荡、过早宣告完成。

**指标**：重复动作率、无效步骤率、首次正确计划率、超预算率。

### L3：上下文与记忆失败

- 关键约束被截断或压缩丢失；
- 使用过期状态；
- 错误记忆污染后续决策；
- 多来源冲突未显式解决。

**指标**：关键事实保留率、过期引用率、冲突检测率。

### L4：检索与证据失败

- 没搜到需要的资料；
- 召回了不相关或低可信来源；
- 有证据但回答无法追溯；
- 训练/评测数据泄漏造成虚高。

**指标**：evidence recall、citation precision、污染审计结果。

### L5：工具失败

分成四类，不要统称“Tool Error”：

1. Selection：选错工具；
2. Argument：Schema 对但业务参数错；
3. Execution：超时、限流、依赖或网络失败；
4. Interpretation：工具成功但模型误读结果。

还要区分可重试、不可重试、需要补偿和需要人工介入。

### L6：协作失败

- 错误委派、重复劳动、信息未传递；
- Agent 间角色冲突；
- 局部成功却破坏全局目标；
- 无法确定失败由谁引入。

可用 MAST 式思路把多 Agent 失败归到任务规范、Agent 间失配和验证不足，而不是简单怪罪“某个模型不够聪明”。

### L7：Runtime 与环境失败

- 沙箱不可用、依赖漂移、文件状态错误；
- 恢复后重复执行副作用；
- GUI/网页元素变化；
- 数据、时钟或外部服务非确定性。

**指标**：环境启动成功率、恢复成功率、副作用重复率、不可复现率。

### L8：验证失败

- 只检查模型文本，不检查环境最终状态；
- 测试过拟合或 grader 有漏洞；
- 忽略回归、安全、成本和用户体验；
- 测试通过但 Artifact 没有交付。

**原则**：Outcome、Trajectory、State 三层判定互补，最终状态优先于模型自述。

### L9：安全与治理失败

- Prompt Injection、越权工具、数据泄漏；
- 高风险动作未审批；
- 审计缺失；
- 评测 Agent 使用了生产凭证或污染真实数据。

安全失败即使任务完成，也不能算成功。

## 四、多 Agent 失败传播

```text
错误检索
 → Researcher 生成错误事实
 → Planner 把事实固化为方案
 → Executor 正确执行错误方案
 → Reviewer 只查格式未查事实
 → 全局失败
```

### 责任归因记录

每个关键决定至少记录：

```yaml
decision_id: d-42
actor: planner-1
inputs: [evidence-7, requirement-3]
output: plan-step-5
assumptions: ["API supports idempotency key"]
validator: reviewer-2
verdict: accepted
```

归因时区分：

- **origin**：错误最初产生在哪里；
- **propagation**：哪些节点未阻止错误；
- **detection**：哪一步发现；
- **impact**：最终影响什么；
- **recovery**：从哪个 checkpoint 恢复最便宜。

## 五、从公开 Benchmark 转为企业 Eval

### 第一步：抽象能力，不照搬题目

例如从 τ-bench 提取：多轮澄清、Policy 遵循、工具写入和最终状态检查；从 SWE-bench 提取：环境准备、Patch、回归测试和 Artifact 验证。

### 第二步：建立 Task/Environment/Outcome Spec

```yaml
task:
  goal: 为客户退款
  constraints: [只能退最近30天订单, 超过500元需审批]
environment:
  fixtures: [customer.json, orders.db]
  tools: [lookup_order, request_approval, issue_refund]
outcome:
  required_state: refund.status == "completed"
  forbidden_state: approval.required && !approval.granted
  budget: {turns: 12, tool_calls: 20, cost_usd: 1.0}
```

### 第三步：覆盖失败与恢复

- 正常任务；
- 工具超时、限流和部分成功；
- 恶意网页/文档注入；
- 用户中途改需求；
- Worker 重启与 checkpoint 恢复；
- 人工拒绝审批；
- 多 Agent 交接丢失关键信息。

### 第四步：形成发布门禁

```text
离线回归 → Safety → Shadow → Canary → 受控写入 → 全量
```

门禁至少看：成功率、稳定成功、严重安全失败数、P95 延迟、成本、恢复成功率和人工接管率。

## 六、面试答题模板

> 我不会先问排行榜第一是谁，而会先固定 Task、Environment、Harness、Model、Grader 和 Budget。公开 Benchmark 用于确认通用能力，企业 Eval 用真实流程的脱敏 fixture 与状态判定。失败按理解、规划、上下文、检索、工具、协作、Runtime、验证和安全分类，并保存轨迹做责任归因。上线前再经过回归、安全、Shadow 和 Canary 门禁。

## 七、高频面试题

1. SWE-bench 分高为什么不代表能直接做企业研发？
2. pass@k 与 pass^k 分别适合什么场景？
3. 如何区分模型失败、Harness 失败和环境失败？
4. 为什么最终状态 Grader 通常比 LLM-as-a-Judge 更可靠？
5. 多 Agent 系统如何做责任归因？
6. 如何防止 Agent 通过“投机”绕过 Grader？
7. Benchmark 环境漂移时如何保持可比性？
8. 如何把线上 bad case 转为回归集且避免泄漏隐私？
9. 为什么安全失败不能被平均成功率抵消？
10. 设计一个 Coding Agent 的企业发布门禁。

## 八、参考资料（官方/论文）

- [SWE-bench](https://www.swebench.com/)
- [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
- [SWE-Lancer](https://openai.com/index/swe-lancer/)
- [WebArena repository](https://github.com/web-arena-x/webarena)
- [BrowserGym](https://github.com/ServiceNow/BrowserGym)
- [OSWorld](https://os-world.github.io/)
- [τ-bench](https://arxiv.org/abs/2406.12045)
- [AppWorld paper](https://aclanthology.org/2024.acl-long.850/)
- [GAIA](https://arxiv.org/abs/2311.12983)
- [AgentBench](https://arxiv.org/abs/2308.03688)
- [AgentDojo](https://arxiv.org/abs/2406.13352)
- [PaperBench](https://openai.com/index/paperbench/)
- [MAST: Multi-Agent System Failure Taxonomy](https://github.com/multi-agent-systems-failure-taxonomy/MAST)
