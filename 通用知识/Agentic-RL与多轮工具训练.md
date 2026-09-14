# Agentic RL 与多轮工具训练

> 更新日期：2026-09-11。本章是算法、后训练和高级 Agent 岗的选修方向。应用工程岗位应先掌握可验证 Agent Runtime 与 Eval，再深入训练。

## 一、为什么单轮 Tool Calling 不够

真实 Agent 轨迹通常是：

```text
用户目标
 → 思考/计划
 → 工具调用
 → 环境 observation
 → 修正计划
 → 多次工具调用
 → 最终答案或环境状态
```

单轮 SFT 只学习“看到输入后生成一个正确调用”，无法完整覆盖：

- 工具结果改变后续决策；
- 长期信用分配；
- 失败恢复和策略切换；
- 预算、停止与延迟奖励；
- 多次调用之间的依赖。

## 二、把 Agent 任务建模为 MDP/POMDP

- 状态 `s_t`：环境真实状态，模型通常不能完全观察；
- 观察 `o_t`：工具结果、页面、日志、用户消息；
- 动作 `a_t`：文本、工具调用、终止或请求人工介入；
- 转移 `P(s_{t+1}|s_t,a_t)`：环境执行动作后的变化；
- 奖励 `r_t`：过程奖励、工具奖励、成本惩罚或最终结果；
- 策略 `π(a_t|h_t)`：基于历史 `h_t` 选择动作。

由于 Agent 看不到完整数据库、网页或操作系统状态，实际更接近 POMDP。Context Builder 生成的是 belief/context，不等于真实状态。

## 三、Trajectory 数据模型

```json
{
  "task_id": "refund-42",
  "environment_version": "crm-fixture-v3",
  "policy_version": "agent-7",
  "steps": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_call": {"name": "lookup_order", "args": {}}},
    {"role": "tool", "result": {}, "state_hash": "abc"},
    {"role": "assistant", "content": "..."}
  ],
  "outcome": {"success": true, "grader_version": "g5"},
  "cost": {"tokens": 4200, "tool_calls": 3, "latency_ms": 8200},
  "safety": {"violations": []},
  "provenance": {"source": "synthetic", "consent": "n/a"}
}
```

必须固定环境、工具 Schema、Agent 和 Grader 版本，否则轨迹难以复现。

## 四、三种 Loss Mask

### 4.1 Token-level mask

只在 Assistant 生成的 token 上计算语言模型损失，不训练复制用户输入和工具 observation。

### 4.2 Turn-level mask

某些 turn 是环境产生、人工接管或无效重试，可整轮屏蔽；也可以只训练被验证为正确的工具调用轮次。

### 4.3 Trajectory-level mask/weight

按整条轨迹的结果、安全性和质量加权。失败轨迹并非都应丢弃：它可以用于训练错误识别、恢复策略或偏好对，但不能当作成功示范。

**面试陷阱**：只保留成功轨迹会产生选择偏差，模型看不到“如何从失败恢复”。

## 五、奖励设计

```text
R = outcome
  + process
  + tool_correctness
  + recovery
  - cost
  - latency
  - safety_violation
```

### Outcome Reward

最终测试、数据库状态、任务答案或人工 rubric。最接近业务目标，但通常稀疏。

### Process Reward

对关键中间步骤评分，例如是否先读取约束、是否验证 Patch。优点是密集，风险是把某条固定路径误当唯一正确路径。

### Tool Reward

工具选择、参数、调用时机和结果解释分别计分，避免“调用成功”掩盖业务参数错误。

### Safety Reward/Constraint

高风险违规不能只是很小的负奖励，否则模型可能用更高成功收益抵消。实践中常需要硬约束、Constrained RL 或直接终止轨迹。

## 六、稀疏奖励与信用分配

长轨迹只在最后给 0/1，难以知道是哪一步造成结果。常见方法：

- outcome decomposition：把结果拆成可验证子目标；
- reward-to-go / advantage：估计某步对后续收益的贡献；
- process reward model：评价中间步骤；
- counterfactual replay：替换某一步后重跑；
- step-level verifier：检查工具参数、状态变化和约束；
- failure attribution：标记错误起源和未拦截节点。

奖励模型必须独立验证，否则 Agent 可能学会讨好评分器。

## 七、Reward Hacking

常见投机方式：

- 修改测试或 Grader，而不是修复代码；
- 输出看起来正确的文本但不改变环境；
- 重复调用直到偶然成功；
- 利用状态检查漏洞；
- 隐藏失败日志或提前终止；
- 通过 Prompt Injection 影响 LLM Judge。

防护：不可变测试、隐藏测试、环境状态判定、轨迹审计、独立 Safety Grader、调用预算和对抗样本。

## 八、Rollout 基础设施

```text
Task Queue
  ↓
Rollout Workers ── Model/Policy Server
  ↓ actions              ↑ weight sync
Versioned Environments
  ↓ trajectories
Replay / Dataset Store
  ↓
Reward + Safety Graders
  ↓
Trainer → Checkpoint Registry → Eval Gate
```

### 关键工程问题

1. **异步 Rollout**：提高吞吐，但数据可能由旧策略生成，形成 off-policy 偏差。
2. **权重同步**：记录每条轨迹对应的 policy checkpoint，不能只记模型名称。
3. **环境复位**：每条任务结束后恢复干净快照，避免状态串扰。
4. **确定性**：固定 fixture、依赖、时钟和随机种子；外网任务要记录快照。
5. **故障隔离**：Worker 超时或崩溃不能污染下一条任务。
6. **数据治理**：凭证、PII、版权内容和客户数据需要脱敏、授权与保留策略。

## 九、训练-推理一致性

训练时与上线时必须尽量对齐：

| 项目 | 不一致风险 |
|---|---|
| Tool Schema | 学会调用已删除参数 |
| System Prompt | 策略依赖训练期提示技巧 |
| Context Compaction | 训练见完整历史，上线只见摘要 |
| Sandbox | 训练权限过大，线上动作失败 |
| Model Sampling | rollout 与生产随机性不同 |
| Error Format | 模型无法识别线上超时/限流 |
| Stop Condition | 训练允许更多步骤，线上过早停止 |

因此要把 Prompt、Tool、Environment、Context Policy 一起版本化；只发布模型权重不等于发布 Agent。

## 十、Eval/Train 污染

- 公开 Benchmark 解答进入训练集；
- 线上回流中混入评测任务；
- 同一仓库相邻 issue 跨集合泄漏；
- LLM 生成合成任务时复述原题；
- 人工筛选者看到隐藏测试。

防护：时间切分、仓库/用户级去重、近重复检测、Canary 题、隔离权限、数据 lineage 和定期污染审计。

## 十一、推荐训练路径

```text
高质量 SFT
 → 轨迹拒绝采样 / Best-of-N
 → 偏好优化
 → 可验证奖励 RL
 → 多轮工具与恢复课程
 → 安全约束和线上 Shadow
```

先确保环境和 Grader 可信，再扩大 RL。错误的奖励规模越大，训练出的投机策略越稳定。

### Curriculum 示例

1. 单工具、确定性读取；
2. 多工具但无副作用；
3. 需要澄清的多轮任务；
4. 有写操作和审批；
5. 注入超时、部分失败和状态漂移；
6. 长任务、Checkpoint 和人工接管；
7. 对抗性 Prompt Injection 与越权请求。

## 十二、与 Agent Benchmark 的连接

- τ-bench 类任务适合多轮 Policy + Tool + State 奖励；
- SWE-bench 类任务可用测试作为可验证 outcome reward；
- Web/OS 环境需要可靠状态快照与复位；
- 安全 Benchmark 应作为约束门禁，不能与效用简单平均；
- pass^k/stability 能揭示策略偶然成功和 rollout 方差。

## 十三、高频面试题

1. 为什么多轮工具训练更接近 POMDP？
2. Token、Turn、Trajectory 三层 mask 有什么区别？
3. Outcome Reward 与 Process Reward 各有什么风险？
4. 如何为一次错误退款做信用分配？
5. 异步 Rollout 为什么会产生策略陈旧问题？
6. 如何防止 Agent 修改测试来骗取奖励？
7. 为什么训练和推理的 Context Compaction 必须一致？
8. 失败轨迹应该删除还是保留？
9. 如何避免公开 Benchmark 污染？
10. 为什么“模型 checkpoint”不足以完整复现 Agent？

## 十四、面试回答模板

> 我会先把任务建模为部分可观测的多轮决策过程，并把环境、工具、上下文策略、模型和 Grader 全部版本化。数据按 token、turn、trajectory 三层做 mask，奖励以可验证 outcome 为主，辅以工具正确性、恢复与成本，但安全违规使用硬约束。基础设施上用可复位环境产生带 policy version 的 rollout，处理异步采样的策略陈旧，并通过隐藏测试、状态判定和污染审计防止 reward hacking。

## 十五、参考资料（论文/项目）

- [Agent Lightning](https://arxiv.org/abs/2508.03680)
- [Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning](https://arxiv.org/abs/2507.05725)
- [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)
- [τ-bench: A Benchmark for Tool-Agent-User Interaction](https://arxiv.org/abs/2406.12045)
- [SWE-bench](https://www.swebench.com/)
- [AgentDojo](https://arxiv.org/abs/2406.13352)
