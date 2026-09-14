# 自进化 Agent 与 Context Evolution

## 一、自进化不等于允许 Agent 随意修改自己

安全的自进化是把任务经验变成受控知识资产：

```text
Task → Trace → Reflection → Candidate
                           ├─ Memory
                           ├─ Skill
                           ├─ Policy
                           └─ Eval Case
                                ↓
                    Review / Version / Rollback
```

系统改变的不是模型权重本身，而可能是上下文、记忆、技能、路由策略和评测集。

## 二、经验应该写到哪里

| 经验 | 适合载体 | 示例 |
|---|---|---|
| 用户或项目事实 | Memory | 测试命令、输出格式偏好 |
| 可复用操作流程 | Skill | 代码审查、数据分析流程 |
| 安全约束 | Policy | 删除前确认、禁止读取密钥 |
| 失败案例 | Eval Case | 工具超时、恶意网页注入 |
| 临时任务状态 | Checkpoint | 当前步骤和已产生 Artifact |

不要把失败样本、临时状态和用户偏好全部写成同一类 Memory。

## 三、Candidate 到正式资产

每条候选经验至少包含：

```yaml
source_run_id: run-123
source_span_ids: [span-7, span-8]
type: skill
scope: repository
confidence: 0.78
created_by: reflector-v2
contains_sensitive_data: false
status: proposed
```

进入正式 Skill 前需要：

1. 去重；
2. 冲突检测；
3. 隐私与注入检查；
4. 适用范围判断；
5. 在回归任务上验证；
6. 人工或策略审批；
7. 分配版本并支持回滚。

## 四、Curator 的职责

Curator 是知识维护者，不是另一个无限权限 Agent。它负责：

- 合并重复条目；
- 降低过期内容权重；
- 发现相互冲突的规则；
- 将重复流程提升为 Skill；
- 将失败转成 Eval Case；
- 归档低质量或长期未使用资产；
- 输出 Diff 和理由。

Curator 默认不应：

- 删除不可恢复的数据；
- 扩大自身权限；
- 修改内置安全规则；
- 根据单次成功重写核心 Skill；
- 直接把外部网页内容写入长期知识。

## 五、Skill 生命周期

```text
Draft → Proposed → Evaluating → Approved → Active
                                  ↓
                           Deprecated → Archived
```

每个 Skill 应记录：

- Owner；
- 适用范围；
- 输入输出；
- 依赖工具；
- 权限和风险；
- 示例和测试；
- 版本与变更记录；
- 成功率、使用次数和失败模式；
- 回滚目标。

## 六、如何避免错误经验固化

主要风险：

- 把模型猜测当事实；
- 一次偶然成功形成错误规则；
- 恶意网页通过 Prompt Injection 写入 Skill；
- 旧经验与新环境不兼容；
- 只记录成功，不记录代价和失败。

防护：

- 来源追踪；
- 多次独立证据；
- Scope 限制；
- 置信度；
- 沙箱评测；
- 人工批准；
- 灰度启用；
- 自动回滚；
- TTL 和定期复审。

## 七、Context Evolution

上下文演化不是无限追加文字，而是对条目执行：

```text
Add / Update / Merge / Split / Deprecate / Archive
```

### 7.1 三个角色

```text
Generator：从 Trace 生成候选经验
Reflector：分析成败、反事实和适用边界
Curator：去重、冲突处理、验证、发布和归档
```

三个角色可以由同一模型分阶段完成，但权限必须分离：Generator 不直接发布，Reflector 不应修改原始证据，Curator 只能发布通过评测的候选。

### 7.2 六类操作

| 操作 | 适用场景 | 必须保留的证据 |
|---|---|---|
| Add | 出现全新稳定事实或流程 | 来源、Scope、首次验证 |
| Update | 原条目仍有效但内容变化 | Before/After、变化原因 |
| Merge | 多条高度重复 | 原条目 ID、去重判据 |
| Split | 一条规则适用范围过宽 | 新 Scope、冲突样本 |
| Deprecate | 不再推荐但历史仍需追溯 | 替代项、弃用日期 |
| Archive | 长期不用或已失效 | 最终版本、恢复方法 |

删除应是例外。审计和回滚通常要求先 Deprecate/Archive。

### 7.3 正负证据

只记录成功会产生幸存者偏差。候选条目需要同时关联：

- Positive evidence：在哪些任务、环境、模型版本上成功；
- Negative evidence：在哪些边界条件下失败；
- Counterexample：能推翻过度泛化规则的样本；
- Confidence：由独立样本数、时效和来源质量共同决定；
- Provenance：谁、何时、从哪条 Trace 生成和批准。

评测不能只看上下文是否更短，还要看：

- 任务成功率；
- 关键事实保留率；
- 冲突率；
- 过期信息引用率；
- Token 成本；
- 对新任务的迁移效果。

过度摘要可能产生 `context collapse`：文本越来越短，但关键约束逐步消失。

另一种风险是 `brevity bias`：Curator 偏爱短、通用、容易评分的规则，把少见但关键的异常条件删掉。可用“关键约束保留测试 + 长尾失败集 + 新旧上下文 A/B”检测。

### 7.4 防投毒

外部网页、工具输出和用户上传文档默认都属于不可信数据。演化管线应：

1. 将指令与引用数据分隔；
2. 清除凭证、PII 和不可持久化内容；
3. 检查候选是否试图扩大权限或绕过 Policy；
4. 对来源做允许列表和签名/哈希记录；
5. 在隔离环境跑回归与对抗样本；
6. 由独立策略或人工批准高影响变更；
7. Canary 到少量任务并保留自动回滚。

### 7.5 Trace 到资产的闭环

```text
Trace
 → Failure Attribution / Success Pattern
 → Reflection
 → Candidate Diff
 ├─ Memory：事实与偏好
 ├─ Skill：可复用流程
 ├─ Policy：受治理的约束提案
 └─ Eval Case：失败和边界样本
 → Offline Eval
 → Review
 → Canary
 → Active / Rollback
```

Policy 只能形成“变更提案”，不能由普通任务 Agent 自动放宽。

### 7.6 评价一次 Evolution

至少比较旧版本与新版本：

```text
任务成功率、稳定成功率、关键事实保留率
冲突率、过期引用率、安全违规数
Token/延迟/成本、人工接管率、迁移收益
```

如果只提升训练任务、损害未见任务，属于过拟合而不是进化。每次发布都要保存 Context/Skill 版本、评测集版本和回滚指针。

## 八、后台复盘任务

后台 Reflection 应使用：

- 只读或受限工具；
- 独立预算；
- 最大运行时间；
- 可取消任务；
- 输出候选变更而非直接发布；
- 完整 Trace；
- 与前台用户任务隔离的 Worker。

## 九、Agentic RL 的连接点

高质量轨迹可以用于：

- SFT 工具调用样本；
- 偏好对；
- Outcome Reward；
- Process Reward；
- Tool Selection Reward；
- 失败分类器；
- Curriculum 任务生成。

但训练集、评测集和线上回流数据必须隔离，防止评测污染。

## 十、高频面试题

1. 自进化 Agent 与长期记忆 Agent 有什么区别？
2. 一次成功任务什么时候可以沉淀成 Skill？
3. Memory、Skill、Policy 和 Eval Case 如何区分？
4. Curator 为什么必须限制权限？
5. 如何防止 Prompt Injection 污染长期知识？
6. Skill 如何做版本、灰度和回滚？
7. 如何评价一次 Context Evolution 是否有效？
8. 用户如何查看和删除 Agent 学到的个人信息？
9. 线上轨迹如何用于训练但不污染评测？
10. Agent 能否修改自身安全 Policy？为什么？
11. Generator、Reflector、Curator 为什么需要权限分离？
12. Context Evolution 的 Merge 与 Split 如何决定？
13. 如何用正负证据避免一次偶然成功被固化？
14. 什么是 context collapse 与 brevity bias？
15. 如何把一次线上失败转成 Memory、Skill、Policy 或 Eval Case？

## 十一、延伸资料

- [ACE: Agentic Context Engineering](https://arxiv.org/abs/2510.04618)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Voyager: An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291)
- [OpenAI Evals design guide](https://platform.openai.com/docs/guides/evals)
