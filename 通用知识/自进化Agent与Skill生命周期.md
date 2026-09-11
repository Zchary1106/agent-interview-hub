# 自进化 Agent 与 Skill 生命周期

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

评测不能只看上下文是否更短，还要看：

- 任务成功率；
- 关键事实保留率；
- 冲突率；
- 过期信息引用率；
- Token 成本；
- 对新任务的迁移效果。

过度摘要可能产生 `context collapse`：文本越来越短，但关键约束逐步消失。

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
