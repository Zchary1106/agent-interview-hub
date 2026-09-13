# Agent 高频题证据模型：从“题库”升级为可追溯信号

> 更新时间：2026-09-11。结构化数据位于 [`data/question_signals.json`](../data/question_signals.json)。

## 一、为什么不能只维护题目列表

一道题是否值得优先复习，不能只看它是否出现过。至少要同时回答：在哪些公司出现、处于哪轮、由多少独立来源支持、最近何时出现、来源是一手复盘还是汇总文章。

因此，本项目把“题目”与“证据”分开：

```text
Question Signal
  ├─ source_ids → data/interviews.json
  ├─ companies / rounds
  ├─ source_count / last_seen_at
  ├─ source_types / confidence
  ├─ answer_outline
  └─ evaluation_points
```

## 二、字段说明

| 字段 | 含义 | 使用原则 |
|---|---|---|
| `companies` | 出现过该问题的公司 | 不能由一家公司外推整个行业 |
| `rounds` | 一面、二面、算法面、系统设计等 | 汇总帖无法确认时不强行填写 |
| `source_ids` | 支撑该题的来源记录 | 必须指向 `data/interviews.json` 中真实存在的 ID |
| `source_count` | 去重后的来源数量 | 转载不能重复计数 |
| `last_seen_at` | 最近一次公开出现日期 | 面试日期优先于发帖日期 |
| `source_types` | 一手、汇总或参考资料 | `reference` 不计为真实面试 |
| `confidence` | high / medium / low | 由来源完整度决定，不由答案质量决定 |
| `answer_outline` | 回答骨架 | 不把唯一实现包装成标准答案 |
| `evaluation_points` | 面试官可能关注的判断点 | 用于自测和评分 Rubric |

## 三、当前高频信号

当前数据中优先级较高的主题包括：

1. Agent 离线评测、线上指标与失败样本回流；
2. Context、Session、Working State 与 Memory 分层；
3. Tool、Function Calling、MCP 与 Skill 的边界；
4. Agent、Orchestrator、Harness 与 Runtime 的边界；
5. RAG 失败分层诊断；
6. 为什么需要 Multi-Agent；
7. 工具超时后的幂等重试；
8. AI Coding 代码质量验证；
9. Agent 链路的延迟和成本优化；
10. 多步工具轨迹的 Reward 与信用分配。

## 四、来源去重规则

### 4.1 同一事件

满足以下大部分条件时，应放入同一个 `duplicate_group_id`：

- 公司和业务线一致；
- 面试日期一致；
- 面试轮次一致；
- 题目顺序高度相似；
- 项目背景和结果一致。

### 4.2 汇总文章

汇总文章可以提升“趋势置信度”，但不能把文章中的每道题都当作独立一手样本。正确关系是：

```text
Aggregation Source
  ├─ Original Interview A
  ├─ Original Interview B
  └─ Unresolved Claim C
```

找不到原帖的条目保留为 `aggregation`，置信度不得自动升级为 `high`。

## 五、排序建议

推荐分数不是简单累计：

```text
priority =
  0.30 × recency
+ 0.25 × independent_source_count
+ 0.20 × company_coverage
+ 0.15 × source_quality
+ 0.10 × role_relevance
```

需要额外施加：

- 重复来源惩罚；
- 营销内容惩罚；
- 无日期内容惩罚；
- 已过时框架 API 惩罚；
- 与目标岗位无关的题目降权。

## 六、页面应该支持的筛选

- 最近 30 / 90 / 180 天；
- 公司；
- 岗位类型；
- 一面 / 二面 / 系统设计 / 算法面；
- 一手来源；
- 来源数量；
- 置信度；
- 技术主题。

## 七、维护流程

1. 新来源进入 `data/interviews.json`；
2. 判断是否为同一场面试的转载；
3. 抽取问题信号，不复制原文；
4. 关联已有 Question Signal；
5. 更新来源数、公司、轮次和最近日期；
6. 人工确认置信度；
7. 运行 `python3 scripts/validate_data.py`；
8. 构建站点并检查筛选结果。

## 八、边界

频次只能说明“公开样本中出现较多”，不能证明：

- 公司统一使用这套题；
- 某题一定会再次出现；
- 没有公开记录的题不重要；
- 汇总文章中的每条记录都真实独立。

面试准备应使用频次确定顺序，再结合目标岗位 JD、个人项目和基础能力调整。
