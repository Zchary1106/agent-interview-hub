# AI 模型与 Agent 手撕代码题

> 与普通 LeetCode 不同，这类题同时考察数学理解、接口设计、异常处理和可运行性。建议先独立完成，再看参考实现。

## 题目 1：稳定 Softmax

**要求**：实现一维 Softmax，避免大数溢出。

```python
import math


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]
```

**追问**：为什么减去最大值不改变结果？如何扩展到二维 Batch？

## 题目 2：交叉熵

```python
import math


def cross_entropy(probabilities: list[float], target_index: int) -> float:
    if not 0 <= target_index < len(probabilities):
        raise IndexError("target_index out of range")
    probability = max(probabilities[target_index], 1e-12)
    return -math.log(probability)
```

**追问**：如何直接从 Logits 计算，避免先算 Softmax？Label Smoothing 如何修改公式？

## 题目 3：Scaled Dot-Product Attention

```python
import math


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def attention(query, keys, values):
    dimension = len(query)
    scores = [dot(query, key) / math.sqrt(dimension) for key in keys]
    weights = softmax(scores)
    output = [0.0] * len(values[0])
    for weight, value in zip(weights, values):
        for index, number in enumerate(value):
            output[index] += weight * number
    return output, weights
```

**追问**：Mask 加在哪里？为什么除以 `sqrt(d_k)`？Multi-Head 如何拆分和合并？

## 题目 4：LoRA Linear

```python
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.scale = alpha / rank

    def forward(self, inputs):
        delta = (inputs @ self.a.T) @ self.b.T
        return self.base(inputs) + self.scale * delta
```

**追问**：为什么 B 通常初始化为零？推理时如何 Merge？Rank 越大一定越好吗？

## 题目 5：余弦检索

```python
import math


def cosine(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def top_k(query, documents, k=3):
    scored = [(cosine(query, vector), doc_id) for doc_id, vector in documents]
    return sorted(scored, reverse=True)[:k]
```

**追问**：为什么大规模场景不用全量扫描？HNSW 的召回率和延迟怎么权衡？

## 题目 6：RRF 融合排序

```python
from collections import defaultdict


def reciprocal_rank_fusion(rankings, constant=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (constant + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

**追问**：为什么 RRF 不直接使用不同检索器的原始分数？`constant` 如何影响头部结果？

## 题目 7：Tool 参数校验

```python
TOOLS = {
    "get_weather": {
        "required": {"city"},
        "allowed": {"city", "date"},
        "risk": "read"
    },
    "send_email": {
        "required": {"to", "subject", "body"},
        "allowed": {"to", "subject", "body"},
        "risk": "write"
    }
}


def validate_tool_call(name: str, arguments: dict) -> list[str]:
    spec = TOOLS.get(name)
    if spec is None:
        return ["unknown tool"]
    errors = []
    missing = spec["required"] - arguments.keys()
    extra = arguments.keys() - spec["allowed"]
    if missing:
        errors.append(f"missing: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected: {sorted(extra)}")
    if spec["risk"] == "write":
        errors.append("approval required")
    return errors
```

**追问**：格式合法但业务参数越权怎么办？Schema 版本升级如何兼容运行中的任务？

## 题目 8：带循环守卫的 ReAct Agent

```python
import json


def action_fingerprint(action: dict) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def run_agent(model, tools, objective, max_steps=12, repeat_limit=2):
    messages = [{"role": "user", "content": objective}]
    counts = {}

    for step in range(max_steps):
        decision = model(messages, list(tools))
        if decision["type"] == "final":
            return {"status": "completed", "answer": decision["answer"], "steps": step}

        fingerprint = action_fingerprint(decision)
        counts[fingerprint] = counts.get(fingerprint, 0) + 1
        if counts[fingerprint] > repeat_limit:
            return {"status": "blocked", "reason": "repeated_action", "action": decision}

        tool = tools.get(decision["tool"])
        if tool is None:
            observation = {"ok": False, "error": "unknown_tool"}
        else:
            try:
                observation = {"ok": True, "result": tool(**decision.get("arguments", {}))}
            except Exception as exc:
                observation = {"ok": False, "error": type(exc).__name__}

        messages.append({"role": "assistant", "content": decision})
        messages.append({"role": "tool", "content": observation})

    return {"status": "blocked", "reason": "step_budget_exhausted"}
```

**追问**：如何检测 A→B→A→B 的路径震荡？怎样保存 Checkpoint？工具有副作用时如何恢复？

## 题目 9：幂等工具执行器

```python
class IdempotentExecutor:
    def __init__(self):
        self.results = {}

    def execute(self, call_id, function, **kwargs):
        if call_id in self.results:
            return self.results[call_id]
        result = function(**kwargs)
        self.results[call_id] = result
        return result
```

**追问**：进程在外部操作成功后、写入 `results` 前崩溃怎么办？为什么真正系统需要业务端支持幂等？

## 题目 10：流式 SSE 解析

```python

def parse_sse(lines):
    event = {"event": "message", "data": []}
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            if event["data"]:
                yield {"event": event["event"], "data": "\n".join(event["data"])}
            event = {"event": "message", "data": []}
        elif line.startswith("event:"):
            event["event"] = line[6:].strip()
        elif line.startswith("data:"):
            event["data"].append(line[5:].lstrip())
```

**追问**：断线重连如何避免漏事件？`Last-Event-ID` 如何使用？SSE 与 WebSocket 如何选择？

## 题目 11：Agent Trace 聚合

输入一组 Span，统计：

- 总延迟；
- 模型、检索、工具耗时；
- Token 和成本；
- 重试次数；
- 最慢步骤；
- 相同错误指纹。

要求输出结构化报告，并处理 Span 缺失、重复和乱序。

## 题目 12：多 Agent 文件冲突检测

给定多个 Agent 的修改清单：

```json
[
  {"agent": "backend", "files": ["api.py", "schema.py"]},
  {"agent": "test", "files": ["test_api.py", "schema.py"]}
]
```

输出：

- 文件所有者；
- 冲突文件；
- 可以并行的任务组；
- 建议的合并顺序。

**追问**：文件不冲突但接口语义冲突怎么办？为什么需要 Task Contract 和集成测试？

## 评分方式

每题按四层评分：

1. **正确性**：核心结果是否正确；
2. **边界**：空输入、异常、数值稳定性；
3. **工程性**：类型、测试、错误信息和复杂度；
4. **追问**：是否能解释取舍和生产风险。
