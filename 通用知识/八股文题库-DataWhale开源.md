# AI Agent 面试 - 八股文题库（DataWhale 开源）

> 来源：github.com/datawhalechina/hello-agents 面试问题总结
> 作者投递岗位：大模型算法工程师、Agent工程师、AI开发工程师、算法评测工程师
> 所有问题均来自**真实线上技术面试**

---

## 一、LLM 基础（必考）

1. Transformer 自注意力机制如何工作？为什么比 RNN 更适合长序列？
2. 位置编码是什么？为什么必需？列举至少两种实现方式
3. 详细介绍 ROPE，对比绝对位置编码优劣势
4. MHA、MQA、GQA 的区别
5. Encoder-Only / Decoder-Only / Encoder-Decoder 各擅长什么任务？
6. Scaling Laws 揭示了什么？对研发有什么指导意义？
7. 推理阶段解码策略：Greedy / Beam / Top-K / Nucleus 原理与优缺点
8. 词元化（Tokenization）：BPE vs WordPiece 比较
9. NLP 和 LLM 最大的区别？
10. "涌现能力"如何理解？
11. LLM 常用激活函数有哪些？为什么选用？
12. MoE 如何不增加推理成本扩大参数？
13. 训练百/千亿参数 LLM 面临哪些挑战？

## 二、VLM 多模态（高频新方向）

14. VLM 核心挑战：不同模态信息如何对齐融合？
15. CLIP 模型工作原理
16. LLaVA / MiniGPT-4 如何连接视觉编码器和 LLM？
17. 视觉指令微调为什么是关键步骤？
18. 处理视频时 VLM 需要额外解决什么？
19. Grounding 在 VLM 中的含义
20. 高分辨率输入图像带来什么挑战？
21. VLM 的幻觉问题与纯文本 LLM 有何不同？

## 三、RLHF / 对齐技术（深水区）

22. RLHF 三个核心阶段详解
23. 成对比较数据 vs 绝对打分，各自优劣？
24. 奖励模型架构如何选择？损失函数背后的数学原理？
25. 为什么选 PPO 而不是 REINFORCE？KL 惩罚项的作用？
26. KL 系数 β 过大/过小分别什么问题？
27. 什么是 Reward Hacking？举例 + 缓解策略
28. DPO 核心思想？与 PPO 的区别和优势
29. DeepSeek 的 GRPO 与 PPO 的区别？
30. GSPO 和 DAPO 与 GRPO 的区别？
31. Token 级别 vs Seq 级别奖励的不同？
32. RLAIF 的理解、潜力和风险

## 四、Agent 核心

33. 如何定义基于 LLM 的 Agent？核心组件？
34. ReAct 框架详解
35. 规划能力的主流方法：CoT / ToT / GoT
36. Memory 设计：短期 + 长期
37. Tool Use / Function Calling 原理
38. LangChain vs LlamaIndex 核心区别
39. 构建复杂 Agent 的最主要挑战？
40. 多智能体系统的优势和复杂性
41. A2A 框架与普通 Agent 框架的区别
42. Agent 框架选型：用过哪些？怎么选？评价指标？
43. 微调过 Agent 能力吗？数据集如何收集？

## 五、RAG

44. RAG 工作原理？与微调相比解决什么问题？
45. 完整 RAG 流水线描述
46. 文本切块策略和权衡
47. Embedding 模型选择和评估指标
48. 提升检索质量的技术
49. "Lost in the Middle" 问题及缓解
50. RAG 系统性能评估：检索 + 生成两阶段
51. 图数据库/知识图谱 vs 向量数据库
52. 复杂 RAG 范式：多次检索、自适应检索
53. RAG 部署中的挑战

## 六、评估

54. BLEU/ROUGE 对 LLM 的局限性
55. 综合基准：MMLU / Big-Bench / HumanEval
56. LLM-as-a-Judge 的优点和偏见
57. 如何评估事实性/推理/安全性？
58. 评估 Agent 为什么比评估 LLM 更难？
59. Agent 评估基准测试有哪些？
60. Agent 过程指标：效率、成本、鲁棒性
61. 红队测试的角色

## 七、开放性问题（必问）

62. 当前 LLM 距离 AGI 还有多远？
63. 开源 vs 闭源模型生态的未来？
64. Transformer 会被 Mamba/SSM 取代吗？
65. Agent 领域最大瓶颈是什么？
66. 最近半年印象最深的 Agent 论文/项目？
67. 未来 1-2 年 Agent 最可能在哪个行业落地？
68. 如果让你自由探索，你想创造什么 Agent？
69. 顶尖 AI Agent 工程师应具备哪些核心素质？

---

## 🔗 原始链接
- https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra01-面试问题总结.md
