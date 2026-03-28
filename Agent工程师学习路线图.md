# 🗺️ AI Agent 工程师学习路线图（2025-2026）

> 本路线图面向**国内互联网大厂跳槽**场景，系统梳理从零到拿到 Offer 的完整学习路径。
> 适用于有一定编程基础（Python熟练）、想转型或深入 AI Agent 方向的工程师。

---

## 目标定位

| 维度 | 说明 |
|------|------|
| **目标岗位** | 大模型算法工程师 / AI Agent工程师 / AI应用开发工程师 / LLM应用架构师 |
| **目标公司** | 字节跳动、阿里巴巴、腾讯、百度、小红书、美团、蚂蚁集团、华为、京东、快手、商汤、月之暗面、智谱AI、MiniMax、百川智能 |
| **薪资区间** | 40-80W（base 30-55K × 14-18薪），部分头部公司 Senior 可达 100W+ |
| **城市分布** | 北京（字节/百度/美团/快手）、杭州（阿里/蚂蚁/网易）、深圳（腾讯/华为）、上海（小红书/商汤/MiniMax） |
| **学历要求** | 硕士优先，本科需要有突出项目经验或论文，海外TOP校有加分 |

### 岗位能力模型

```
┌─────────────────────────────────────────────────────────┐
│                  AI Agent 工程师能力模型                    │
├──────────────┬──────────────┬──────────────┬─────────────┤
│   基础理论    │   工程能力    │   系统设计    │  业务理解   │
│  (30%)       │  (30%)       │  (25%)       │  (15%)     │
├──────────────┼──────────────┼──────────────┼─────────────┤
│ Transformer  │ RAG系统      │ Agent架构    │ 场景分析    │
│ LLM原理      │ Agent框架    │ 高可用设计    │ 产品思维    │
│ 训练/微调    │ 模型部署      │ 性能优化     │ 数据飞轮    │
│ 对齐技术     │ 工具链       │ 安全防护      │ ROI评估    │
└──────────────┴──────────────┴──────────────┴─────────────┘
```

---

## 一、学习阶段划分

### 阶段一：基础夯实（2-4周）

> **目标：** 建立扎实的大模型理论基础，能够清晰地解释 Transformer 架构、LLM 训练流程和推理过程。

#### 1.1 Transformer 架构（第1周重点）

**核心知识点：**

**（1）自注意力机制（Self-Attention）**
- 理解 Query、Key、Value 的含义和计算过程
- 注意力分数的计算：`Attention(Q,K,V) = softmax(QK^T / √d_k) V`
- 为什么要除以 √d_k（防止梯度消失/爆炸）
- 注意力矩阵的可视化解读
- 自注意力 vs 交叉注意力的区别

**面试高频问题：**
- 自注意力的计算复杂度是多少？O(n²d)，为什么？
- 如何降低自注意力的计算复杂度？（Flash Attention、稀疏注意力、线性注意力）
- 自注意力和 CNN/RNN 相比有什么优缺点？

**（2）多头注意力（Multi-Head Attention）**
- 为什么需要多头？（不同头关注不同语义子空间）
- 多头注意力的参数量计算
- MHA vs MQA（Multi-Query Attention）vs GQA（Grouped-Query Attention）
  - MQA：所有 head 共享 K、V，推理速度快，Llama2-70B 等使用
  - GQA：将 head 分组共享 K、V，性能和速度的平衡，Llama3 使用

**（3）位置编码（Positional Encoding）**
- 为什么 Transformer 需要位置编码（自注意力本身是排列不变的）
- 绝对位置编码：正弦余弦编码、可学习位置编码
- **RoPE（旋转位置编码）**—— 面试必考：
  - 原理：将位置信息编码到旋转矩阵中
  - 优势：相对位置的内积只依赖相对距离
  - 外推性：NTK-aware RoPE、YaRN 等长度外推方法
  - 实现：在 Q、K 上应用旋转变换
- ALiBi（Attention with Linear Biases）

**（4）FFN（前馈神经网络）**
- 标准 FFN：两层线性变换 + 激活函数
- SwiGLU 激活函数（Llama系列使用）：`SwiGLU(x) = Swish(xW₁) ⊙ xV`
- FFN 的参数量通常占模型总参数的 2/3

**（5）归一化层**
- LayerNorm vs RMSNorm
  - RMSNorm 去掉了均值中心化，计算更快，效果相当
  - Pre-Norm vs Post-Norm：现代 LLM 普遍使用 Pre-Norm（训练更稳定）

**（6）完整的 Transformer 架构**
- Encoder-Decoder（原始论文）vs Decoder-Only（GPT系列）vs Encoder-Only（BERT）
- 为什么 LLM 普遍使用 Decoder-Only？
- KV Cache 的原理和实现

**推荐学习资源：**
| 资源 | 类型 | 说明 |
|------|------|------|
| [Jay Alammar - The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) | 博客 | 最经典的图解，必看 |
| [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 视频 | 从零手写 GPT，2小时深入理解 |
| [3Blue1Brown - Transformer 可视化](https://www.youtube.com/watch?v=wjZofJX0v4M) | 视频 | 数学直觉，看完理解更深 |
| [李沐 - Transformer 论文精读](https://www.bilibili.com/video/BV1pu411o7BE/) | 视频 | 中文精读，逐段讲解 |
| Attention Is All You Need (2017) | 论文 | 原始论文，必读 |

**学习产出：**
- [ ] 手写简化版 Self-Attention（纯 PyTorch）
- [ ] 画出完整的 Transformer Decoder 架构图
- [ ] 写一篇笔记：RoPE 的原理和推导

---

#### 1.2 LLM 基础（第2周重点）

**（1）Tokenization**
- 为什么需要 Tokenization？（将文本转换为模型可处理的数字序列）
- **BPE（Byte Pair Encoding）**—— 面试常考：
  - 训练过程：从字符级开始，迭代合并最高频的相邻对
  - 推理过程：贪心匹配
  - 变体：Byte-level BPE（GPT-2/3/4使用）、SentencePiece（Llama使用）
- WordPiece（BERT使用）vs Unigram（T5使用）
- 词表大小的影响：太小→序列太长，太大→参数量增加
- 中文 Tokenization 的特殊性（Qwen、DeepSeek 的词表设计）

**（2）预训练（Pre-training）**
- 训练目标：Next Token Prediction（因果语言模型）
- 训练数据：Common Crawl、Books、Wikipedia、Code 等
- 数据清洗和去重的重要性
- 训练超参数：学习率调度（cosine decay）、batch size、warmup
- 分布式训练：数据并行（DP/DDP）、模型并行（TP/PP）、ZeRO

**（3）Scaling Laws**
- Chinchilla Scaling Law：最优的模型大小和数据量的关系
  - `L(N,D) ∝ N^(-0.076) + D^(-0.103)`
  - 关键结论：大多数 LLM 都是"欠训练"的
- 涌现能力（Emergent Abilities）：随规模增大突然出现的能力
- 对实际选型的指导意义

**（4）解码策略**
- Greedy Decoding：每步选概率最大的 token
- Beam Search：保留 top-k 个候选序列
- Sampling 策略：
  - Temperature：控制分布的平滑程度
  - Top-k Sampling：只从概率最高的 k 个 token 中采样
  - Top-p (Nucleus) Sampling：从累积概率达到 p 的最小集合中采样
  - Min-p Sampling：过滤掉概率低于 min_p × max_prob 的 token
- Repetition Penalty、Frequency Penalty、Presence Penalty

**（5）主流模型对比**

| 模型 | 公司 | 开源 | 架构特点 | 适用场景 |
|------|------|------|---------|---------|
| GPT-4o/4.1 | OpenAI | ❌ | MoE（传闻）| 综合能力最强 |
| Claude 3.5/4 | Anthropic | ❌ | Constitutional AI | 长文本、代码、安全性 |
| Llama 3.1/4 | Meta | ✅ | GQA + RoPE | 开源标杆，微调基座 |
| Qwen 2.5/3 | 阿里 | ✅ | GQA + YaRN | 中文最强开源 |
| DeepSeek V3/R1 | 深度求索 | ✅ | MoE + MLA | 性价比极高，推理能力强 |
| GLM-4 | 智谱AI | ✅ | 自回归填空 | 中文理解好 |
| Gemini 2.0 | Google | ❌ | 多模态原生 | 多模态能力强 |

**推荐学习资源：**
| 资源 | 说明 |
|------|------|
| [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | Tokenizer 从零实现 |
| [Andrej Karpathy - State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) | LLM 训练全流程综述 |
| [李沐 - GPT/GPT-2/GPT-3 论文精读](https://www.bilibili.com/video/BV1AF411b7xQ/) | 系列论文精读 |
| [Chinchilla 论文](https://arxiv.org/abs/2203.15556) | Scaling Laws 经典 |

**学习产出：**
- [ ] 手写 BPE Tokenizer（训练+推理）
- [ ] 整理主流模型架构对比表（含参数量、训练数据量、上下文长度）
- [ ] 写一篇笔记：解码策略的原理与适用场景

---

#### 1.3 Python + ML 基础（贯穿前2周）

**（1）PyTorch 核心操作**
- Tensor 操作：创建、索引、变形、广播
- 自动微分（autograd）：计算图、梯度计算、`backward()`
- 模型定义：`nn.Module`、`forward()`、参数管理
- 训练循环：DataLoader、Optimizer、Loss、学习率调度器
- GPU 操作：`.cuda()`、`.to(device)`、混合精度训练（AMP）
- 常用操作：`einsum`、`torch.nn.functional`

**（2）HuggingFace Transformers**
- Pipeline API：快速推理
- Model + Tokenizer：加载预训练模型
- Trainer API：标准训练流程
- 模型配置：`AutoConfig`、`AutoModel`、`AutoTokenizer`
- 模型保存和加载：`save_pretrained()`、`from_pretrained()`
- 常用模型：`AutoModelForCausalLM`、`AutoModelForSequenceClassification`

**（3）其他工具**
- NumPy：矩阵运算
- Pandas：数据处理
- Matplotlib/Seaborn：可视化
- Jupyter Notebook：实验环境
- Git：版本控制（面试项目必须放 GitHub）

**学习产出：**
- [ ] 用 PyTorch 实现一个简单的语言模型（bigram 或 mini-GPT）
- [ ] 用 HuggingFace 跑通一个文本分类任务
- [ ] 配置好自己的开发环境（GPU 云服务器 or 本地 GPU）

---

### 阶段二：核心技能（4-6周）

> **目标：** 掌握 AI Agent 工程师的核心技术栈——Prompt Engineering、RAG、Agent 设计模式和主流框架。

#### 2.1 Prompt Engineering → Context Engineering（第3-4周）

**（1）Prompt Engineering 基础技巧**

**Zero-shot Prompting**
- 直接给指令，不提供示例
- 适用于简单任务
- 技巧：明确角色、任务、输出格式

**Few-shot Prompting**
- 提供几个输入-输出示例
- 示例选择的策略：多样性、相关性、边界case
- 示例数量的影响（通常3-5个）

**Chain-of-Thought (CoT)**
- 让模型"逐步思考"
- `"Let's think step by step"` 的魔力
- 变体：Zero-shot CoT、Manual CoT、Auto-CoT
- 适用场景：数学推理、逻辑推理、多步骤问题

**ReAct（Reasoning + Acting）**
- 思考-行动-观察的循环
- 将推理和工具调用结合
- Agent 的基础范式

**其他高级技巧**
- Self-Consistency：多次采样取多数投票
- Tree of Thoughts (ToT)：树状搜索推理路径
- Reflection/Self-Critique：让模型审视自己的输出
- Constitutional AI：通过原则约束输出
- Structured Output：JSON mode、Function Calling

**（2）从 Prompt Engineering 到 Context Engineering**

> 2025年的关键转变：不仅仅是写好 prompt，而是**系统地设计模型接收的全部上下文**。

**Context Engineering 的核心理念：**
- Prompt 只是上下文的一部分
- 完整上下文 = System Prompt + 用户消息 + 检索结果 + 工具输出 + 对话历史 + 结构化数据
- 上下文窗口是有限资源，需要精心管理

**系统设计要素：**
- **上下文组装（Context Assembly）**：从多个来源收集信息
- **上下文压缩（Context Compression）**：摘要、截断、过滤
- **上下文排序（Context Ordering）**：关键信息放在开头和结尾（Lost in the Middle 效应）
- **上下文缓存（Context Caching）**：减少重复计算
- **动态上下文管理**：根据任务阶段调整上下文内容

**面试常考：**
- 如何在有限的上下文窗口中最大化信息密度？
- Lost in the Middle 问题怎么解决？
- System Prompt 的设计原则是什么？

**推荐学习资源：**
| 资源 | 说明 |
|------|------|
| [Anthropic - Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering) | 工业级 prompt 指南 |
| [OpenAI - Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) | 官方最佳实践 |
| [DAIR.AI - Prompt Engineering Guide](https://www.promptingguide.ai/) | 最全面的开源指南 |
| [吴恩达 - ChatGPT Prompt Engineering](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | 免费课程 |

**学习产出：**
- [ ] 整理一份 Prompt Engineering 速查手册
- [ ] 设计一个复杂场景的 System Prompt（如客服Agent、代码审查Agent）
- [ ] 实现一个 Context Engineering 的 demo：动态组装上下文

---

#### 2.2 RAG 完全掌握（第4-5周）

> RAG（Retrieval-Augmented Generation）是面试必考、工作必用的核心技术。

**（1）Naive RAG（基础版本）**

**完整流程：**
```
用户查询 → 查询向量化 → 向量检索 → 获取文档片段 → 组装上下文 → LLM生成答案
```

**文档处理（Indexing）：**
- 文档加载：PDF、Word、HTML、Markdown 等格式解析
- 文本分割策略：
  - 固定大小分割（Fixed-size Chunking）
  - 递归字符分割（RecursiveCharacterTextSplitter）—— 最常用
  - 语义分割（Semantic Chunking）
  - 按文档结构分割（Markdown Header、HTML Section）
- Chunk 大小的选择：通常 256-1024 tokens，需要根据场景调优
- Chunk Overlap：通常 10-20%，防止信息截断

**Embedding 模型：**
- 原理：将文本映射到高维向量空间，语义相似的文本距离近
- 主流模型：
  - OpenAI text-embedding-3-small/large
  - BGE 系列（智源，中文最佳）：bge-large-zh-v1.5、bge-m3
  - GTE 系列（阿里）
  - Jina Embeddings
  - Cohere Embed v3
- 选择要素：维度、最大长度、中英文效果、开源vs闭源
- 评估基准：MTEB、C-MTEB

**向量数据库：**

| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| Chroma | 轻量级，易上手 | 本地开发、原型验证 |
| Milvus | 分布式，高性能 | 生产环境、大规模数据 |
| Pinecone | 托管服务，免运维 | 快速上线 |
| Weaviate | 支持混合检索 | 需要多种检索方式 |
| Qdrant | Rust 实现，高性能 | 性能敏感场景 |
| FAISS | Meta 开源，纯库 | 已有基础设施的场景 |
| Elasticsearch | 传统搜索+向量 | 已有 ES 集群的团队 |

**实操任务：**
- [ ] 用 Chroma 搭建一个本地 RAG 系统
- [ ] 对比不同 Embedding 模型在自己数据上的效果
- [ ] 尝试不同的 chunk 策略并对比检索质量

**（2）Advanced RAG（进阶优化）**

**检索优化：**
- **混合检索（Hybrid Search）**：向量检索 + 关键词检索（BM25）
  - 融合策略：RRF（Reciprocal Rank Fusion）
  - 为什么混合检索通常效果更好（互补性）
- **查询改写（Query Rewriting）**：
  - HyDE（Hypothetical Document Embedding）：先让 LLM 生成假设性答案，再用答案去检索
  - 多查询（Multi-Query）：将原始查询拆分为多个子查询
  - Step-back Prompting：先问一个更宏观的问题
- **重排序（Re-ranking）**：
  - Cross-Encoder Re-ranker：用交叉编码器对检索结果重新排序
  - BGE-Reranker、Cohere Rerank
  - 为什么先召回再重排（精排+效率的平衡）
- **元数据过滤**：结合结构化过滤缩小检索范围

**生成优化：**
- 上下文压缩：LongLLMLingua 等
- 引用溯源：让模型标注信息来源
- 幻觉检测：NLI-based 方法检测答案是否基于上下文
- 多轮对话中的 RAG：对话历史管理、查询带入上下文

**评估框架：**
- RAGAs：自动化评估
  - Faithfulness（忠实度）：答案是否基于上下文
  - Answer Relevance（答案相关性）
  - Context Precision（上下文精确度）
  - Context Recall（上下文召回率）
- 人工评估的标准和流程

**（3）高级 RAG 架构（面试加分项，可在阶段三深入）**
- GraphRAG：基于知识图谱的 RAG
- Agentic RAG：Agent 驱动的自适应检索
- Multi-modal RAG：图文混合检索
- Self-RAG：模型自己判断是否需要检索

**推荐学习资源：**
| 资源 | 说明 |
|------|------|
| [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/) | 官方教程，从基础到进阶 |
| [吴恩达 - Building RAG Agents](https://www.deeplearning.ai/short-courses/) | 免费短课程系列 |
| [RAG 综述论文](https://arxiv.org/abs/2312.10997) | 学术全面综述 |
| [Advanced RAG 技术汇总](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6) | 图解进阶技术 |

**学习产出：**
- [ ] 实现一个完整的 Advanced RAG 系统（含混合检索、重排序）
- [ ] 用 RAGAs 评估系统效果并写对比报告
- [ ] 整理一份 RAG 常见问题及解决方案的文档

---

#### 2.3 Agent 设计模式（第5-6周）

> Agent 是 2025-2026 年最热门的方向，也是面试的核心考点。

**（1）什么是 AI Agent？**
- 定义：能够感知环境、做出决策、执行行动的智能系统
- 与简单 LLM 调用的区别：循环执行、工具调用、状态管理
- Agent 的核心组件：
  - **规划（Planning）**：任务分解、推理
  - **记忆（Memory）**：短期（上下文）、长期（外部存储）
  - **工具（Tools）**：API调用、代码执行、搜索等
  - **行动（Action）**：执行具体操作

**（2）核心设计模式**

**ReAct 模式（Reasoning + Acting）**
```
思考(Thought) → 行动(Action) → 观察(Observation) → 思考 → ... → 最终答案
```
- 将推理和行动交替进行
- 通过观察外部反馈来修正推理
- 实现简单，效果好，是最基础的 Agent 模式
- 局限：容易陷入循环、缺乏全局规划

**Plan-and-Execute 模式**
```
1. 规划阶段：将复杂任务分解为子任务列表
2. 执行阶段：按顺序执行每个子任务
3. 可选：执行过程中根据结果调整计划
```
- 适合复杂的多步骤任务
- 规划和执行解耦，可以用不同的模型
- 变体：Plan-and-Solve、LLM Compiler

**Reflection 模式**
```
生成 → 反思/批评 → 修改 → 再反思 → ... → 满意的输出
```
- 让 Agent 审视和改进自己的输出
- Reflexion：在 episode 间学习
- Self-Refine：在单次生成中迭代
- 适用于代码生成、写作等需要迭代的任务

**Tool Use 模式**
- Function Calling：模型输出结构化的工具调用请求
- 工具定义：参数 schema、描述、示例
- 工具调用的解析和执行
- 错误处理和重试策略
- 工具选择策略：如何在大量工具中选择合适的

**（3）多 Agent 协作模式**

**Supervisor 模式**
```
Supervisor Agent → 分配任务 → Worker Agent 1, 2, 3...
                 → 收集结果 → 汇总输出
```
- 一个主 Agent 协调多个子 Agent
- 适合任务分解明确的场景

**Hierarchical 模式**
- 多层级的 Agent 组织
- 上级 Agent 管理下级 Agent
- 适合大型复杂系统

**Debate/Discussion 模式**
- 多个 Agent 从不同角度讨论
- 通过辩论达成共识
- 适合需要多视角分析的场景

**Swarm 模式**
- 去中心化的 Agent 群体
- 每个 Agent 独立运作，通过共享状态协作
- OpenAI Swarm 框架

**（4）Agent 记忆系统**
- **短期记忆**：对话上下文、工作区状态
- **长期记忆**：
  - 向量数据库存储历史交互
  - 摘要记忆：定期总结对话历史
  - 实体记忆：提取和维护关键实体信息
- **工作记忆（Scratchpad）**：临时推理空间
- 记忆检索和更新策略

**面试高频问题：**
- Agent 和 Chain 的区别是什么？（Agent 有循环和决策能力）
- 如何防止 Agent 陷入无限循环？（最大步数限制、循环检测、超时）
- 多 Agent 系统的通信方式有哪些？（直接消息、共享黑板、事件驱动）
- 如何评估 Agent 的效果？（任务完成率、步骤效率、工具调用准确率）

**推荐学习资源：**
| 资源 | 说明 |
|------|------|
| [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | 经典综述博客 |
| [吴恩达 - AI Agents 课程](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | 实战课程 |
| [Anthropic - Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | 工程最佳实践 |
| ReAct 论文 (2022) | 奠基论文 |
| Reflexion 论文 (2023) | 反思范式 |

**学习产出：**
- [ ] 用纯 Python 实现一个 ReAct Agent（不用框架）
- [ ] 实现一个 Plan-and-Execute Agent
- [ ] 画出 4 种 Agent 设计模式的架构图

---

#### 2.4 框架实战（第6-8周）

**（1）LangChain + LangGraph —— 必须精通**

**LangChain 核心概念：**
- LCEL（LangChain Expression Language）：声明式编排
- 核心组件：
  - Chat Models：统一的模型接口
  - Prompt Templates：模板化提示词
  - Output Parsers：结构化输出解析
  - Tools：工具定义和调用
  - Retrievers：检索器抽象
- Chains：将组件串联成流水线
- Memory：对话记忆管理

**LangGraph —— Agent 编排框架（重点）：**
- 核心概念：
  - **State**：图的状态定义（TypedDict 或 Pydantic）
  - **Node**：处理函数（每个节点执行一个操作）
  - **Edge**：节点间的连接（条件边、普通边）
  - **Graph**：有向图，定义整个工作流
- 关键特性：
  - 循环支持（区别于 DAG）
  - 人工介入（Human-in-the-loop）
  - 持久化状态（Checkpointing）
  - 流式输出
  - 子图（Subgraph）
- 常用模式实现：
  - ReAct Agent
  - Supervisor Multi-Agent
  - Plan-and-Execute
  - Human-in-the-loop 审核流

**实操任务：**
```python
# 用 LangGraph 实现一个简单的 ReAct Agent
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# 1. 定义状态
# 2. 定义工具
# 3. 定义节点（模型调用、工具执行）
# 4. 定义条件边（是否需要调用工具）
# 5. 编译并运行
```

**（2）其他框架了解**

**CrewAI**
- 基于角色的多 Agent 框架
- 定义 Agent（角色、目标、工具）和 Task
- 适合快速搭建多角色协作系统
- 学习成本低，适合demo

**AutoGen（微软）**
- 对话驱动的多 Agent 框架
- 支持人类参与对话
- 适合研究和实验

**Dify**
- 低代码 AI 应用开发平台
- 可视化编排工作流
- 内置 RAG、Agent、工具集成
- 适合快速搭建和部署 AI 应用
- 面试时可以作为"快速验证"能力展示

**MCP（Model Context Protocol）**
- Anthropic 发起的工具集成协议
- 标准化 LLM 与外部工具的通信
- 2025年快速普及，多家厂商支持
- 了解协议设计和使用方式

**A2A（Agent-to-Agent Protocol）**
- Google 发起的 Agent 间通信协议
- Agent 发现、能力声明、任务委托
- 还在早期，但值得关注

**框架选型建议：**
| 场景 | 推荐框架 |
|------|---------|
| 生产级 Agent 系统 | LangGraph |
| 快速原型验证 | Dify / CrewAI |
| 研究/实验 | AutoGen / 纯代码 |
| 多角色协作演示 | CrewAI |
| 工具集成 | MCP + LangChain |

**学习产出：**
- [ ] 用 LangGraph 实现一个完整的多 Agent 系统
- [ ] 用 Dify 搭建一个 RAG 应用并部署
- [ ] 对比 LangGraph 和 CrewAI 的代码实现差异

---

### 阶段三：进阶深入（4-6周）

> **目标：** 掌握模型微调、对齐技术、推理优化等进阶技能，建立技术深度。

#### 3.1 模型微调（第9-10周）

**（1）SFT（Supervised Fine-Tuning）**

**基本概念：**
- 在预训练模型基础上，用标注数据进行有监督微调
- 目标：让模型学会特定任务或遵循特定格式
- 数据格式：指令-输入-输出 三元组

**全参数微调 vs 参数高效微调（PEFT）：**
- 全参数微调：更新所有参数，效果最好但资源消耗大
- PEFT：只更新少量参数，资源消耗小

**（2）LoRA（Low-Rank Adaptation）—— 面试必考**

**原理：**
- 冻结原始权重 W₀
- 添加低秩分解矩阵 ΔW = BA（B ∈ R^{d×r}, A ∈ R^{r×k}，r << min(d,k)）
- 前向传播：h = W₀x + BAx
- 参数量从 d×k 降到 (d+k)×r

**关键超参数：**
- `r`（秩）：通常 8-64，越大表达能力越强
- `alpha`：缩放系数，通常设为 r 的1-2倍
- `target_modules`：选择哪些层加 LoRA（通常 q_proj, v_proj, k_proj, o_proj）
- `dropout`：防过拟合

**QLoRA（Quantized LoRA）：**
- 基模型用 4-bit 量化，LoRA 参数用全精度
- 显存大幅降低（70B 模型可在单卡 A100 上微调）
- 使用 NF4（NormalFloat4）量化 + 双重量化
- 实现：bitsandbytes + PEFT

**其他 PEFT 方法：**
- Prefix Tuning：在每层注意力前添加可训练的前缀向量
- Adapter：在 FFN 后添加小型网络
- IA3：学习缩放向量（参数量更少）
- DoRA：分解权重为方向和大小

**（3）训练数据构建**

**数据来源：**
- 人工标注（质量最高，成本高）
- LLM 生成（Self-Instruct、Evol-Instruct）
- 开源数据集改造

**数据质量 >> 数据数量：**
- LIMA 论文：1000条高质量数据效果超过大量低质量数据
- 数据清洗：去重、过滤低质量、去除有害内容
- 数据多样性：覆盖不同任务类型和难度

**数据格式（以 Alpaca 格式为例）：**
```json
{
  "instruction": "将以下英文翻译为中文",
  "input": "Hello, how are you?",
  "output": "你好，你怎么样？"
}
```

**实操工具：**
- HuggingFace TRL（Transformer Reinforcement Learning）
- Axolotl：简化微调流程
- LLaMA-Factory：一站式微调平台（国产，推荐）
- Unsloth：加速微调

**学习产出：**
- [ ] 用 LoRA 微调 Qwen2.5-7B 做一个垂域问答模型
- [ ] 构建一份高质量的微调数据集（至少500条）
- [ ] 对比全参数微调和 LoRA 的效果差异

---

#### 3.2 RLHF / 对齐技术（第10-11周）

**（1）RLHF（Reinforcement Learning from Human Feedback）**

**完整流程：**
```
SFT模型 → 收集人类偏好数据 → 训练奖励模型(RM) → PPO优化策略模型
```

**奖励模型（Reward Model）：**
- 输入：(prompt, response)对
- 输出：标量分数
- 训练数据：人类对多个回复的排序
- 损失函数：Bradley-Terry 模型，最大化偏好回复和非偏好回复的分数差

**PPO（Proximal Policy Optimization）：**
- 核心思想：限制策略更新的幅度（clip ratio）
- 目标：最大化奖励，同时不偏离 SFT 模型太远（KL 散度约束）
- 实现复杂度高，训练不稳定
- 需要 4 个模型：策略模型、参考模型、奖励模型、价值模型

**（2）DPO（Direct Preference Optimization）—— 面试常考**

**核心思想：**
- 跳过显式的奖励模型训练
- 直接用偏好数据优化策略模型
- 将 RLHF 问题转化为分类问题

**损失函数：**
```
L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```
- y_w：偏好回复（winner）
- y_l：非偏好回复（loser）
- β：温度参数

**优势：**
- 实现简单，不需要训练奖励模型
- 训练稳定，不需要 PPO 的复杂调参
- 计算资源需求低

**（3）GRPO（Group Relative Policy Optimization）**

- DeepSeek 提出的对齐方法
- 核心思想：在一组采样中使用相对排名作为奖励
- 不需要显式的奖励模型
- 适合推理任务的优化
- DeepSeek-R1 的核心训练方法之一

**（4）其他对齐技术**
- RLAIF：用 AI 反馈替代人类反馈
- KTO（Kahneman-Tversky Optimization）：基于前景理论
- IPO（Identity Preference Optimization）
- ORPO：不需要参考模型的对齐方法
- Constitutional AI：基于原则的自我改进

**面试高频问题：**
- DPO 和 RLHF 的区别和优缺点？
- GRPO 的创新点是什么？
- 奖励模型的过度优化（Reward Hacking）怎么解决？
- 对齐税（Alignment Tax）是什么？如何减少？

**学习产出：**
- [ ] 用 DPO 对一个 SFT 模型进行对齐训练
- [ ] 整理 RLHF → DPO → GRPO 的技术演进脉络
- [ ] 写一篇笔记对比各种对齐方法的原理和适用场景

---

#### 3.3 推理优化（第11-12周）

**（1）推理加速**

**KV Cache**
- 原理：缓存之前 token 的 Key、Value，避免重复计算
- 显存占用：随序列长度线性增长
- 优化：MQA/GQA（减少 KV 头数）、PagedAttention

**vLLM —— 面试必知**
- PagedAttention：借鉴 OS 的虚拟内存管理
  - 将 KV Cache 分成固定大小的 block
  - 按需分配，减少显存碎片和浪费
  - 共享前缀的 KV Cache（Prefix Caching）
- Continuous Batching：动态 batch，提高 GPU 利用率
- 吞吐量比 HuggingFace 高 2-24 倍
- 使用：`python -m vllm.entrypoints.openai.api_server --model xxx`

**其他推理框架：**
- TensorRT-LLM（NVIDIA）：深度优化，性能最好
- SGLang：Structured Generation，约束解码加速
- llama.cpp：CPU/Apple Silicon 推理
- Ollama：本地部署利器

**（2）模型量化**

**量化基础：**
- 将高精度浮点数（FP16/BF16）转换为低精度（INT8/INT4）
- 目标：减少显存占用，加速推理，轻微牺牲精度

**量化方法：**
- **PTQ（Post-Training Quantization）**：训练后直接量化
  - GPTQ：基于 Hessian 信息的逐层量化
  - AWQ：基于激活感知的量化（保护重要权重）
  - GGUF：llama.cpp 的量化格式
- **QAT（Quantization-Aware Training）**：训练时模拟量化
- 精度级别：INT8 → INT4 → INT3 → INT2

**（3）部署架构**

**单机部署：**
- vLLM / Ollama / TGI（Text Generation Inference）
- GPU 选型：A100（80GB）、H100、A10G、RTX 4090
- 显存估算：模型参数量 × 2字节（FP16）或 × 1字节（INT8）

**分布式部署：**
- 张量并行（TP）：将模型分到多卡
- 流水线并行（PP）：将不同层分到不同卡
- 数据并行（DP）：多实例处理不同请求

**服务化：**
- OpenAI 兼容 API 接口
- 负载均衡、限流、熔断
- 监控：延迟、吞吐、显存使用率
- 成本优化：Spot 实例、自动扩缩容

**学习产出：**
- [ ] 用 vLLM 部署一个开源模型并压测
- [ ] 对比 INT8 和 INT4 量化模型的效果和速度
- [ ] 画出一个生产级 LLM 服务的部署架构图

---

#### 3.4 高级 RAG（第12-13周）

**（1）GraphRAG（知识图谱 + RAG）**

**微软 GraphRAG：**
- 原理：从文档中提取实体和关系，构建知识图谱
- 索引阶段：
  1. 文档分块
  2. LLM 提取实体和关系
  3. 构建知识图谱
  4. 社区检测（Leiden 算法）
  5. 为每个社区生成摘要
- 查询阶段：
  - Local Search：基于实体的局部搜索
  - Global Search：基于社区摘要的全局搜索
- 优势：擅长回答需要全局理解的问题（如"数据集的主要主题是什么？"）
- 劣势：索引成本高（大量 LLM 调用）

**LightRAG / nano-GraphRAG：**
- 轻量级 GraphRAG 替代方案
- 降低索引成本，保留核心效果

**（2）Agentic RAG**
- RAG 流程由 Agent 驱动
- Agent 决定是否检索、检索什么、如何处理结果
- 自适应检索：根据查询难度选择策略
  - 简单问题 → 直接回答
  - 中等问题 → 单次检索
  - 复杂问题 → 多步检索 + 推理
- 查询路由：将查询分发到不同的数据源
- 工具化检索：将检索作为 Agent 的工具

**（3）多模态 RAG**
- 图文混合检索
- 图像 Embedding：CLIP、SigLIP
- 文档中的图表理解
- 视频检索

**学习产出：**
- [ ] 用微软 GraphRAG 在一个中文语料上构建知识图谱
- [ ] 实现一个 Agentic RAG，对比与普通 RAG 的效果
- [ ] 写一篇 GraphRAG vs Vector RAG 的对比分析

---

#### 3.5 安全与评估（第13-14周）

**（1）Guardrails 设计**

**Prompt Injection 防护：**
- 直接注入：用户在输入中嵌入恶意指令
- 间接注入：通过检索内容注入
- 防护策略：
  - 输入检测和过滤
  - 指令层级隔离
  - 输出验证
  - 使用专门的安全分类模型

**内容安全：**
- 有害内容过滤
- PII（个人身份信息）检测和脱敏
- 版权内容检测
- 幻觉检测和缓解

**框架和工具：**
- Guardrails AI：规则定义和验证
- NeMo Guardrails（NVIDIA）：可编程的安全防护
- LLM Guard：输入输出过滤
- Lakera Guard：Prompt injection 检测

**（2）Agent 评估框架**

**评估维度：**
- 任务完成率（Task Success Rate）
- 步骤效率（Step Efficiency）
- 工具调用准确率（Tool Call Accuracy）
- 安全性（Safety Score）
- 用户满意度

**评估工具：**
- LangSmith：LangChain 的可观测性平台
- AgentBench：多维度 Agent 评估基准
- GAIA：通用 AI 助手评估
- SWE-bench：代码Agent评估

**学习产出：**
- [ ] 为自己的 Agent 项目设计评估方案
- [ ] 实现基本的 Guardrails（输入检测、输出验证）
- [ ] 用 LangSmith 监控和调试一个 Agent 应用

---

### 阶段四：项目实战（4-8周）

> **目标：** 完成 3 个高质量项目，作为面试的核心武器。每个项目都要能在 GitHub 上展示。

#### 项目一：RAG 知识问答系统（2-3周）

**项目描述：**
基于企业文档构建一个智能知识问答系统，支持多格式文档解析、混合检索、多轮对话。

**技术栈：**
- 框架：LangChain + LangGraph
- 向量数据库：Milvus（体现生产级考量）
- Embedding：BGE-M3 或 GTE
- Re-ranker：BGE-Reranker-v2
- LLM：Qwen2.5-72B（API）或 DeepSeek-V3
- 前端：Streamlit / Gradio
- 后端：FastAPI

**核心架构：**
```
用户查询 → 查询理解（改写/分类）
         → 混合检索（向量+BM25+RRF融合）
         → Re-ranking（交叉编码器重排序）
         → 上下文组装（压缩+排序）
         → LLM 生成（带引用标注）
         → 后处理（幻觉检测+格式化）
```

**实现步骤：**

1. **文档处理流水线**
   - 支持 PDF、Word、Markdown、HTML 格式
   - 智能分割：先按结构分割（标题/段落），再按语义分割
   - 元数据提取：标题、来源、时间等
   - 构建索引：向量索引 + BM25 倒排索引

2. **检索系统**
   - 稠密检索：Embedding + Milvus
   - 稀疏检索：BM25
   - 混合检索：RRF 融合
   - Re-ranking：BGE-Reranker 精排
   - 查询改写：Multi-Query + HyDE

3. **生成系统**
   - System Prompt 设计：角色、规则、格式
   - 上下文管理：Lost-in-the-Middle 优化
   - 引用溯源：标注信息来源
   - 幻觉检测：NLI 模型验证

4. **多轮对话**
   - 对话历史管理
   - 指代消解（"这个"指代什么）
   - 查询带上上下文

5. **评估和优化**
   - RAGAs 自动评估
   - 构建测试集（至少 100 个 Q&A pair）
   - 各模块 A/B 测试

**项目亮点（面试时重点讲）：**
- 混合检索 + Re-ranking 使检索 Recall@5 提升 15%
- 使用 semantic chunking 替代固定分割，答案准确率提升 10%
- 幻觉检测模块将无据回答率从 12% 降到 3%
- 支持增量更新，新文档 5 分钟内可检索

---

#### 项目二：多 Agent 协作系统（2-3周）

**项目描述：**
用 LangGraph 实现一个 Supervisor 模式的多 Agent 研究助手，能够自动进行课题研究、信息收集、报告撰写。

**系统架构：**
```
用户需求 → Supervisor Agent（规划/调度）
           ├→ Researcher Agent（搜索/检索信息）
           ├→ Analyst Agent（分析/推理）
           ├→ Writer Agent（撰写报告）
           └→ Reviewer Agent（审核/修改）
```

**技术栈：**
- 核心框架：LangGraph
- LLM：GPT-4o / Claude 3.5 / Qwen-Max
- 工具：Tavily Search（联网搜索）、Python REPL（数据分析）、文件读写
- 持久化：SQLite + LangGraph Checkpointing
- 部署：FastAPI + Redis

**实现步骤：**

1. **定义 Agent 角色**
   ```python
   # Supervisor Agent
   # - 接收用户需求
   # - 制定研究计划（子任务列表）
   # - 分配任务给 Worker Agent
   # - 检查完成状态
   # - 汇总最终结果
   
   # Researcher Agent
   # - 使用搜索工具收集信息
   # - 使用 RAG 检索内部知识
   # - 返回结构化的信息摘要
   
   # Analyst Agent
   # - 对收集的信息进行分析
   # - 使用 Python REPL 进行数据计算
   # - 生成分析结论
   
   # Writer Agent
   # - 将分析结果组织成报告
   # - 遵循指定的报告格式
   
   # Reviewer Agent
   # - 审核报告质量
   # - 提出修改建议
   # - 验证事实准确性
   ```

2. **LangGraph 实现**
   - 定义 State（包含任务列表、每个 Agent 的输出、当前阶段）
   - 定义条件边（Supervisor 根据状态决定下一步）
   - 实现 Human-in-the-loop（关键决策点需要人工确认）
   - 错误处理和重试机制

3. **工具集成**
   - 搜索工具：Tavily / Bing Search
   - 代码执行：安全沙箱中的 Python REPL
   - 文件操作：读写 Markdown/PDF
   - 内部知识库：RAG 检索

4. **状态持久化**
   - LangGraph Checkpointing
   - 支持中断恢复
   - 对话历史保存

**项目亮点：**
- Supervisor 模式的任务分解和动态调度
- Human-in-the-loop 实现关键决策审核
- 支持中断恢复（Checkpointing）
- Agent 间通信通过共享状态实现，避免信息丢失
- 完整的错误处理和重试机制

---

#### 项目三：生产级 Agent 应用（2-3周）

**项目描述：**
构建一个生产级的个人助理 Agent，支持工具调用、长期记忆、安全防护，可以处理日常任务（日程管理、邮件处理、信息查询等）。

**技术栈：**
- 框架：LangGraph + FastAPI
- LLM：多模型路由（简单任务用小模型，复杂任务用大模型）
- 工具：日历 API、邮件 API、搜索、代码执行
- 记忆：Redis（短期）+ PostgreSQL + pgvector（长期）
- 安全：NeMo Guardrails
- 监控：LangSmith + Prometheus
- 部署：Docker + Kubernetes

**核心功能模块：**

1. **智能路由**
   - 意图识别：分类用户请求类型
   - 模型选择：根据任务复杂度选择合适的模型
   - 工具选择：从工具库中选择合适的工具

2. **工具调用系统**
   - MCP 协议集成
   - 工具注册和发现
   - 参数校验和错误处理
   - 调用链追踪

3. **记忆系统**
   - 短期记忆：当前对话上下文（Redis）
   - 长期记忆：历史交互摘要（PostgreSQL + pgvector）
   - 实体记忆：用户偏好、常用信息
   - 记忆检索和衰减策略

4. **安全防护**
   - 输入检测：Prompt Injection 防护
   - 输出过滤：PII 脱敏、有害内容过滤
   - 权限控制：工具调用权限
   - 审计日志：所有操作记录

5. **可观测性**
   - 请求链路追踪
   - 延迟和成本监控
   - 错误告警
   - A/B 测试支持

**项目亮点：**
- 多模型路由，简单任务成本降低 70%
- 完整的安全防护体系（输入检测+输出过滤+权限控制）
- 长期记忆实现个性化服务
- 完善的可观测性和监控
- Docker 化部署，一键启动

---

### 阶段五：面试冲刺（2-4周）

> **目标：** 系统梳理知识体系，大量刷题和模拟面试，确保能拿到 Offer。

#### 5.1 高频题清单

**基础理论题（必背）：**
1. Transformer 的自注意力机制是什么？计算复杂度是多少？
2. RoPE 的原理是什么？相比绝对位置编码有什么优势？
3. MHA、MQA、GQA 的区别？各自的优缺点？
4. KV Cache 的原理？如何优化 KV Cache 的显存占用？
5. BPE 的训练和推理过程？
6. Scaling Laws 的核心结论是什么？
7. Pre-Norm vs Post-Norm 的区别？
8. SwiGLU 激活函数是什么？为什么比 ReLU 好？
9. Flash Attention 的原理？为什么能加速？
10. LLM 的涌现能力（Emergent Abilities）如何解释？

**RAG 相关题：**
1. RAG 系统的核心流程是什么？每个环节有哪些优化点？
2. 混合检索（向量+BM25）为什么通常效果更好？
3. Re-ranking 的原理？Bi-Encoder vs Cross-Encoder 的区别？
4. 如何评估 RAG 系统的效果？有哪些指标？
5. Chunk 策略如何选择？chunk size 对效果的影响？
6. Lost in the Middle 问题是什么？如何解决？
7. 如何处理 RAG 中的幻觉问题？
8. GraphRAG 适合什么场景？与向量 RAG 的区别？
9. Embedding 模型如何微调？在什么场景下需要微调？
10. 向量数据库的索引类型有哪些？HNSW vs IVF 的区别？

**Agent 相关题：**
1. AI Agent 的核心组件有哪些？
2. ReAct 模式的原理？优缺点？
3. 多 Agent 协作有哪些模式？各自的适用场景？
4. Agent 的记忆系统如何设计？
5. 如何防止 Agent 陷入无限循环？
6. Function Calling 的实现原理？
7. 如何评估 Agent 的效果？
8. Agent 的安全性如何保证？Prompt Injection 如何防护？
9. LangGraph 的核心概念是什么？和 LangChain Chains 的区别？
10. MCP 协议是什么？解决了什么问题？

**微调与对齐题：**
1. LoRA 的原理是什么？为什么有效？
2. QLoRA 如何降低显存？NF4 量化是什么？
3. SFT 的数据如何构建？质量和数量哪个更重要？
4. RLHF 的完整流程是什么？有哪些挑战？
5. DPO 相比 RLHF 的优势是什么？
6. GRPO 的创新点？DeepSeek-R1 的训练方法？
7. Reward Hacking 是什么？如何缓解？
8. 对齐税（Alignment Tax）如何理解？
9. 什么情况下需要微调？什么情况下 RAG/Prompt 就够了？
10. 微调的过拟合如何判断和缓解？

**推理优化题：**
1. vLLM 的 PagedAttention 原理？
2. Continuous Batching vs Static Batching 的区别？
3. 模型量化有哪些方法？GPTQ vs AWQ 的区别？
4. 如何估算一个模型的显存占用？
5. 张量并行和流水线并行的区别？

#### 5.2 项目深挖准备

**每个项目准备以下问题的答案：**

1. **Why** — 为什么做这个项目？解决什么问题？
2. **What** — 整体架构是什么？核心模块有哪些？
3. **How** — 关键技术点是怎么实现的？
4. **Metrics** — 效果怎么衡量？具体数据是多少？
5. **Challenge** — 遇到的最大挑战是什么？怎么解决的？
6. **Trade-off** — 做了哪些技术选型？为什么？
7. **Improvement** — 如果重新做，有什么改进空间？
8. **Scale** — 如何应对流量增长？系统瓶颈在哪？

**示例：RAG 项目深挖**
```
Q: 你的 RAG 系统检索效果如何优化的？
A: 主要做了三方面优化：
   1. 检索策略：从纯向量检索改为混合检索（向量+BM25+RRF融合），Recall@5 提升 15%
   2. 分割策略：从固定 512 token 分割改为语义分割（Semantic Chunking），减少了信息截断
   3. 重排序：加入 BGE-Reranker-v2 精排，Precision@3 提升 20%
   效果：整体答案准确率从 72% 提升到 89%（基于 200 个标注测试集）

Q: 遇到的最大挑战是什么？
A: 长文档中的表格数据检索效果差。解决方案：
   1. 对表格进行结构化解析，转换为自然语言描述
   2. 为表格数据单独建立 metadata 索引
   3. 查询路由：识别到数值类查询时优先走结构化检索
```

#### 5.3 系统设计题准备

**常见系统设计题：**

1. **设计一个企业级 RAG 系统**
   - 考点：文档处理流水线、检索优化、高可用、增量更新

2. **设计一个多 Agent 客服系统**
   - 考点：Agent 路由、知识库集成、人工转接、对话管理

3. **设计一个代码生成 Agent**
   - 考点：代码理解、测试生成、安全沙箱、迭代改进

4. **设计一个 LLM 推理服务**
   - 考点：高可用、负载均衡、模型更新、成本优化

**系统设计答题框架：**
```
1. 需求澄清（5分钟）
   - 功能需求、非功能需求
   - 规模、延迟、可用性要求

2. 高层架构（10分钟）
   - 画出核心组件和数据流
   - 说明技术选型理由

3. 核心模块深入（15分钟）
   - 重点模块的详细设计
   - 关键算法和数据结构

4. 扩展性和优化（5-10分钟）
   - 性能优化
   - 扩展方案
   - 容灾设计
```

#### 5.4 模拟面试策略

**自我模拟：**
- 设定计时器，每道题限时回答
- 录音回放，检查表达清晰度
- 写下答案大纲，确保逻辑清晰

**找人模拟：**
- 找同方向的朋友互相面试
- 付费 mock interview 服务
- 牛客网模拟面试

**面试节奏：**
- 开场：30秒自我介绍（突出 AI Agent 经验）
- 项目介绍：STAR 法则（Situation-Task-Action-Result）
- 技术问题：先说结论，再展开细节
- 不会的题：说出思路方向，展示思考过程

---

## 二、推荐学习资源

### 必读论文（10篇）

| # | 论文名称 | 年份 | 一句话总结 | 为什么重要 |
|---|---------|------|-----------|-----------|
| 1 | **Attention Is All You Need** | 2017 | 提出 Transformer 架构，用自注意力替代 RNN | 一切的起点，必须精读 |
| 2 | **BERT: Pre-training of Deep Bidirectional Transformers** | 2018 | 双向预训练+微调范式 | 理解 Encoder 架构和预训练 |
| 3 | **Language Models are Few-Shot Learners (GPT-3)** | 2020 | 展示大模型的 in-context learning 能力 | 理解 LLM 的涌现能力 |
| 4 | **Training language models to follow instructions (InstructGPT)** | 2022 | RLHF 对齐方法的首次大规模应用 | RLHF 的开山之作 |
| 5 | **LoRA: Low-Rank Adaptation of Large Language Models** | 2021 | 低秩分解实现参数高效微调 | 微调领域最重要的论文 |
| 6 | **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | 2020 | 首次提出 RAG 范式 | RAG 的奠基论文 |
| 7 | **ReAct: Synergizing Reasoning and Acting in Language Models** | 2022 | 将推理和行动交替进行 | Agent 设计的基础范式 |
| 8 | **Direct Preference Optimization (DPO)** | 2023 | 无需奖励模型的偏好优化 | 简化 RLHF 的里程碑 |
| 9 | **DeepSeek-V3 / DeepSeek-R1 技术报告** | 2024-2025 | MoE + MLA + GRPO | 国产顶尖模型的技术细节 |
| 10 | **Graph RAG: Unlocking LLM discovery on narrative private data** | 2024 | 知识图谱增强的 RAG | RAG 的重要进化方向 |

### 必看视频/课程（10个）

| # | 资源 | 平台 | 说明 |
|---|------|------|------|
| 1 | **Andrej Karpathy - Let's build GPT** | YouTube | 从零实现 GPT，2小时深度理解 Transformer |
| 2 | **Andrej Karpathy - Let's build the GPT Tokenizer** | YouTube | 理解 BPE Tokenizer 的实现 |
| 3 | **Andrej Karpathy - Intro to Large Language Models** | YouTube | LLM 全景概述，最佳入门 |
| 4 | **3Blue1Brown - Neural Networks / Transformer 系列** | YouTube | 数学直觉可视化，必看 |
| 5 | **吴恩达 - AI Agents in LangGraph** | DeepLearning.ai | Agent 和 LangGraph 实战 |
| 6 | **吴恩达 - Building RAG Agents with LLMs** | DeepLearning.ai | RAG 从入门到进阶 |
| 7 | **吴恩达 - ChatGPT Prompt Engineering** | DeepLearning.ai | Prompt Engineering 经典入门 |
| 8 | **李沐 - Transformer / GPT / BERT 论文精读** | B站 | 中文论文精读，逐段讲解 |
| 9 | **李沐 - 动手学深度学习** | B站/课程 | PyTorch 深度学习基础 |
| 10 | **Hugging Face NLP Course** | HuggingFace | Transformers 库实战 |

### 必读博客/文档

**国外：**
| 资源 | 说明 |
|------|------|
| [LangChain 官方文档](https://python.langchain.com/docs/) | Agent/RAG 框架，必须精读 |
| [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) | Agent 编排框架 |
| [Anthropic Engineering Blog](https://www.anthropic.com/engineering) | Building Effective Agents 等高质量文章 |
| [OpenAI Cookbook](https://cookbook.openai.com/) | 官方最佳实践和示例 |
| [Lilian Weng's Blog](https://lilianweng.github.io/) | LLM Agent、RAG 综述类必读 |
| [Jay Alammar's Blog](http://jalammar.github.io/) | 图解 Transformer 系列 |
| [DAIR.AI Prompt Engineering Guide](https://www.promptingguide.ai/) | 最全面的 Prompt 工程指南 |
| [Chip Huyen's Blog](https://huyenchip.com/blog/) | LLM 工程化实践 |

**国内：**
| 资源 | 说明 |
|------|------|
| [面壁智能技术博客](https://modelbest.cn/blog) | CPM 系列、Agent 相关 |
| [智源研究院](https://www.baai.ac.cn/) | BGE Embedding、学术前沿 |
| [深度求索技术博客](https://www.deepseek.com/) | DeepSeek 技术报告 |
| [通义实验室](https://tongyi.aliyun.com/) | Qwen 模型系列 |
| [知乎 AI 专栏](https://www.zhihu.com/) | 大量中文技术解读 |
| [极客时间 - AI 大模型课程](https://time.geekbang.org/) | 系统化付费课程 |

### 必刷的开源项目

| 项目 | GitHub Stars | 说明 | 学习重点 |
|------|-------------|------|---------|
| [LangChain](https://github.com/langchain-ai/langchain) | 95k+ | LLM 应用开发框架 | 架构设计、抽象层 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 10k+ | Agent 编排框架 | 状态图、循环、持久化 |
| [vLLM](https://github.com/vllm-project/vllm) | 35k+ | 高性能推理引擎 | PagedAttention、性能优化 |
| [Dify](https://github.com/langgenius/dify) | 55k+ | LLM 应用开发平台 | 工作流编排、RAG 实现 |
| [RAGFlow](https://github.com/infiniflow/ragflow) | 25k+ | 深度文档理解 RAG | 文档解析、混合检索 |
| [GraphRAG](https://github.com/microsoft/graphrag) | 20k+ | 知识图谱增强 RAG | 图谱构建、社区检测 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 40k+ | 一站式微调平台 | LoRA/QLoRA、数据处理 |
| [Ollama](https://github.com/ollama/ollama) | 110k+ | 本地模型运行 | 模型管理、API 设计 |
| [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) | 55k+ | 代码执行 Agent | Agent 设计、安全沙箱 |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 25k+ | 多 Agent 框架 | 角色协作、任务编排 |

---

## 三、面试准备策略

### 3.1 简历优化

**项目描述模板：**
```
【项目名称】xxx 智能问答系统
【项目背景】解决 xxx 场景下的 xxx 问题
【技术方案】基于 LangGraph + Milvus + Qwen 构建混合检索 RAG 系统
           实现 xxx、xxx、xxx 核心功能
【个人职责】- 设计并实现 xxx 模块
           - 优化 xxx，效果提升 xxx%
           - 搭建 xxx 评估框架
【项目成果】- 准确率从 xx% 提升至 xx%
           - 日均服务 xxx 次查询
           - 延迟 P99 < xxx ms
```

**技术关键词布局：**
- 标题/概述：LLM、Agent、RAG、大模型
- 技术栈：LangChain、LangGraph、vLLM、Milvus、PyTorch
- 方法：LoRA微调、混合检索、ReAct Agent、RLHF/DPO
- 模型：Qwen、DeepSeek、Llama、GPT-4
- 工具：Docker、Kubernetes、Redis、PostgreSQL

**注意事项：**
- 数据要具体：不要"显著提升"，要"提升15%"
- 技术选型要有理由
- 突出自己的贡献，不要只写团队成果
- GitHub 链接放上去（有代码加分）

### 3.2 面试流程

**大厂 AI 岗面试通常 3-5 轮：**

| 轮次 | 面试官 | 时长 | 考察重点 |
|------|--------|------|---------|
| **一面：技术面** | 高级工程师 | 60-90min | 基础知识 + 项目深挖 + 手撕代码 |
| **二面：技术面** | 技术专家/主管 | 60-90min | 系统设计 + 项目深挖 + 技术深度 |
| **三面：交叉面** | 其他团队技术Leader | 45-60min | 技术广度 + 项目经验 + 问题解决能力 |
| **四面：HR面/主管面** | HR/部门主管 | 30-45min | 职业规划 + 团队匹配 + 薪资 |
| **加面（可能）** | VP/高层 | 30min | 视野 + 潜力 + 文化匹配 |

**每一轮的考察重点：**

**一面 - 技术基础：**
- Transformer 原理（注意力、位置编码、归一化）
- LLM 训练和推理（预训练、SFT、解码策略）
- RAG 系统设计和优化
- 代码题：手写注意力机制 / BPE / 简单 Agent 逻辑

**二面 - 技术深度：**
- 项目深挖（追问3-5层，确认真正做过）
- 系统设计（设计一个 Agent 系统 / RAG 服务）
- 技术 trade-off 讨论
- 开放性问题（如何看待 xxx 技术的发展？）

**三面 - 综合能力：**
- 跨领域知识（NLP + CV + 推荐系统）
- 问题分析和解决思路
- 学习能力和技术热情
- 沟通和表达能力

### 3.3 高频考点分布（按公司）

**字节跳动：**
- 重视工程能力，代码题难度较高
- Agent 系统设计（豆包等产品线）
- RAG 优化和评估
- 模型部署和推理优化
- 考 LeetCode Medium-Hard

**阿里巴巴/蚂蚁集团：**
- 通义千问相关的模型知识
- 大规模 RAG 系统（企业级文档场景）
- Agent 在电商/金融场景的应用
- 分布式系统设计
- 考 Java/Python 基础 + 算法

**腾讯：**
- 混元大模型相关
- 社交/游戏场景的 AI 应用
- 多模态能力
- 微信生态的 Agent 设计
- 考编程基础扎实

**百度：**
- 文心一言技术栈
- 搜索增强的 RAG
- 知识图谱 + LLM
- 传统 NLP 基础也考
- 考 Paddle 框架经验加分

**小红书：**
- Agent 在内容创作/推荐中的应用
- 多模态理解（图文）
- 用户意图理解
- 快速迭代和工程能力

**美团：**
- 本地生活场景的 AI 应用
- NL2SQL（自然语言转SQL）
- 搜索和推荐
- 工程化和可靠性

**华为：**
- 盘古大模型技术栈
- 端侧 LLM 部署
- 模型压缩和量化
- 安全和隐私

### 3.4 面试心态与技巧

**不会的题怎么答：**
```
"这个问题我没有深入研究过，但我可以分享一下我的理解和思路：
1. 从 xxx 原理出发，我认为...
2. 类似的问题 xxx 我有经验，可以类比...
3. 如果让我去解决，我会先...然后...
比起给一个不确定的答案，我更愿意承认盲区并展示我的思考过程。"
```

**如何展示深度思考：**
- 不要只说"是什么"，要说"为什么"
- 主动讲 trade-off：这个方案的优缺点分别是...
- 对比不同方案：A 方案 vs B 方案，我选择 A 是因为...
- 提到实际经验：在我的项目中，我发现...
- 关联最新进展：最近 xxx 论文/框架提出了...

**项目被追问时的策略：**
- 诚实回答，不要编造没做过的事
- 准备好每个技术选型的理由
- 准备好"如果重新来过"的改进方案
- 准备好具体数据（准确率、延迟、QPS等）
- 如果是团队项目，明确说清自己的贡献

**其他技巧：**
- 面试前了解公司的 AI 产品和技术博客
- 准备2-3个问面试官的好问题
- 面试后做复盘，记录考了什么
- 保持积极心态，一家不行还有下一家

---

## 四、每周学习计划模板（16周）

> 以下是完整的 16 周学习计划，每周明确学习主题、具体任务和预期产出。
> 每天建议投入 3-4 小时（工作日）+ 6-8 小时（周末）。

### 第1周：Transformer 架构

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | 自注意力机制 | 看 Jay Alammar 图解 Transformer，手推 QKV 计算 | 笔记：自注意力机制原理 |
| 周二 | 多头注意力 + MQA/GQA | 看 3Blue1Brown 视频，整理 MHA/MQA/GQA 区别 | 对比表格 |
| 周三 | 位置编码 | 学习 RoPE 原理，看论文/博客解读 | 笔记：RoPE 推导和直觉 |
| 周四 | FFN + 归一化 | SwiGLU、RMSNorm、Pre-Norm vs Post-Norm | 笔记：现代 Transformer 的改进 |
| 周五 | 完整架构 | 看 Andrej Karpathy Let's build GPT 视频 | 跟着视频写代码 |
| 周末 | 动手实践 | 用 PyTorch 手写简化版 Self-Attention 和 Transformer Block | ✅ 代码：mini-transformer |

### 第2周：LLM 基础

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | Tokenization | 看 Karpathy Tokenizer 视频，实现 BPE | 代码：BPE Tokenizer |
| 周二 | 预训练流程 | 看 State of GPT 视频，学习训练目标 | 笔记：LLM 训练全流程 |
| 周三 | Scaling Laws | 读 Chinchilla 论文，理解最优配比 | 笔记：Scaling Laws 核心结论 |
| 周四 | 解码策略 | 实现 greedy/beam search/sampling | 代码：各种解码策略 |
| 周五 | 主流模型 | 调研 GPT/Claude/Llama/Qwen/DeepSeek | 整理：模型对比表 |
| 周末 | PyTorch 实战 | 用 PyTorch 训练一个 mini 语言模型 | ✅ 代码：mini-LM 训练 |

### 第3周：Prompt Engineering

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | Zero/Few-shot | 学习基础 Prompting 技巧，实践各种场景 | 笔记：Prompt 技巧速查 |
| 周二 | Chain-of-Thought | 学习 CoT 及变体，做推理任务实验 | 实验报告：CoT 效果对比 |
| 周三 | ReAct + Tool Use | 学习 ReAct 模式，理解 Function Calling | 笔记：ReAct 原理 |
| 周四 | 高级技巧 | Self-Consistency、ToT、Reflection | 笔记：高级 Prompt 技巧 |
| 周五 | Context Engineering | 学习上下文工程的系统设计方法 | 笔记：Context Engineering |
| 周末 | 综合实践 | 设计一个复杂场景的完整 Prompt 系统 | ✅ 完整 System Prompt 设计 |

### 第4周：RAG 基础

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | 文档处理 | 学习文本分割策略，实现各种 chunking 方法 | 代码：文档处理 pipeline |
| 周二 | Embedding | 对比不同 Embedding 模型，学习评估方法 | 实验：Embedding 效果对比 |
| 周三 | 向量数据库 | Chroma 实操，学习 HNSW/IVF 索引 | 代码：向量检索 demo |
| 周四 | Naive RAG | 搭建第一个完整的 RAG 系统 | ✅ 代码：Naive RAG |
| 周五 | 混合检索 | 实现 BM25 + 向量 + RRF 融合 | 代码：混合检索 |
| 周末 | RAG 评估 | 学习 RAGAs，构建评估数据集 | 评估报告 |

### 第5周：Advanced RAG

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | 查询改写 | 实现 Multi-Query、HyDE | 代码：查询改写模块 |
| 周二 | Re-ranking | 集成 BGE-Reranker，对比效果 | 实验：Re-ranking 效果 |
| 周三 | 多轮对话 RAG | 实现对话历史管理、指代消解 | 代码：多轮 RAG |
| 周四 | 幻觉检测 | 实现基于 NLI 的幻觉检测 | 代码：幻觉检测模块 |
| 周五 | 系统集成 | 整合所有模块成 Advanced RAG 系统 | ✅ 代码：Advanced RAG |
| 周末 | 效果对比 | Naive vs Advanced，写对比报告 | 对比分析报告 |

### 第6周：Agent 设计模式

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | Agent 概念 | 读 Lilian Weng 博客，理解核心组件 | 笔记：Agent 核心概念 |
| 周二 | ReAct 模式 | 用纯 Python 实现 ReAct Agent | ✅ 代码：ReAct Agent |
| 周三 | Plan-and-Execute | 实现计划-执行模式 | 代码：Plan-Execute Agent |
| 周四 | Reflection | 实现自我反思模式 | 代码：Reflection Agent |
| 周五 | 多 Agent | 理解 Supervisor/Debate 等模式 | 笔记 + 架构图 |
| 周末 | 记忆系统 | 实现短期+长期记忆系统 | 代码：Agent 记忆模块 |

### 第7周：LangGraph 精通

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | LangGraph 基础 | State、Node、Edge 概念，官方教程 | 笔记 + 基础 demo |
| 周二 | ReAct in LangGraph | 用 LangGraph 实现 ReAct Agent | 代码 |
| 周三 | 多 Agent | Supervisor 模式实现 | 代码 |
| 周四 | Human-in-the-loop | 实现人工审核节点 | 代码 |
| 周五 | 持久化 + 流式 | Checkpointing 和 Streaming | 代码 |
| 周末 | 综合项目 | 完成一个功能完整的 LangGraph Agent | ✅ 项目：LangGraph Agent |

### 第8周：其他框架 + MCP

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | CrewAI | 搭建一个多角色协作系统 | 代码：CrewAI demo |
| 周二 | Dify | 用 Dify 搭建一个 RAG 应用 | 应用：Dify RAG |
| 周三 | MCP 协议 | 学习 MCP 协议规范，实现一个工具 Server | 代码：MCP Server |
| 周四 | A2A 协议 | 了解 Agent 间通信协议 | 笔记：A2A 概述 |
| 周五 | 框架对比 | 横向对比各框架 | 对比分析文档 |
| 周末 | 复习总结 | 复习阶段一二所有内容 | ✅ 阶段复习笔记 |

### 第9周：模型微调

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | SFT 基础 | 学习 SFT 流程和数据格式 | 笔记：SFT 流程 |
| 周二 | LoRA 原理 | 推导 LoRA 数学原理 | 笔记：LoRA 原理推导 |
| 周三 | LoRA 实战 | 用 PEFT 库微调 Qwen2.5-7B | 代码：LoRA 微调 |
| 周四 | QLoRA | 4-bit 量化 + LoRA 微调 | 代码：QLoRA 微调 |
| 周五 | 数据构建 | 构建高质量微调数据集 | 数据集：500+ 条 |
| 周末 | 效果评估 | 对比微调前后效果 | ✅ 评估报告 |

### 第10周：对齐技术

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | RLHF 原理 | 学习 RLHF 完整流程 | 笔记：RLHF 全流程 |
| 周二 | 奖励模型 | 理解 RM 训练 | 笔记：奖励模型设计 |
| 周三 | DPO 原理 | 推导 DPO 损失函数 | 笔记：DPO 原理 |
| 周四 | DPO 实战 | 用 TRL 库做 DPO 训练 | 代码：DPO 训练 |
| 周五 | GRPO | 学习 DeepSeek-R1 的 GRPO | 笔记：GRPO 原理 |
| 周末 | 技术演进 | 整理对齐技术发展脉络 | ✅ 技术综述文档 |

### 第11周：推理优化

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | KV Cache | 学习 KV Cache 原理和优化 | 笔记 |
| 周二 | vLLM | 部署模型，学习 PagedAttention | 代码：vLLM 部署 |
| 周三 | 量化 | GPTQ、AWQ 量化实操 | 实验：量化效果对比 |
| 周四 | 性能压测 | 压测不同配置的推理性能 | 压测报告 |
| 周五 | 部署架构 | 设计生产级部署方案 | 架构设计文档 |
| 周末 | 高级 RAG | GraphRAG 实操 | ✅ 代码：GraphRAG demo |

### 第12周：安全评估 + 项目一启动

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | Prompt Injection | 学习攻击和防护方法 | 笔记：安全防护 |
| 周二 | Guardrails | 实现基本的输入输出防护 | 代码：Guardrails |
| 周三 | Agent 评估 | 设计评估框架 | 评估方案文档 |
| 周四 | 项目一规划 | RAG 知识问答系统设计 | 项目设计文档 |
| 周五 | 项目一开发 | 文档处理 + 索引构建 | 代码 |
| 周末 | 项目一开发 | 检索系统实现 | 代码 |

### 第13周：项目一完成

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | 项目一开发 | 混合检索 + Re-ranking | 代码 |
| 周二 | 项目一开发 | 生成系统 + 幻觉检测 | 代码 |
| 周三 | 项目一开发 | 多轮对话 + 前端 | 代码 |
| 周四 | 项目一优化 | 效果评估和优化 | 评估报告 |
| 周五 | 项目一收尾 | README、文档、部署 | ✅ 项目一完成 |
| 周末 | 项目二启动 | 多 Agent 系统设计 | 项目设计文档 |

### 第14周：项目二 - 多 Agent 协作系统

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | Agent 定义 | 设计各 Agent 角色和工具 | 代码 |
| 周二 | LangGraph 编排 | 实现 Supervisor + Worker 图 | 代码 |
| 周三 | 工具集成 | 搜索/代码执行/文件操作 | 代码 |
| 周四 | Human-in-the-loop | 实现人工审核 | 代码 |
| 周五 | 持久化 + 错误处理 | Checkpointing + 重试 | 代码 |
| 周末 | 测试和部署 | 完善文档，部署 | ✅ 项目二完成 |

### 第15周：项目三 - 生产级 Agent 应用

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | 架构设计 | 系统架构 + 技术选型 | 架构文档 |
| 周二 | 核心功能 | 智能路由 + 工具系统 | 代码 |
| 周三 | 记忆系统 | 短期 + 长期记忆 | 代码 |
| 周四 | 安全防护 | 输入检测 + 输出过滤 + 权限 | 代码 |
| 周五 | 可观测性 | 监控 + 告警 + 日志 | 代码 |
| 周末 | Docker 化 | 容器化部署 + 文档 | ✅ 项目三完成 |

### 第16周：面试冲刺

| 日期 | 学习内容 | 具体任务 | 产出 |
|------|---------|---------|------|
| 周一 | 知识梳理 | 复习所有核心知识点 | 知识思维导图 |
| 周二 | 高频题刷题 | 理论题 + 代码题 | 答案整理 |
| 周三 | 项目深挖 | 准备每个项目的 8 个追问答案 | 项目深挖文档 |
| 周四 | 系统设计 | 练习 2-3 道系统设计题 | 设计方案 |
| 周五 | 模拟面试 | 找人模拟一次完整面试 | 面试复盘 |
| 周末 | 最终准备 | 简历打磨、公司调研、查缺补漏 | ✅ 准备完毕，开始投递！ |

---

## 附录

### A. 学习路线图速览

```
Week 1-2   ▸ 基础夯实：Transformer + LLM + PyTorch
Week 3-4   ▸ Prompt + RAG 基础
Week 5-6   ▸ Advanced RAG + Agent 设计模式
Week 7-8   ▸ LangGraph + 框架实战
Week 9-10  ▸ 微调 + 对齐技术
Week 11    ▸ 推理优化
Week 12-13 ▸ 安全评估 + 项目一（RAG系统）
Week 14    ▸ 项目二（多Agent系统）
Week 15    ▸ 项目三（生产级Agent）
Week 16    ▸ 面试冲刺
```

### B. 每日学习建议

| 时间 | 工作日 | 周末 |
|------|--------|------|
| 早上 | 30min 论文/博客阅读 | 2h 深度学习/实践 |
| 午休 | 20min 刷面试题 | — |
| 晚上 | 2-3h 代码实践 | 4-6h 项目开发 |
| 睡前 | 15min 复习笔记 | 30min 总结 |

### C. GitHub 仓库组织建议

```
your-github/
├── ai-agent-learning/          # 学习笔记和实验
│   ├── transformer/            # Transformer 实现
│   ├── tokenizer/              # BPE 实现
│   ├── rag-experiments/        # RAG 实验
│   └── agent-patterns/         # Agent 模式实现
├── rag-qa-system/              # 项目一：RAG 知识问答
├── multi-agent-research/       # 项目二：多 Agent 研究助手
├── production-agent/           # 项目三：生产级 Agent
└── interview-notes/            # 面试题整理
```

### D. 社区和交流

- **GitHub**：关注上述开源项目，看 issue 和 PR
- **知乎/公众号**：关注 AI Agent 领域的优质作者
- **Discord**：LangChain、HuggingFace 等社区
- **牛客网**：面经和讨论
- **技术会议**：WAIC、智源大会、QCon AI 专场

### E. 心态建议

1. **不要贪多求全**：AI 领域发展太快，聚焦核心技术
2. **动手 > 看书**：代码实现 > 理论学习的比例应该 6:4
3. **项目驱动**：所有学习最终要落地到项目
4. **持续更新**：关注最新论文和技术，每周至少读 2-3 篇
5. **面试是双向选择**：不要害怕被拒，每次面试都是学习机会
6. **技术深度 > 广度**：在核心方向（如 RAG 或 Agent）做到深入，比什么都会一点更有竞争力
7. **保持健康**：学习是持久战，注意休息和运动

---

> 📌 **最后的话：** 这份路线图覆盖了从基础到面试的完整路径。不需要全部掌握，根据目标岗位选择重点方向。最重要的是：**动手做项目，写到简历上，讲给面试官听。** 祝你拿到理想 Offer！🎯

---

*最后更新：2025年*
*本文档持续更新中，欢迎提交 PR 补充内容*
