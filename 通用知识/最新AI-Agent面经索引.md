# 最新 AI Agent / 大模型面经索引（2026）

> 本文整理自公开网页搜索结果，用作面经入口和复习优先级索引。为避免搬运平台原文，本文只保留来源链接、摘要和考点标签。结构化数据见 [`data/interviews.json`](../data/interviews.json)。

## 采集概况

| 来源 | 高质量候选 | 说明 |
|---|---:|---|
| 小红书 | 13 | 时效强，覆盖字节、腾讯、美团、快手、阿里、百度、小红书等一面/二面 |
| GitHub | 10 | 高质量题库和学习资料，适合作为系统复习资料 |
| 牛客 | 8 | 国内大厂面经密度高，适合补充真实面试流程和追问 |
| CSDN / 博客园 / 掘金 | 8 | 适合作为 Agent/RAG 高频题和专题归纳来源 |
| 知乎 | 2 | 适合作为大模型算法实习和海外 AI 面试补充 |

> 注：小红书搜索结果链接依赖登录态和平台签名，公开静态站直链容易失效。因此本文只保留小红书原标题，阅读时请在小红书站内搜索对应标题。

## P0：优先阅读的真实面经

### 字节跳动

| 来源 | 标题 | 核心考点 |
|---|---|---|
| 小红书 | 字节跳动Agent开发岗二面（贼难）（小红书站内搜索原标题） | 多Agent、LangGraph、Skills、上下文工程、Agent评估、SFT |
| 小红书 | 字节 agent开发 1-3面面经 5月（小红书站内搜索原标题） | 豆包场景、长会话、多Agent A2A、Memory、RAG、幻觉治理 |
| 小红书 | 字节跳动AI开发 一面（小红书站内搜索原标题） | AI 应用开发一面 |
| 牛客 | [字节 AI 应用岗面试真题](https://www.nowcoder.com/discuss/882634966025175040) | 法律RAG、工具路由、评测、chunking、rerank |
| 牛客 | [4轮拿下字节Offer！LLM面试题合集](https://www.nowcoder.com/discuss/746382064101908480) | RAG召回排序、Prompt评测、ReAct、LoRA、SFT、Attention、幻觉 |
| 博客园 | [字节 AI Agent 二面（飞连）面试题与参考解答](https://www.cnblogs.com/tuaran/p/20164742) | Tool Calling、bad case、Memory、RAG优化、Agentic RAG |
| CSDN | [字节跳动大模型实习面经：从 Agent 记忆到 RAG 优化](https://gitcode.csdn.net/6a2ccdca10ee7a33f27c07bf.html) | 长短期记忆、Query Rewrite、Hybrid Retrieval、RRF、Rerank、HyDE |
| CSDN | [双非本｜字节跳动飞书团队 RAG 面经](https://devpress.csdn.net/v1/article/detail/151567056) | BGE-M3、Qwen3-Embedding、LoRA、多路召回、父子文档 |

### 腾讯

| 来源 | 标题 | 核心考点 |
|---|---|---|
| 小红书 | 腾讯ai应用一面面经（小红书站内搜索原标题） | 多Agent协作、AgentContext、Run Trace、Skills、上下文注入、风控Agent |
| 小红书 | 腾讯 Agent二面凉经带答案（小红书站内搜索原标题） | Agent 二面深挖 |
| 小红书 | 腾讯 ai 应用开发 一面（小红书站内搜索原标题） | AI 应用开发一面 |
| 牛客 | [大模型、Agent面经总结【04/28】腾讯 / 百度 总结](https://www.nowcoder.com/discuss/878600528970735616) | Agent编排、RAG热更新、失败重试、金融安全、LoRA/DPO |

### 阿里 / 蚂蚁 / 淘天

| 来源 | 标题 | 核心考点 |
|---|---|---|
| 小红书 | 淘天AI Agent一面 问麻了（小红书站内搜索原标题） | RAG、BM25、OCR、多Agent State、Checkpoint、MCP、LangGraph、SSE |
| 牛客 | [阿里淘宝闪购 · Agent 算法工程师 · 27届实习一面](https://www.nowcoder.com/discuss/879393838081597440) | Agent框架选型、HITL、风险控制、Memory、AI Coding |
| 牛客 | [5月20日，蚂蚁智能体与大模型应用 一面](https://www.nowcoder.com/discuss/888874988554448896) | 幻觉治理、Skill、Spring AI、Claude Code、混合检索 |
| 牛客 | [Agent 开发面经总结【04/24】阿里巴巴 / 蚂蚁 / 字节跳动 总结](https://www.nowcoder.com/discuss/877151327091027968) | Multi-Agent、RAG、MCP、Function Calling、LangChain/LangGraph |
| 牛客 | [蚂蚁秋招时间线+面经](https://www.nowcoder.com/discuss/800426409796624384) | RAG vs Fine-tuning、Rerank、NDCG、Agent评测、MCP/A2A |

### 美团 / 快手 / 百度 / 京东 / 小红书

| 公司 | 来源 | 标题 | 核心考点 |
|---|---|---|---|
| 美团 | 小红书 | 美团 AI Agent开发 一面面经（小红书站内搜索原标题） | PDF RAG、RAGAS、MCP、ReAct、Rerank、JSON工具调用 |
| 美团 | 小红书 | 美团AI-Agent工程师面经，看看难度（小红书站内搜索原标题） | AI-Agent 工程师面经 |
| 美团 | 知乎 | [Meituan Large Model Algorithm Intern questions](https://www.zhihu.com/en/article/688624199) | RAG项目、指标、Qwen、LoRA、PDF/表格解析 |
| 快手 | 小红书 | 快手AI Agent开发一面（小红书站内搜索原标题） | 父子索引、BM25、Rerank、Memory、Prompt注入防御、RAG评测 |
| 快手 | 小红书 | 快手AI Agent开发实习生面经2026.6.12（小红书站内搜索原标题） | ReAct、多模型API、单/多Agent、RAG优化、算法题 |
| 百度 | 小红书 | 百度 大模型应用开发 一面面经（小红书站内搜索原标题） | 多模态RAG、表格切片、Memory总结、Function Call、语义缓存 |
| 京东 | 牛客 | [京东 Agent开发 暑期一面](https://www.nowcoder.com/discuss/876932752833077248) | 电商RAG、意图识别、Redis向量检索、HNSW、幻觉兜底 |
| 小红书 | 小红书 | 小红书 AI Agent开发一面（小红书站内搜索原标题） | 电商Agent、主从架构、Tool调用、RAG索引、Prompt调试 |
| 理想汽车 | 知乎 | [Li Auto Large Model Algorithm Intern experience](https://www.zhihu.com/en/article/680860432) | RAG、数据集规模、SFT、CoT/ToT、部署、vLLM |

## P1：专题题库与高频考点

| 来源 | 标题 | 建议用途 |
|---|---|---|
| CSDN | [AI Agent开发面试高频题曝光！从203篇面经提炼](https://blog.csdn.net/Trb701012/article/details/162102884) | 提炼 Agent 高频题，补充通用题库 |
| 掘金 | [RAG大厂面试题汇总：向量检索、混合检索、Rerank、幻觉处理](https://juejin.cn/post/7652557621874409522) | 补充 RAG 专项题 |
| 掘金 | [2026 最新 AI Agent 岗面试复盘：拿到三个 offer](https://juejin.cn/post/7625576464485842979) | 做 Agent 岗准备策略 |
| 博客园 | [面试 AI Agent 工程师会被问什么？40+ 真题 + 知识图谱全梳理](https://www.cnblogs.com/itech/p/20111938) | 通用 Agent 知识图谱 |
| CSDN | [大模型应用开发面试宝典：22家公司真实面试经验与技术考点总结](https://devpress.csdn.net/v1/article/detail/151832648) | 多公司 LLM 应用岗总结 |

## P2：外部高质量 GitHub 资料

> 这些资源更适合作为“参考资料索引”，不应标记为真实面经。

| 资源 | 价值 |
|---|---|
| [AgentGuide](https://github.com/adongwanai/AgentGuide) | AI Agent开发指南、LangGraph、高级RAG、大模型面试 |
| [ai-agent-interview-guide](https://github.com/bcefghj/ai-agent-interview-guide) | AI Agent 面试全攻略，含题库、项目、简历和系统设计 |
| [LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody) | 大模型知识与面试准备 |
| [FAQ_Of_LLM_Interview](https://github.com/aceliuchanghong/FAQ_Of_LLM_Interview) | 大模型算法岗面试题含答案 |
| [LLM-Agent-Interview-Guide](https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide) | 300+ Q&A，字节/阿里/腾讯真题 |
| [AngleMAXIN/llm-application-interview](https://github.com/AngleMAXIN/llm-application-interview) | 多家大厂真实 LLM 应用面试问题 |
| [JavaGuide docs/ai](https://github.com/Snailclimb/JavaGuide/tree/main/docs/ai) | 高可信中文 AI 面试体系 |
| [toBeBetterJavaer AI section](https://github.com/itwanger/toBeBetterJavaer/tree/master/docs/src/sidebar/itwanger/ai) | Agent 258题、大模型333题、RAG+Agent+MCP |
| [wdndev/llm_interview_note](https://github.com/wdndev/llm_interview_note) | 中文 LLM 面试笔记 |
| [RAG-Interview-Questions-and-Answers-Hub](https://github.com/KalyanKS-NLP/RAG-Interview-Questions-and-Answers-Hub) | 英文 RAG 100+ Q&A |

## 复习优先级

1. **先看公司面经**：字节、腾讯、阿里/蚂蚁、美团、快手、百度。
2. **再刷 RAG 专项**：Chunk、混合检索、Rerank、RAGAS、语义缓存、表格/PDF。
3. **补 Agent 工程化**：LangGraph、State/Checkpoint、Skills、MCP、Tool safety、Run trace。
4. **最后补 LLM 基础**：LoRA/SFT/DPO、KV Cache、推理框架、量化、上下文压缩。

## 入库原则

- 只保留摘要、问题要点和来源链接，不大段复制原文。
- 小红书和牛客优先作为真实面经来源。
- CSDN、掘金、知乎中推广内容较多，优先作为题目种子和专题归纳来源。
- GitHub 资源单独作为外部资料索引。
