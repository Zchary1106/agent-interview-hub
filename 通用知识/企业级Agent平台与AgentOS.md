# 企业级 Agent 平台与 AgentOS

## 一、平台与单个 Agent 的区别

单个 Agent 解决一个任务；企业级平台要让多个团队安全、稳定、低成本地创建和运行大量 Agent。

```text
渠道层：Web / App / IM / CLI / Email / Webhook
控制面：Identity / Policy / Registry / Config / Release
运行面：Runtime / Queue / Scheduler / Checkpoint / Sandbox
能力面：Model / Tool / MCP / Skill / Memory / Knowledge
治理面：Trace / Eval / Audit / Cost / Incident
```

## 二、AgentOS 的核心抽象

AgentOS 不是普通聊天 UI，也不是一个 Agent 框架。它管理：

- 用户和 Agent 身份；
- Session、Task 和长期工作区；
- 前后台任务；
- interrupt、resume 和 cancel；
- Tool、Skill、Plugin 和模型 Provider；
- 权限、审批、沙箱和凭证；
- Trace、预算和通知。

## 三、多通道 Gateway

统一入口需要把不同平台事件规范化：

```json
{
  "channel": "slack",
  "tenant_id": "tenant-a",
  "user_id": "user-7",
  "session_id": "session-21",
  "thread_id": "thread-9",
  "message_id": "msg-101",
  "content": [],
  "capabilities": ["reply", "upload"],
  "auth_context": {}
}
```

必须隔离四种边界：

1. Tenant：企业数据不能串；
2. User：Agent 不能继承其他用户权限；
3. Channel：同一用户在邮件和 IM 的行为规则可能不同；
4. Workspace：项目文件、记忆和凭证独立。

## 四、控制平面和执行平面

### 控制平面

- Agent Registry；
- Tool/MCP Registry；
- Skill Registry；
- Model Gateway；
- Prompt 和 Policy 版本；
- 发布和回滚；
- 租户、配额和成本中心。

### 执行平面

- 任务队列和 Worker；
- 沙箱；
- 工具执行；
- Checkpoint；
- Artifact Store；
- Event Stream；
- 实时和后台任务。

控制平面不可因为一次模型调用失败而不可用；执行平面也不应拥有无限制修改策略的权限。

## 五、身份与权限

必须区分：

```text
User Identity：谁提出请求
Agent Identity：哪个 Agent 在执行
Runtime Identity：哪个进程或 Worker 在运行
Tool Credential：访问外部系统使用的凭证
```

授权检查应包含：

```text
用户是否有权发起
× Agent 是否允许该能力
× 当前任务是否在授权范围
× 工具凭证是否具有最小 Scope
× 是否需要实时人工审批
```

仅使用 RBAC 往往不够；资源、时间、风险等级等动态条件适合 ABAC。

## 六、三个 Registry

| Registry | 保存内容 |
|---|---|
| Agent Registry | Agent 版本、Owner、模型、策略、允许能力 |
| Tool/MCP Registry | Schema、Scope、风险、超时、幂等、健康状态 |
| Skill Registry | 适用条件、说明、资源、版本、来源、质量和权限 |

Registry 是控制面，不应该直接执行工具。

## 七、模型网关

模型网关负责：

- Provider 适配；
- 模型路由和降级；
- Token/费用统计；
- 限流和配额；
- 数据驻留策略；
- Prompt/Response 审计；
- Tool Call 格式兼容。

不同模型可能产生不同的工具调用格式，需要独立 Parser 和兼容测试，不能假设 OpenAI 风格 JSON 在所有 Provider 上完全一致。

## 八、Agent Builder

Builder 至少提供：

- Prompt/Instruction 编辑；
- Workflow 或状态图；
- Tool、Skill 和知识库绑定；
- 权限和审批配置；
- 评测集；
- 版本、发布和回滚；
- 运行记录和失败分析。

低代码 Builder 适合稳定、边界明确的业务流程；高度动态的 Coding/Research Agent 通常需要代码级扩展。

## 九、多租户治理

- 数据库行级或库级隔离；
- 向量库 Namespace 隔离；
- 对象存储前缀与密钥隔离；
- 租户级模型和工具白名单；
- 配额、并发和预算限制；
- 审计日志不可跨租户查询；
- 管理员操作也要留痕。

## 十、平台发布路径

```text
本地开发
→ 离线 Eval
→ 安全评测
→ Shadow
→ 小流量 Canary
→ 受控写入
→ 自动执行
```

成熟度应按风险逐级开放：

1. 只读；
2. 给建议；
3. 经批准写入；
4. 低风险自动执行；
5. 高风险操作始终保留人工责任人。

## 十一、四平面 AgentOS

仅用“控制面 + 执行面”描述企业平台还不够，面试时可以进一步拆成四个平面：

| 平面 | 核心职责 | 典型状态 |
|---|---|---|
| Control Plane | 定义、版本、发布、路由和配额 | Agent/Prompt/Policy/Model 版本 |
| Data Plane | 知识、记忆、Artifact、Trace 和数据血缘 | 文档、向量、会话、证据 |
| Execution Plane | 调度、Sandbox、Tool、Checkpoint 和恢复 | Task、Lease、Event、Side Effect |
| Governance Plane | 身份、审批、审计、风险、合规和事件响应 | Grant、Audit、Incident、Rollback |

治理不能只是控制面的一个布尔开关。高风险审批、审计保留、跨租户访问检测和事故响应需要独立 Owner、策略与证据。

## 十二、Registry 不止三个

成熟平台通常需要：

| Registry | 需要版本化的内容 |
|---|---|
| Agent Registry | Owner、入口、模型、允许能力、SLO |
| Tool Registry | Schema、风险、幂等、超时、健康状态 |
| MCP Registry | Server、Transport、认证、暴露工具和信任级别 |
| Skill Registry | 适用范围、依赖、来源、测试、权限和版本 |
| Prompt Registry | System/Developer Prompt、变量、兼容模型、评测结果 |
| Model Registry | Provider、能力、上下文、数据策略、价格和弃用日期 |
| Policy Registry | 主体、资源、动作、条件、审批和生效范围 |
| Eval Registry | 数据集、Grader、环境版本、阈值和历史结果 |

Tool 与 MCP 可以共享搜索入口，但不应混为同一概念：Tool 是 Agent 可调用的能力；MCP Server 是发现和调用能力的一种协议端点。

## 十三、协议与网关兼容

企业平台往往同时接入模型 API、MCP、A2A、Webhook、消息渠道和内部 RPC。网关要解决：

- 协议版本和能力协商；
- Streaming、取消、超时和背压；
- Tool Call、错误码和 Artifact 引用的规范化；
- 用户身份向下游的安全委托，而不是共享平台超级凭证；
- 幂等键、Trace Context 和审计字段透传；
- 不兼容能力的显式降级，不能静默丢掉审批或安全字段。

协议转换只解决格式，不自动解决语义。例如两个 Provider 都支持 Tool Calling，也可能在并行调用、严格 Schema、错误重试和历史回放上行为不同。

## 十四、租户、用户、渠道与 Workspace 隔离

隔离至少有四个正交维度：

```text
Tenant：组织数据、密钥、配额、策略
User：个人权限、委托关系、隐私
Channel：Web/IM/Email/CLI 的交互与审批能力
Workspace：仓库、文件、记忆、环境和 Artifact
```

常见漏洞是只在数据库查询加 `tenant_id`，却让缓存键、向量 Namespace、对象存储、队列、日志或模型 Prompt 跨租户复用。应使用统一的资源作用域标识，并在入口、检索、执行和观测四层校验。

## 十五、发布与事故模型

Agent 发布物不只是模型或 Prompt，而是：

```text
Release =
Agent + Prompt + Model + Tool Schema + Skill + Policy
+ Context Strategy + Runtime Image + Eval/Grader
```

每个 Release 应有不可变版本、变更 Diff、离线结果、批准人和回滚目标。事故处理至少包含：

1. Kill switch 停止新任务；
2. 取消或隔离在途任务；
3. 撤销临时授权与凭证；
4. 保全 Trace、Artifact 和环境证据；
5. 判断是否发生不可逆副作用；
6. 回滚 Release 或关闭特定 Tool；
7. 把事故转成回归和安全用例。

只回滚 Prompt 可能无效，因为问题也可能来自 Tool Schema、Skill、模型路由或 Runtime 镜像。

## 十六、Coding Agent / AgentOS 源码证据清单

做平台竞品或源码分析时，不要只画概念图。至少找到：

```text
[ ] CLI/Web/IDE 的真实入口
[ ] Session、Turn、Event 的状态定义
[ ] Agent Loop 与停止条件
[ ] Context 构建、压缩和恢复
[ ] Tool/MCP 注册、选择与执行
[ ] Permission、Approval、Policy、Sandbox 的边界
[ ] Task Queue、Lease、Checkpoint 与幂等
[ ] Protocol、Streaming、Cancel 与 Backpressure
[ ] Trace、Artifact、Eval 与发布门禁
[ ] Plugin/Skill 的安装、权限、版本和撤销
```

证据记录建议固定为：

```yaml
claim: 支持会话分叉
source_type: official_source
repository: owner/repo
commit: <sha>
path: path/to/file
symbol: fork_thread
verified_at: 2026-09-11
inference: false
```

如果只能从产品界面观察到行为，应标记为黑盒验证，不能写成源码事实。

## 十七、平台反模式

1. **万能超级账号**：所有 Agent 共用一套高权限凭证；
2. **Registry 即执行器**：控制面直接执行用户工具；
3. **只存聊天记录**：没有任务状态、环境版本和副作用日志；
4. **Prompt 发布等于系统发布**：忽略工具、策略和 Runtime 的兼容性；
5. **Sandbox 万能论**：忽略网络、凭证、审批和业务授权；
6. **所有工具全塞 Prompt**：成本高、选择混乱、泄露不必要能力；
7. **多租户只加 tenant_id**：缓存、日志、向量和 Artifact 仍串租户；
8. **用平均成功率掩盖严重事故**：一次越权写入不能被九十九次成功抵消；
9. **无法取消的长任务**：用户停止后 Worker 仍继续产生副作用；
10. **平台化过早**：单个 Agent 尚未形成稳定需求，就先建设复杂 Marketplace。

## 十八、高频系统设计题

1. 设计支持一万家企业的 Agent 平台。
2. 如何让同一个 Agent 同时服务 Web、Slack、邮件和 CLI？
3. 用户身份和 Agent 身份为什么必须分离？
4. Tool Registry 和 MCP Registry 是否应该合并？
5. 如何实现租户级模型路由和成本预算？
6. Agent Marketplace 如何审核恶意 Skill？
7. 长任务如何跨 Worker 恢复？
8. 如何支持 Agent 灰度、回滚和审计？
9. 私有化部署与 SaaS 部署的架构差异是什么？
10. 如何避免平台成为拥有全部企业权限的超级账号？
11. 为什么 AgentOS 要拆成控制、数据、执行和治理四个平面？
12. Agent Release 应包含哪些可版本化组件？
13. 如何设计 Agent 事故的 Kill Switch 和证据保全？
14. MCP Registry 与 Tool Registry 如何共享发现能力但保持边界？
15. 如何用源码证据证明一个平台真的支持取消、恢复与回滚？
