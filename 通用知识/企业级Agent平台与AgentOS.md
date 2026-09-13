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

## 十一、高频系统设计题

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
