# MCP 与工具生态面试题

## 一、MCP 核心概念

### 什么是 MCP（Model Context Protocol）？

MCP（Model Context Protocol）是由 Anthropic 于 2024 年底开源的一套**标准化协议**，旨在解决 LLM 应用与外部工具/数据源之间的集成碎片化问题。它定义了一种通用的通信方式，让 AI 模型能够以统一接口发现、调用和管理外部工具与资源。

**核心问题：为什么需要 MCP？**

在 MCP 出现之前，每个 AI 应用都需要为每个外部服务编写定制化的集成代码，形成 M×N 的集成困境：

```
传统模式（M×N 集成）：
┌──────────┐     ┌──────────────┐
│ App 1    │────→│ GitHub API   │  每个 App 都要写
│ App 2    │────→│ Slack API    │  每个 API 的
│ App 3    │────→│ Database     │  集成代码
│ ...      │────→│ ...          │
└──────────┘     └──────────────┘
  M 个应用   ×   N 个服务 = M×N 集成

MCP 模式（M+N 集成）：
┌──────────┐     ┌─────┐     ┌──────────────┐
│ App 1    │────→│     │────→│ GitHub Server│
│ App 2    │────→│ MCP │────→│ Slack Server │
│ App 3    │────→│     │────→│ DB Server    │
└──────────┘     └─────┘     └──────────────┘
  M 个应用   +   N 个服务 = M+N 集成
```

类比：MCP 之于 AI 工具调用，就像 USB 之于外设连接——一个标准接口解决所有设备的兼容问题。

### MCP 解决的核心痛点

1. **集成碎片化**：不同模型厂商的 Function Calling 格式不统一，工具开发者要适配多套 API
2. **工具发现困难**：缺乏标准化的工具注册和发现机制，用户需手动配置
3. **上下文管理复杂**：工具返回的结果如何注入模型上下文，缺乏统一规范
4. **安全边界模糊**：工具的权限控制、用户授权没有标准化方案
5. **生态孤岛**：每个平台的插件/工具不可复用，开发者重复造轮子

## 二、MCP 架构详解

### 四大核心组件

```
┌─────────────────────────────────────────────────┐
│                   MCP Host                       │
│  (Claude Desktop / IDE / Agent Framework)        │
│                                                  │
│  ┌──────────────┐    ┌──────────────┐           │
│  │  MCP Client  │    │  MCP Client  │           │
│  │  (实例 1)    │    │  (实例 2)    │           │
│  └──────┬───────┘    └──────┬───────┘           │
└─────────┼───────────────────┼───────────────────┘
          │ Transport         │ Transport
          │ (stdio/SSE/HTTP)  │
    ┌─────┴──────┐     ┌─────┴──────┐
    │ MCP Server │     │ MCP Server │
    │ (GitHub)   │     │ (Database) │
    └────────────┘     └────────────┘
```

**1. Host（宿主）**
- 运行 LLM 的应用程序，如 Claude Desktop、Cursor、自定义 Agent 框架
- 负责管理多个 MCP Client 实例的生命周期
- 控制安全策略和权限边界
- 示例：Claude Desktop 就是一个 Host，它可以同时连接多个 MCP Server

**2. Client（客户端）**
- Host 内部的协议实例，每个 Client 维护与一个 Server 的 1:1 连接
- 负责协议协商（capability negotiation）、消息路由
- 管理请求/响应的生命周期

**3. Server（服务端）**
- 对外暴露工具（Tools）、资源（Resources）和提示模板（Prompts）
- 一个 Server 可以提供多个工具，如 GitHub Server 提供 create_issue、list_repos 等
- 轻量化设计，专注于单一数据源或服务的封装

**4. Transport（传输层）**
- **stdio**：通过标准输入/输出通信，适合本地进程间通信，最常用
- **HTTP + SSE（Server-Sent Events）**：适合远程通信，Server 通过 SSE 推送消息
- **Streamable HTTP**：2025 年新增的传输方式，替代 SSE，更灵活

### MCP 三大能力原语

| 原语 | 说明 | 控制方 | 示例 |
|------|------|--------|------|
| **Tools** | 模型可调用的函数 | 模型决定是否调用 | `create_issue()`, `query_db()` |
| **Resources** | 模型可读取的数据源 | 应用程序控制 | 文件内容、数据库 schema、API 文档 |
| **Prompts** | 预定义的提示模板 | 用户触发 | "总结这个 PR"、"分析这段代码" |

## 三、MCP vs Function Calling vs Plugin

### 三者对比

| 维度 | Function Calling | Plugin（如 ChatGPT） | MCP |
|------|-----------------|---------------------|-----|
| **定义层** | 模型推理层 | 应用平台层 | 通信协议层 |
| **标准化** | 各厂商格式不同 | 平台私有 | 开放标准 |
| **工具发现** | 开发者手动注册 | 平台商店 | 协议内建 `tools/list` |
| **可移植性** | 绑定特定模型 | 绑定特定平台 | 跨模型、跨平台 |
| **传输方式** | HTTP API 内嵌 | HTTP API | stdio / HTTP / SSE |
| **状态管理** | 无 | 会话级 | 连接级，支持有状态 |
| **安全模型** | API Key | OAuth | 协议内建权限协商 |
| **典型场景** | 单模型工具调用 | 消费级产品 | 企业级 Agent 集成 |

**关键区别：**
- **Function Calling** 是"模型怎么调工具"——关注的是 LLM 输出结构化的工具调用指令
- **Plugin** 是"平台怎么管工具"——关注的是工具在特定平台的注册、分发、计费
- **MCP** 是"工具怎么接入"——关注的是工具与 AI 应用之间的通信标准

三者不是竞争关系，而是互补：MCP Server 暴露工具 → Host 通过 Function Calling 让模型决定调用哪个工具 → 平台以 Plugin 形式分发。

## 四、MCP Server 开发实战

### Python（FastMCP）

```python
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气信息。

    Args:
        city: 城市名称，如 "北京"、"上海"
        unit: 温度单位，celsius 或 fahrenheit
    """
    # 实际实现会调用天气 API
    return f"{city} 当前温度 25°C，晴天"

@mcp.resource("config://app-settings")
def get_settings() -> str:
    """返回应用配置信息"""
    return '{"theme": "dark", "language": "zh-CN"}'

@mcp.prompt()
def summarize_weather(city: str) -> str:
    """生成天气摘要提示"""
    return f"请根据以下天气数据，为{city}生成一段简洁的天气摘要。"

# 运行
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### TypeScript（MCP SDK）

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "weather-server",
  version: "1.0.0",
});

server.tool(
  "get_weather",
  "获取指定城市的当前天气",
  { city: z.string(), unit: z.enum(["celsius", "fahrenheit"]).default("celsius") },
  async ({ city, unit }) => {
    return {
      content: [{ type: "text", text: `${city} 当前温度 25°C，晴天` }],
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 开发最佳实践

1. **工具描述要精确**：LLM 通过描述决定何时调用，含糊的描述会导致错误调用
2. **参数用 JSON Schema 严格约束**：类型、枚举、默认值、必填项都要明确
3. **错误处理要优雅**：返回结构化错误信息，而非抛异常让连接断开
4. **幂等设计**：同一调用多次执行应产生相同结果，防止重复操作
5. **超时控制**：长时间运行的工具应支持 progress notification

## 五、MCP 在企业级 Agent 中的应用场景

1. **统一工具网关**：企业内部多个 Agent 通过 MCP 接入统一的工具注册中心，避免重复集成
2. **权限隔离**：不同角色的 Agent 通过 MCP 的权限协商机制，访问不同级别的工具
3. **审计追踪**：所有工具调用通过 MCP 标准协议，天然支持日志审计
4. **多模型切换**：工具层与模型层解耦，切换底层 LLM 不需要修改工具集成代码
5. **渐进式迁移**：企业可以逐步将内部 API 封装为 MCP Server，无需一次性重构

## 六、常见面试题与参考答案

### Q1：MCP 是什么？它解决了什么核心问题？
**答**：MCP 是 Anthropic 开源的标准化协议，定义了 AI 应用与外部工具/数据源之间的通信规范。核心解决 M×N 集成碎片化问题——让工具开发者写一次 MCP Server，就能被所有支持 MCP 的 Host 调用，将集成复杂度从 M×N 降为 M+N。

### Q2：MCP 的四大组件分别是什么？各自职责是什么？
**答**：Host（宿主应用，管理生命周期和安全策略）、Client（协议实例，维护与 Server 的 1:1 连接）、Server（暴露 Tools/Resources/Prompts）、Transport（传输层，支持 stdio/HTTP+SSE/Streamable HTTP）。

### Q3：MCP 的三大能力原语是什么？它们的控制方分别是谁？
**答**：Tools（模型控制，模型决定是否调用）、Resources（应用控制，如自动注入上下文）、Prompts（用户控制，用户选择使用的模板）。设计理念是不同的能力由最合适的角色控制。

### Q4：MCP vs Function Calling 的核心区别是什么？
**答**：Function Calling 解决"模型如何输出结构化工具调用指令"，是推理层的能力；MCP 解决"工具如何接入 AI 应用"，是通信协议层的标准。两者互补——MCP Server 暴露工具，模型通过 Function Calling 决定调用哪个。

### Q5：MCP 支持哪些 Transport？各适用什么场景？
**答**：①stdio：通过标准输入输出通信，适合本地进程，启动快、零配置；②HTTP+SSE：适合远程部署，Server 通过 SSE 推送事件；③Streamable HTTP：2025 年新增，无状态 HTTP 请求可选升级为 SSE 流，兼顾简单性和实时性。

### Q6：如何设计一个高质量的 MCP Tool 描述？
**答**：①名称用动词+名词（如 `create_issue`）；②description 精确说明功能、适用场景、限制条件；③参数用 JSON Schema 严格定义类型、枚举、默认值；④返回值结构化；⑤加入使用示例。差的描述会导致 LLM 误调用或不调用。

### Q7：MCP 的安全模型是如何设计的？
**答**：多层安全机制：①协议层的 capability negotiation，Client 和 Server 在连接时协商各自支持的能力；②Host 层的权限控制，Host 决定哪些 Server 可连接、哪些工具可暴露给模型；③Transport 层可叠加 TLS、认证；④用户确认机制（human-in-the-loop），敏感操作需用户授权。

### Q8：如果一个 MCP Server 响应很慢，怎么优化？
**答**：①使用 progress notification 机制，向 Client 报告进度，避免超时；②工具拆分——将耗时操作拆为"提交任务"和"查询结果"两个工具；③缓存常用数据；④Server 端异步处理；⑤Transport 层考虑用 Streamable HTTP 替代 stdio 以支持并发。

### Q9：MCP 中的 Resource 和 Tool 有什么区别？什么时候用 Resource？
**答**：Tool 是模型主动调用的函数，有副作用（如创建 issue）；Resource 是被动提供的数据源（如文件内容、数据库 schema），由应用程序控制注入时机。当数据是"背景信息"而非"操作"时用 Resource，如把项目的 README 作为 Resource 注入上下文。

### Q10：如何实现 MCP Server 的工具权限控制？
**答**：①Server 端：在工具注解中声明所需权限，如 `annotations: {"readOnly": true}`；②Host 端：维护白名单，只暴露授权的工具给模型；③Runtime：敏感工具标记 `destructive`，Host 在调用前要求用户确认；④多租户场景通过 Transport 层携带用户身份，Server 端做鉴权。

### Q11：MCP 的 capability negotiation 是什么？为什么需要它？
**答**：Client 和 Server 在建立连接时互相声明自己支持的功能（如 Server 声明支持 tools、resources；Client 声明支持 sampling）。这样双方只使用共同支持的功能，保证向前兼容——旧 Client 遇到新 Server 不会因为不认识新特性而崩溃。

### Q12：在企业级场景中，如何管理大量 MCP Server？
**答**：①建立统一的 Server 注册中心（类似服务发现）；②用配置文件统一管理 Server 连接信息；③实现健康检查和自动重连；④日志集中收集和审计；⑤版本管理，支持灰度升级；⑥基于角色的访问控制（RBAC），不同 Agent 看到不同的工具集。

### Q13：MCP 的 Sampling 机制是什么？
**答**：Sampling 允许 Server 反过来请求 Host 的 LLM 进行推理。使用场景：Server 在处理复杂任务时需要 LLM 辅助决策（如分析查询结果、生成摘要）。这形成了"LLM ↔ Server"的双向通信，但始终由 Host 控制 LLM 的调用权限，保证安全。

### Q14：如何测试和调试 MCP Server？
**答**：①使用 MCP Inspector（官方调试工具），可视化查看 Server 暴露的工具和资源；②单元测试：直接调用工具函数；③集成测试：用 MCP Client SDK 模拟连接；④日志：Server 通过 `logging` 能力发送日志给 Host；⑤错误注入：测试网络断开、超时等异常情况。

### Q15：MCP 与 OpenAPI/Swagger 有什么关系？
**答**：OpenAPI 定义 REST API 的接口规范，MCP 定义 AI 工具的通信协议。MCP Server 可以包装 OpenAPI 定义的 API——用 OpenAPI 描述 HTTP 接口，用 MCP 让 AI 模型能发现和调用这些接口。社区有工具可以自动将 OpenAPI spec 转为 MCP Server。

### Q16：MCP 协议是有状态还是无状态的？
**答**：MCP 连接是有状态的。Client 和 Server 建立连接后，维护会话状态（如协商的 capabilities、订阅的资源变更通知）。但 Streamable HTTP Transport 支持无状态模式——简单的请求-响应可以不维护长连接，需要时再升级为有状态的 SSE 流。

### Q17：MCP 如何处理工具调用的并发和顺序问题？
**答**：MCP 基于 JSON-RPC 2.0，每个请求有唯一 ID，支持并发发送多个请求。Client 可以同时发起多个工具调用，Server 异步处理并按 ID 返回。顺序依赖由 Host/Agent 框架在上层控制——MCP 协议本身不强制顺序。

### Q18：MCP 生态中有哪些知名的 Server 实现？
**答**：官方和社区提供了大量 Server：GitHub（代码管理）、Filesystem（文件操作）、PostgreSQL/SQLite（数据库）、Brave Search（搜索）、Slack（消息）、Google Drive（文档）等。这些 Server 可以直接在 Claude Desktop、Cursor 等 Host 中使用，体现了 MCP 的即插即用优势。

### Q19：如果让你设计一个 MCP Server，你会如何考虑错误处理？
**答**：①区分可恢复和不可恢复错误——可恢复的返回错误信息让模型重试，不可恢复的返回明确失败原因；②使用 MCP 标准错误码（基于 JSON-RPC 错误码）；③工具返回的 `isError: true` 标记，让 Host 知道这是工具执行失败而非协议错误；④敏感信息脱敏——错误信息不应泄露内部实现细节；⑤超时处理——长任务用 progress notification，避免 Client 超时断连。

### Q20：MCP 的未来发展方向是什么？
**答**：①远程 MCP Server 的标准化（目前以本地 stdio 为主，远程部署正在演进）；②认证授权标准化（OAuth 2.1 集成）；③工具市场/注册中心（类似 npm，方便发现和安装 MCP Server）；④多模态支持（图片、音视频等非文本内容的传输）；⑤与 Agent 框架的深度集成（LangChain、CrewAI 等已开始原生支持 MCP）。
