# generic_agent — 通用 ReAct Agent 框架

`generic_agent` 是一个领域无关的 ReAct (Reasoning + Acting) Agent 框架，从 Vibe-Trading 交易 Agent
的核心循环中提取并泛化而来。你只需定义**工具**和**系统提示词**，即可为任意领域创建智能 Agent
——日志分析、DevOps 运维、代码审查、数据管道监控等，无需修改框架代码。

```
┌─────────────────────────────────────────────────────┐
│                  Your Application                    │
│  ┌──────────┐  ┌──────────────────────────────────┐ │
│  │  Tools    │  │  System Prompt (domain-specific) │ │
│  └────┬─────┘  └──────────────┬───────────────────┘ │
│       └──────────┬────────────┘                      │
│                  ▼                                   │
│  ┌─────────────────────────────────────────────────┐ │
│  │              generic_agent                       │ │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │ │
│  │  │AgentLoop │  │Context   │  │ToolRegistry    │  │ │
│  │  │(ReAct)   │◄─│Builder   │◄─│(BaseTool)      │  │ │
│  │  └────┬─────┘  └──────────┘  └───────────────┘  │ │
│  │       │  ┌──────────┐  ┌──────────┐             │ │
│  │       └──│Trace     │  │Memory    │             │ │
│  │          │Writer    │  │(Workspace)│             │ │
│  │          └──────────┘  └──────────┘             │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │           Infrastructure Layer                   │  │
│  │  ChatLLM │ RunStateStore │ Progress │ Content   │  │
│  │          │               │ Heartbeat │ Filter   │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 目录

- [架构概览](#架构概览)
- [核心组件](#核心组件)
  - [AgentLoop](#1-agentloop)
  - [ContextBuilder](#2-contextbuilder)
  - [BaseTool / ToolRegistry](#3-basetool--toolregistry)
  - [WorkspaceMemory](#4-workspacememory)
  - [TraceWriter](#5-tracewriter)
- [五层上下文管理](#五层上下文管理)
- [工具执行模型](#工具执行模型)
- [事件系统](#事件系统)
- [快速开始：30 秒集成](#快速开始30-秒集成)
- [完整示例：日志分析 Agent](#完整示例日志分析-agent)
- [环境变量配置](#环境变量配置)
- [Return 结果格式](#result-结果格式)
- [与原始交易 Agent 的对比](#与原始交易-agent-的对比)
- [高级用法](#高级用法)

---

## 架构概览

`generic_agent` 的核心是一个 **ReAct 循环**：LLM 思考 → 调用工具 → 获取结果 → 继续推理
→ 最终给出答案。整个过程支持：

- **流式输出** — LLM 的 token 逐块通过回调推送，支持实时 UI
- **五层上下文压缩** — 从零成本的字符串裁剪到 LLM 摘要，自动防止 token 溢出
- **读写批处理** — 连续只读工具自动并行执行，写操作保持串行
- **工具超时隔离** — 只读工具超时会被 kill，写工具超时发警告但等待完成
- **心跳/进度事件** — 长时间运行的工具定期推送进度到 UI
- **崩溃安全追踪** — 每条 JSONL 实时 flush，异常退出不丢数据

### 一次 ReAct 迭代的完整流程

```
用户输入
    │
    ▼
ContextBuilder.build_messages()
    ├── 注入系统提示词（含工具列表、状态摘要）
    ├── 追加历史对话（如有）
    └── 可选注入持久记忆（如有）
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  AgentLoop.run()  循环迭代开始                         │
│                                                       │
│  1. 检查取消信号 → 收到则退出                           │
│  2. 检查后台任务通知 → 追加到消息列表                    │
│  3. 估计 token 数 → 按阈值触发 L1/L2/L3 压缩             │
│  4. 接近 max_iterations → 注入收尾提示                  │
│  5. 调用 llm.stream_chat() 流式推理                     │
│     ├── 流式 token 通过 _on_text_chunk 推送             │
│     ├── reasoning 通过 _on_reasoning_chunk 推送         │
│     └── 最后迭代自动移除工具定义（强制纯文本回答）         │
│  6. 内容过滤检查 → 命中则跳过本轮                        │
│  7. 判断结果类型：                                      │
│     ├── 纯文本 → 写 trace、记录 final_content、退出循环   │
│     └── 工具调用 → 进入 _process_tool_calls             │
│                                                       │
│  _process_tool_calls()                                 │
│  ├── 检测 compact 工具 → 标记压缩请求                    │
│  ├── 检测重复调用 → 跳过（非可重复工具）                  │
│  └── 分派执行：                                        │
│      ├── 单工具 → _execute_single()                    │
│      └── 多工具 → _batch_execute()                     │
│          ├── 连续只读 → 线程池并行执行                    │
│          └── 写工具 → 串行执行                          │
│                                                       │
│  8. 如请求了 compact → 执行 _auto_compact()             │
│  9. 回到步骤 1（最多 max_iterations 次）                │
│                                                       │
└──────────────────────────────────────────────────────┘
    │
    ▼
返回 {
    "status": "success" | "failed" | "cancelled",
    "content": "最终回答文本",
    "run_dir": "/path/to/run/dir",
    "react_trace": [...],   // 工具调用记录
    "iterations": N,
    ...
}
```

---

## 核心组件

### 1. AgentLoop

ReAct 循环的主引擎。接收用户消息，驱动 LLM 推理与工具执行交替进行。

```python
from generic_agent import AgentLoop

loop = AgentLoop(
    registry=registry,            # ToolRegistry — 已注册好的工具集
    llm=chat_llm,                 # ChatLLM — LLM 客户端
    memory=WorkspaceMemory(),      # （可选）工作空间内存
    context_builder=context,       # （可选）自定义 ContextBuilder
    event_callback=my_callback,    # （可选）事件回调
    max_iterations=50,            # （可选）最大迭代次数
)
```

**关键方法：**

| 方法 | 说明 |
|---|---|
| `run(user_message, history, session_id)` | 同步执行 ReAct 循环，返回结果 dict |
| `cancel()` | 线程安全地取消正在执行的循环 |

#### AgentLoop.__init__ 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `registry` | `ToolRegistry` | (必填) | 注册了所有可用工具的注册表 |
| `llm` | `ChatLLM` | (必填) | LLM 客户端，负责流式推理 |
| `memory` | `WorkspaceMemory` | `None` | 不传则自动创建空实例 |
| `event_callback` | `Callable` | `None` | 接收所有运行时事件（流式 token、工具结果等） |
| `max_iterations` | `int` | `50` | 最大 ReAct 循环次数 |
| `persistent_memory` | `Any` | `None` | 跨会话持久记忆（实现 `find_relevant` 接口即可） |
| `context_builder` | `ContextBuilder` | `None` | 不传则用默认模板自动创建 |

#### run() 返回值

```python
{
    "status": "success" | "failed" | "cancelled",
    "run_dir": "/path/to/agent/runs/20260713_120000_abc123/",
    "run_id": "20260713_120000_abc123",
    "content": "最终的文本回答...",
    "react_trace": [
        {"type": "tool_call", "tool": "search_logs", "result_preview": "..."},
        {"type": "answer", "content": "最终回答..."}
    ],
    "iterations": 5,
    "max_iterations": 50,
    "reason": "失败原因（仅 status=failed 时存在）",
    "content_filter_warnings": [...]   # 可选，内容过滤统计
}
```

---

### 2. ContextBuilder

负责构建发送给 LLM 的消息列表，核心是可配置的**系统提示词模板**。

```python
from generic_agent import ContextBuilder

context = ContextBuilder(
    registry=registry,
    memory=memory,
    system_prompt_template=MY_PROMPT_TEMPLATE,  # 你定义的模板
    tool_descriptions_formatter=None,            # 可选：自定义工具描述格式器
    persistent_memory=None,                      # 可选：持久记忆
)
```

#### 系统提示词模板

模板支持以下 `{format}` 占位符：

| 占位符 | 说明 | 示例值 |
|---|---|---|
| `{tool_count}` | 工具数量 | `5` |
| `{tool_descriptions}` | 自动生成的工具 Markdown 描述 | `### grep_logs\nSearch...` |
| `{memory_summary}` | 工作空间状态摘要 | `- run_dir: /path` |
| `{current_datetime}` | 当前时间 | `Monday, July 13, 2026 14:30 (local)` |

模板示例：

```python
MY_PROMPT_TEMPLATE = """\
You are a database admin agent with {tool_count} tools.

## Tools

{tool_descriptions}

## State

{memory_summary}

## Guidelines

- Always check table size before running queries.
- Report row counts and execution time.

## Current Date & Time

Today is {current_datetime}.
"""
```

默认模板（`DEFAULT_SYSTEM_PROMPT`）是一个通用的 "helpful AI agent" 模板，不传自定义模板时使用。

---

### 3. BaseTool / ToolRegistry

#### BaseTool

所有工具必须继承 `BaseTool` 并实现 `execute()` 方法。

```python
from generic_agent.tools import BaseTool

class MyTool(BaseTool):
    name = "my_tool"                        # 工具名（LLM 调用时使用）
    description = "Does something useful."  # 描述
    parameters = {                           # JSON Schema 参数定义
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input value"},
        },
        "required": ["input"],
    }
    repeatable: bool = False     # 是否允许在一次运行中重复调用
    is_readonly: bool = True     # 是否是只读操作（影响并行调度）

    def execute(self, **kwargs) -> str:
        # 实现工具逻辑
        # 返回值必须是 JSON 字符串，遵循 {"status": "ok", ...} 或 {"status": "error", ...} 格式
        return json.dumps({"status": "ok", "result": 42})
```

**关键属性：**

| 属性 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 工具唯一标识符 |
| `description` | `str` | `""` | 显示给 LLM 的描述 |
| `parameters` | `dict` | `{}` | JSON Schema 格式的参数定义 |
| `repeatable` | `bool` | `False` | `True` 则允许同一轮中重复调用 |
| `is_readonly` | `bool` | `True` | `True` 表示无副作用，可并行 |

**`execute()` 返回值约定**：

- 成功：`{"status": "ok", ...data...}`
- 失败：`{"status": "error", "error": "错误信息"}`
- 返回类型必须是字符串（通常为 JSON）

#### ToolRegistry

```python
from generic_agent import ToolRegistry

registry = ToolRegistry()
registry.register(MyTool())
registry.register(AnotherTool())

# 框架内部使用：
registry.get_definitions()  # → OpenAI function-calling 格式列表
registry.execute("my_tool", {"input": "value"})  # → str
```

---

### 4. WorkspaceMemory

在一次 AgentLoop.run() 调用期间共享的工作空间状态。跨会话持久化由
`PersistentMemory` 处理。

```python
from generic_agent import WorkspaceMemory

memory = WorkspaceMemory()
memory.run_dir = "/path/to/run"
memory.increment("my_tool")     # 工具调用计数

memory.to_summary()  # → "- run_dir: /path\n- counters: my_tool=1"
```

| 属性/方法 | 类型 | 说明 |
|---|---|---|
| `run_dir` | `Optional[str]` | 当前运行目录 |
| `counters` | `Dict[str, int]` | 工具调用计数器 |
| `increment(key)` | `int` | 增加计数器，返回新值 |
| `to_summary()` | `str` | 生成 LLM 可读的状态摘要 |

---

### 5. TraceWriter

崩溃安全的 JSONL 追踪写入器。每条记录一行 JSON，实时 flush，大字段自动卸载到侧边文件。

```python
from generic_agent import TraceWriter

trace = TraceWriter(dir_path)
trace.write_text_entry({"type": "start"}, field="prompt", value="user msg")
trace.write_tool_result(call_id="call_1", result="...", tool_name="my_tool",
                        status="ok", elapsed_ms=150, iteration=1)
trace.close()
```

追踪文件位于 `run_dir/trace.jsonl`，大文本字段存储在 `run_dir/*.blob` 文件中。

---

## 五层上下文管理

这是 `generic_agent` 的核心能力之一——在长时间运行的对话中自动管理 token 预算。

```
Token 预算线
    │
    ├── Layer 1: microcompact（20K tokens）
    │   触发：超过 MICROCOMPACT_THRESHOLD（TOKEN_THRESHOLD × 0.5）
    │   动作：清除旧工具结果的 content 为 "[cleared]"
    │   成本：零（纯字符串操作）
    │   保留：最近 KEEP_RECENT=3 条结果不动
    │
    ├── Layer 2: context_collapse（28K tokens）
    │   触发：超过 COLLAPSE_THRESHOLD（TOKEN_THRESHOLD × 0.7）
    │   动作：折叠长文本中间部分，保留头尾
    │   成本：零（纯字符串操作）
    │   阈值：>2400 字符且非旧消息才折叠
    │   保留：最近 6 条消息不动
    │
    ├── Layer 3: auto_compact（40K tokens）
    │   触发：超过 TOKEN_THRESHOLD
    │   动作：LLM 生成结构化摘要 + 保留 ~20K tokens 尾部消息
    │   成本：一次 LLM 调用
    │   能力：token-budget tail 保护、工具对修复
    │
    ├── Layer 4: compact tool（模型主动触发）
    │   触发：LLM 调用名为 "compact" 的工具
    │   动作：标记压缩请求，当轮工具执行完毕后执行 L3
    │   用途：让模型能在上下文中自行决定何时需要压缩
    │
    └── Layer 5: iterative update（N 次压缩）
         触发：已有 previous_summary 时的后续 L3 压缩
         动作：增量更新摘要而非从头生成
         优势：信息零衰减，前 N 轮摘要内容完整保留
```

每层依次检查自己的阈值，只有超过阈值才触发。这意味着短对话完全跳过所有压缩开销。

**Token-budget tail**（L3 专属）：L3 压缩时并非压缩所有历史——它从消息列表末尾
往前扫描，保留约 20K tokens 的最近消息（含工具结果）不动，只对更早的部分做摘要。
这保证了模型始终能访问到最近的工具执行结果。

**工具对修复**（`_fix_tool_pairs`）：压缩后自动检查 tool_call/tool_result 的配对关系，
删除孤立的 result，为孤立的 call 补插桩结果，避免 LLM 看到残缺的调用对。

---

## 工具执行模型

`generic_agent` 根据工具的 `is_readonly` 属性自动调度并行或串行执行。

```
工具调用列表
    │
    ├── 只读工具 1 ───┐
    ├── 只读工具 2 ───┤──→ ThreadPoolExecutor (最多 8 线程)
    ├── 只读工具 3 ───┘
    │
    ├── 写工具 4 ─────────→ 串行执行（等待并行批次完成）
    │
    ├── 只读工具 5 ───┐
    └── 只读工具 6 ───┘──→ 下一批线程池
```

**只读工具超时**：超过 `VIBE_TRADING_TOOL_TIMEOUT_SECONDS`（默认 1800s）后，
worker 线程被分离（结果丢弃），返回超时错误。资源泄露通过后台清理机制处理。

**写工具超时**：写工具不会被 kill（防止数据损坏）。超过超时时间后发送警告，
但一直等待工具完成。

### 重复调用防护

非 `repeatable` 的工具在成功执行一次后，同一轮中再次调用会被自动跳过：

```json
{"skipped": true, "reason": "my_tool already completed successfully."}
```

---

## 事件系统

`AgentLoop` 在运行过程中通过 `event_callback` 推送各类事件，可用于驱动实时 UI
或在运行时监控 Agent 状态。

```python
def my_callback(event_type: str, data: dict):
    if event_type == "text_delta":
        print(data["delta"], end="")
    elif event_type == "tool_call":
        print(f"\n[Tool] {data['tool']}({data['arguments']})")
    elif event_type == "tool_result":
        print(f"\n[Result] {data['tool']}: {data['status']} ({data['elapsed_ms']}ms)")

loop = AgentLoop(..., event_callback=my_callback)
```

### 事件类型一览

| 事件类型 | 触发时机 | 关键数据 |
|---|---|---|
| `text_delta` | LLM 输出每个 token | `delta`, `iter` |
| `reasoning_delta` | 推理过程 token | `iter`, `chars` |
| `thinking_done` | 推理/思考结束 | `iter`, `content` (前500 chars) |
| `tool_call` | 工具被调用 | `tool`, `arguments`, `iter` |
| `tool_result` | 工具执行完成 | `tool`, `status`, `elapsed_ms`, `preview` |
| `tool_progress` | 工具报告进度 | `tool`, `stage`, `current`, `total`, `message` |
| `tool_heartbeat` | 工具运行中心跳 | `tool`, `elapsed_s` |
| `llm_usage` | 每次 LLM 响应 | `input_tokens`, `output_tokens`, `total_tokens`, `iter` |
| `compact` | 上下文压缩 | `tokens_before`, `summary` (前200 chars) |
| `goal.updated` | 目标状态更新 | `goal`, `snapshot` |
| `stream_reset` | 流式重试 | `iter`, `reason`, `provider`, `model` |

---

## 快速开始：30 秒集成

以下是一个最小可运行的 Agent 示例：

```python
import json
from generic_agent import (
    AgentLoop, ContextBuilder, SimpleLLM, ToolRegistry, WorkspaceMemory,
)
from generic_agent.tools import BaseTool

# 1. 定义工具
class HelloTool(BaseTool):
    name = "hello"
    description = "Say hello to someone."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person to greet"},
        },
        "required": ["name"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps({"status": "ok", "message": f"Hello, {kwargs['name']}!"})

# 2. 注册工具
registry = ToolRegistry()
registry.register(HelloTool())

# 3. 配置 ContextBuilder（使用自定义系统提示词）
context = ContextBuilder(
    registry=registry,
    memory=WorkspaceMemory(),
    system_prompt_template="""You are a greeting agent with {tool_count} tool.

## Tools

{tool_descriptions}

## Guidelines

- Use hello tool to greet people.
- Respond warmly.

## Current Date & Time

Today is {current_datetime}.
""",
)

# 4. 创建 AgentLoop 并运行
llm = SimpleLLM()
loop = AgentLoop(
    registry=registry,
    llm=llm,
    context_builder=context,
    max_iterations=10,
)

result = loop.run("Say hello to Alice and Bob")
print(result["content"])
```

---

## 完整示例：日志分析 Agent

`examples/log_analysis.py` 包含一个完整的日志分析 Agent，定义了一个日志分析工具包：

| 工具 | 功能 |
|---|---|
| `grep_logs(path, pattern, glob)` | 在日志文件中搜索正则表达式模式 |
| `count_log_levels(path, glob)` | 统计 ERROR/WARN/INFO/DEBUG 等级 |
| `tail_logs(path, n)` | 查看文件末尾 N 行 |
| `list_log_files(path, glob)` | 列出目录下日志文件及大小 |

```bash
# 运行日志分析 Agent
cd generic_agent
python -m examples.log_analysis /var/log/myapp
```

你可以将此文件作为模板，替换为自己的工具和系统提示词来创建任意领域的 Agent。

---

## 环境变量配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TOKEN_THRESHOLD` | `40000` | L3 auto_compact 触发阈值（token 数） |
| `VT_HEARTBEAT_INTERVAL_S` | `3.0` | 工具心跳间隔（秒） |
| `VT_REASONING_DELTA_MIN_INTERVAL_S` | `1.0` | reasoning 事件最小节流间隔 |
| `VT_STREAM_RETRY_DELAY_S` | `1.0` | 流式失败重试等待时间 |
| `VIBE_TRADING_TOOL_TIMEOUT_SECONDS` | `1800` | 工具超时（秒），≤0 则无限等待 |
| `CONTENT_FILTER_WARNING_THRESHOLD` | `0.05` | 内容过滤警告阈值 |

---

## Result 结果格式

`AgentLoop.run()` 返回的 dict 结构：

```python
{
    "status": "success" | "failed" | "cancelled",

    # 仅在 status 为 "failed" 时存在：
    "error_code": "provider_stream_error" | "agent_loop_error",
    "reason": "错误描述字符串",

    # status 为 "cancelled" 时：
    "reason": "cancelled by user",

    # 运行信息
    "run_dir": "/path/to/agent/runs/20260713_120000_abc123/",
    "run_id": "20260713_120000_abc123",

    # 最终回答（status=success 时不为空）
    "content": "最终的文本回答...",

    # 工具调用追踪（按时间顺序）
    "react_trace": [
        {"type": "tool_call", "tool": "search_logs",
         "result_preview": '{"status":"ok","matches":3,...}'},
        {"type": "answer", "content": "分析结果..."}
    ],

    # 迭代统计
    "iterations": 5,
    "max_iterations": 50,

    # 可选：内容过滤统计
    "content_filter_warnings": [
        "Content filter triggered 2 times (40% of iterations)"
    ]
}
```

**`status` 的可能值及含义**：

| status | 含义 |
|---|---|
| `success` | Agent 生成了最终回答（`content` 非空） |
| `failed` | 执行出错：LLM 返回空响应、达到最大迭代次数无结果、内容过滤断路器触发 |
| `cancelled` | 用户调用 `AgentLoop.cancel()` 中断执行 |

---

## 高级用法

### 自定义工具描述格式器

默认的 `_format_tool_descriptions` 生成 Markdown 格式。你可以传入自定义格式器：

```python
def my_formatter() -> str:
    lines = []
    for tool in registry._tools.values():
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)

context = ContextBuilder(
    registry=registry,
    memory=memory,
    system_prompt_template=MY_PROMPT,
    tool_descriptions_formatter=my_formatter,
)
```

### 集成 PersistentMemory（跨会话记忆）

`ContextBuilder` 接受一个 `persistent_memory` 参数。只需实现 `find_relevant()`
接口即可：

```python
class MyMemory:
    def find_relevant(self, query: str, max_results: int = 3):
        # 返回类似 namedtuple 的对象列表，每个对象包含 .title, .memory_type, .body
        ...

context = ContextBuilder(
    registry=registry,
    memory=memory,
    persistent_memory=MyMemory(),
)
```

### 使用历史对话

```python
result = loop.run(
    user_message="分析今天的错误日志",
    history=[
        {"role": "user", "content": "昨天你找到了一些数据库连接超时"},
        {"role": "assistant", "content": "是的，主要来自 app-server-3"},
    ],
    session_id="chat-session-001",
)
```

### 取消长时间运行的任务

```python
import threading

loop = AgentLoop(...)

def on_timeout():
    loop.cancel()

timer = threading.Timer(300.0, on_timeout)
timer.start()
result = loop.run("处理 100 个日志文件")
timer.cancel()
```

---

## 项目结构

```
generic_agent/
├── __init__.py             # 统一导出入口
├── context.py              # ContextBuilder — 可配置系统提示词的消息构建器
├── loop.py                 # AgentLoop — 核心 ReAct 循环 + LLMClient 基类
├── memory.py               # WorkspaceMemory — 运行时状态
├── progress.py             # 心跳/进度事件基础设施
├── simple_llm.py           # SimpleLLM — OpenAI 兼容 LLM 客户端
├── tools.py                # BaseTool / ToolRegistry
├── trace.py                # TraceWriter — 崩溃安全追踪
├── README.md               # 本文件
└── examples/
    ├── __init__.py
    └── log_analysis.py     # 日志分析 Agent 完整示例
```