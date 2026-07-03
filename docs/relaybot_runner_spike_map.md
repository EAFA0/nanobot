# Relaybot Runner Migration Plan (SDK-Native)

Replace the native `AgentRunner` with external coding agent SDKs as the turn
execution engine. Two SDKs cover all target CLIs:

| SDK | Covers | Rationale |
|-----|--------|-----------|
| `openai-codex` (Codex SDK) | codex, coco | coco is built on codex; same app-server protocol |
| `claude-agent-sdk` (Claude Agent SDK) | claude code, relay, seed | relay/seed based on claude code; same CLI subprocess protocol |

No tmux. No paste-verification. No ready-pattern polling. SDKs provide
structured event streams, native multi-turn sessions, and typed results.

---

## 1. Current Fork: Call Chain and Seam Point

### Core path

```
User message → Channel → MessageBus → AgentLoop._dispatch()
  → _process_message() → State machine: BUILD → RUN → SAVE → RESPOND
    → _state_run → _run_agent_loop() [loop.py:726]
      → self.runner.run(AgentRunSpec(...)) [loop.py:874]  ← SEAM HERE
    → _state_save → _save_turn() [loop.py:1690]
    → _state_respond → _assemble_outbound()
  → OutboundMessage → MessageBus → Channel → User
```

### Seam point: `_run_agent_loop` (loop.py:726)

```python
# loop.py:874 — the single call we branch on
result = await self.runner.run(AgentRunSpec(...))

# loop.py:929 — the return contract (consumed by _state_run at line 1585)
return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections
```

`_state_run` (line 1567) unpacks this into `ctx`:
- `ctx.final_content` → `OutboundMessage.content`
- `ctx.all_messages` → `_save_turn(session, all_msgs, save_skip)` — persisted
- `ctx.stop_reason` → `_assemble_outbound` decides `StreamedResponseEvent`
- `ctx.tools_used` → stored in ctx, used for logging/UI
- `ctx.had_injections` → `_assemble_outbound` MessageTool suppression logic

### Session concurrency (already handled)

`_dispatch()` (loop.py:1028) acquires `self._session_locks[session_key]` (an
`asyncio.Lock`) before calling `_process_message`. Two messages for the same
session will never run concurrently. **SDK session cache does NOT need its own
lock** — the dispatch layer guarantees serial access.

### `/stop` mechanism (already handled at task level)

`/stop` → `cmd_stop` (builtin.py:136) → `_cancel_active_tasks(key)` (loop.py:695)
→ `task.cancel()` + `await t`. The running turn gets `CancelledError` at the
next `await` point. `_state_save` and `_state_respond` are skipped (task is
cancelled). **SDK runner must handle `CancelledError` to clean up subprocess
state** (see §6 错误与中断处理).

---

## 2. SDK Architecture

Both SDKs use the same pattern: **spawn a CLI binary subprocess, communicate
over stdin/stdout with a structured JSON protocol.**

```
nanobot SDKRunner
  → spawns codex app-server / claude CLI subprocess (one or many)
  → JSON-RPC / control protocol over stdio
  → subprocess makes real API calls (OpenAI / Anthropic), respects proxy env vars
  → events stream back (delta, tool, completion, usage)
```

### Codex SDK (`openai-codex`)

```python
from openai_codex import AsyncCodex, CodexConfig, Sandbox, ApprovalMode

async with AsyncCodex(config=CodexConfig(
    env={"HTTPS_PROXY": "http://10.3.42.223:8989"},
)) as codex:
    thread = await codex.thread_start(
        model="gpt-4o",
        sandbox=Sandbox.workspace_write,
        approval_mode=ApprovalMode.auto_review,
        cwd="/path/to/workspace",
    )
    # Streaming turn
    turn = await thread.turn("Hello")
    async for event in turn.stream():
        if event.method == "agentMessage/delta":
            print(event.payload.delta, end="", flush=True)
        elif event.method == "item/completed":
            item = event.payload.item.root  # CommandExecutionThreadItem etc.
    result = await turn.run()
    # result.final_response, result.items, result.usage, result.status
```

**Key properties:**
- `AsyncCodex` = one `codex app-server` subprocess, supports multiple threads
- `Thread` = conversation state, identified by `thread.id` (can be resumed later)
- `TurnHandle.stream()` = async iterator of typed events
- `TurnHandle.interrupt()` = stop in-flight turn
- `TurnResult.status`: `completed` | `interrupted` | `failed`

**Event types:** `agentMessage/delta`, `commandExecution/output/delta`,
`fileChange/output/delta`, `item/started`, `item/completed`, `turn/completed`,
`turn/diffUpdated`, `turn/planUpdated`, `thread/tokenUsageUpdated`,
`reasoning/text/delta`, `reasoningSummary/text/delta`, `mcpToolCall/progress`.

**ThreadItem types:** `agentMessage`, `commandExecution` (shell), `fileChange`,
`mcpToolCall`, `dynamicToolCall`, `webSearch`, `plan`, `reasoning`,
`contextCompaction`, `collabAgentToolCall`.

### Claude Agent SDK (`claude-agent-sdk`)

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async with ClaudeSDKClient(options=ClaudeAgentOptions(
    model="claude-sonnet-4-5",
    cwd="/path/to/workspace",
    permission_mode="acceptEdits",
    env={
        "ANTHROPIC_API_KEY": "...",
        "HTTPS_PROXY": "http://10.3.42.223:8989",
    },
    include_partial_messages=True,  # streaming
    max_turns=200,
)) as client:
    await client.query("Hello")
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"\n[Tool: {block.name}]")
                elif isinstance(block, ThinkingBlock):
                    print(f"\n[Thinking...]")
        elif isinstance(msg, ResultMessage):
            print(f"\n[Done: {msg.num_turns} turns]")
```

**Key properties:**
- `ClaudeSDKClient` = one `claude` CLI subprocess, holds conversation state
- Multi-turn: subsequent `client.query()` + `receive_response()` carry context
- `client.interrupt()` = stop running turn
- Session persistence: `ClaudeAgentOptions(session_id="uuid")` for named sessions

**Message types:** `UserMessage`, `AssistantMessage` (content: `TextBlock`,
`ThinkingBlock`, `ToolUseBlock`), `SystemMessage` (task/hook lifecycle),
`ResultMessage` (end-of-turn: `total_cost_usd`, `num_turns`, `stop_reason`,
`duration_ms`, `usage`, `is_error`).

---

## 3. Design Decisions (Resolved)

### 决策 1: Context 注入策略 ✅

**只传递用户消息文本**给 SDK agent。nanobot 构建的 system prompt、runtime context、
skills 文档、会话历史全部不注入。

**理由**：relaybot 定位是"外部 coding agent 的网关"，外部 agent 自治管理 context。
SDK thread/client 持有自己的会话历史。nanobot 的 `ContextBuilder` 输出对 SDK agent
来说是噪音（格式不匹配、还可能干扰 agent 自身的 system prompt）。

**替代方案（Phase 2+）**：如果需要给 SDK agent 注入指令，走 SDK 自己的配置：
- Codex: `codex_base_instructions` / `developer_instructions`
- Claude: `claude_system_prompt`
- 或者在用户消息前面加前缀（简单粗暴但有效）

**媒体/图片**：Phase 1 只传文本。如果 `initial_messages` 里检测到 image content block，
记 warning log 并跳过。Phase 2 补媒体传递（两个 SDK 都原生支持图片输入）。

### 决策 2: 架构层数 ✅

**两层架构**（基类 + 子类），不引入独立 adapter 协议层。

```
SDKRunner (base, shared logic)
  ├── CodexSDKRunner (codex-specific: AsyncCodex, event parsing, thread mgmt)
  └── ClaudeSDKRunner (claude-specific: ClaudeSDKClient, message parsing)
```

Session 管理就是子类里的 `dict[str, _SessionEntry]`，不需要独立 cache 类。
不需要 `AgentSDKAdapter` protocol。

### 决策 3: Model 配置 ✅

**relaybot 不配置 model**。`spec.model`（来自 `self.model`）原样传给 SDK runner，
由 SDK runner 决定是否使用、怎么用：
- Codex runner: `thread.turn(prompt, model=spec.model or default_model)` — 如果 `spec.model`
  是 codex 认识的名字（如 `"gpt-4o"`）就用，否则 fallback 到 config 里的 `codex_model`
- Claude runner: 同理，`spec.model` 传给 turn 或忽略

`/model` 命令对 SDK backend 来说是"高级用户功能"——用户需要知道 SDK 接受的 model 名。
不在 `/backend` 命令里做 model name 映射。

### 决策 4: SDK 客户端生命周期 ✅

**Codex**: 一个共享 `AsyncCodex` 实例（gateway 启动时创建，关闭时销毁），
多 thread 复用。因为 codex app-server 设计为多 thread 服务。

**Claude**: 每个 session_key 一个 `ClaudeSDKClient`（因为 client 持有会话状态）。
Session cache 管理生命周期。

**实现方式**：不使用 `async with` context manager（它要求 enter/exit 在同一个 scope）。
改为手动 `await client.__aenter__()` 创建，`await client.__aexit__(None, None, None)` 销毁。
代码注释说明原因。

### 决策 5: Session 过期清理 ✅

挂在 nanobot 现有的 auto-compact 检查周期上。`run()` 主循环已经每 60 秒检查
session TTL，顺便调用 `self._sdk_runner.evict_stale()` 清理过期 SDK session。

不需要额外定时器。

---

## 4. Module Layout

```
nanobot/agent/
  runner.py                          # UNCHANGED — native AgentRunner + dataclasses
  sdk_runner/
    __init__.py                      # exports SDKRunner, CodexSDKRunner, ClaudeSDKRunner
    base.py                          # SDKRunner base class
    codex.py                         # CodexSDKRunner
    claude.py                        # ClaudeSDKRunner
```

No `session_cache.py` (inline in subclasses). No `fake_adapter.py` (inline in test).

---

## 5. SDKRunner Base Class

### Shared logic in `base.py`

```python
class SDKRunner:
    """Base for SDK-based runners. Implements .run(spec) → AgentRunResult.

    Subclasses implement _run_turn() for SDK-specific execution.
    """

    # Subclass overrides
    backend_name: str  # "codex-sdk" or "claude-sdk"

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        """Entry point — same contract as AgentRunner.run()."""
        # 1. Extract user prompt from spec.initial_messages
        # 2. Resolve cwd from spec.workspace
        # 3. Build hook context, call spec.hook.before_run if hook
        # 4. Wire streaming callbacks
        # 5. try: result = await self._run_turn(...)
        #    except CancelledError: self._handle_cancel(...); raise
        #    except Exception as exc: return error result
        # 6. Build AgentRunResult from turn result
        # 7. Call spec.hook.after_run if hook
        # 8. Return AgentRunResult

    @abstractmethod
    async def _run_turn(
        self,
        session_key: str,
        prompt: str,
        *,
        model: str | None,
        cwd: str,
        on_delta: Callable[[str], Awaitable[None]],
        on_tool_start: Callable[[str, dict], Awaitable[None]],  # name, input
        on_tool_end: Callable[[str, bool, str], Awaitable[None]],  # name, success, preview
        on_reasoning: Callable[[str], Awaitable[None]],
    ) -> _TurnResult:
        """Execute one turn. Subclass implements."""
        ...

    @abstractmethod
    async def evict_stale(self, idle_timeout_s: float) -> int:
        """Remove idle sessions. Return count evicted."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up all SDK resources (subprocesses)."""
        ...
```

```python
@dataclass
class _TurnResult:
    """Internal turn result (subclass → base class)."""
    final_content: str | None
    tools_used: list[str]
    tool_events: list[dict[str, str]]   # nanobot-compatible {name, status, call_id}
    usage: dict[str, int]               # prompt_tokens, completion_tokens, cached_tokens
    stop_reason: str                    # "completed" | "max_iterations" | "error" | "interrupted"
    error: str | None
    messages: list[dict[str, Any]]      # user+assistant pairs for session save
```

### User message extraction

```python
def _extract_user_prompt(self, messages: list[dict]) -> str | None:
    """Extract text from last user message in initial_messages.

    Returns None if no user message found (should not happen in normal flow).
    Logs warning if image content blocks are detected (Phase 1: not supported).
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Anthropic/OpenAI format: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
                texts = [b.get("text", "") for b in content if b.get("type") == "text"]
                images = [b for b in content if b.get("type") in ("image_url", "image")]
                if images:
                    logger.warning("SDK runner received image input — images not supported in Phase 1, ignoring")
                return "\n".join(texts) if texts else None
    return None
```

### Hook wiring in `run()`

The SDK runner translates SDK-native events into `AgentHook` lifecycle calls
so that `AgentProgressHook` produces correct WebUI output:

```python
async def run(self, spec: AgentRunSpec) -> AgentRunResult:
    hook = spec.hook
    prompt = self._extract_user_prompt(spec.initial_messages)
    cwd = str(spec.workspace) if spec.workspace else os.getcwd()
    session_key = spec.session_key or f"sdk-{id(self):x}"

    # Build run-level context
    run_ctx = AgentRunHookContext(
        messages=list(spec.initial_messages),
    )
    if hook:
        await hook.before_run(run_ctx)

    # Per-iteration state (mutated during turn)
    iter_ctx = AgentHookContext(
        iteration=0,
        messages=list(spec.initial_messages),
    )

    # Accumulators
    all_text_parts: list[str] = []
    tools_used: list[str] = []
    tool_events: list[dict[str, str]] = []
    pending_tool_calls: dict[str, ToolCallRequest] = {}  # call_id → ToolCallRequest

    async def on_delta(delta: str) -> None:
        all_text_parts.append(delta)
        if hook:
            iter_ctx.streamed_content = True
            await hook.on_stream(iter_ctx, delta)

    async def on_tool_start(name: str, tool_input: dict) -> None:
        call_id = f"sdk-{time.time_ns()}"
        tc = ToolCallRequest(id=call_id, name=name, arguments=tool_input)
        pending_tool_calls[call_id] = tc
        tools_used.append(name)
        if hook:
            iter_ctx.tool_calls = [tc]
            iter_ctx.tool_results = []
            iter_ctx.tool_events = []
            await hook.before_execute_tools(iter_ctx)

    async def on_tool_end(name: str, success: bool, output_preview: str) -> None:
        # Find matching pending call (last one with this name)
        tc = None
        for cid, c in reversed(list(pending_tool_calls.items())):
            if c.name == name:
                tc = pending_tool_calls.pop(cid)
                break
        if tc is None:
            return
        status = "ok" if success else "error"
        event = {"name": name, "status": status, "call_id": tc.id}
        tool_events.append(event)
        if hook:
            iter_ctx.tool_calls = [tc]
            iter_ctx.tool_results = [output_preview if success else f"Error: {output_preview}"]
            iter_ctx.tool_events = [event]
            await hook.after_iteration(iter_ctx)

    async def on_reasoning(text: str) -> None:
        if hook:
            await hook.emit_reasoning(text)

    # Execute the turn
    try:
        turn_result = await self._run_turn(
            session_key=session_key,
            prompt=prompt or "",
            model=spec.model,
            cwd=cwd,
            on_delta=on_delta,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_reasoning=on_reasoning,
        )
    except asyncio.CancelledError:
        # /stop was called — tell SDK to interrupt, then re-raise
        await self._interrupt_turn(session_key)
        if hook:
            await hook.emit_reasoning_end()
            await hook.on_stream_end(iter_ctx, resuming=False)
        raise

    # Build final content
    final_content = turn_result.final_content or "".join(all_text_parts) or None

    # Finalize hook
    if hook:
        await hook.emit_reasoning_end()
        # If we streamed content, signal stream end
        if iter_ctx.streamed_content or final_content:
            await hook.on_stream_end(iter_ctx, resuming=False)
        # Apply finalize_content pipeline (strips <think> tags etc.)
        final_content = hook.finalize_content(iter_ctx, final_content)

    # Build messages for session save: initial_messages + assistant reply
    messages = list(spec.initial_messages)
    if final_content:
        messages.append({"role": "assistant", "content": final_content})

    # Build result
    result = AgentRunResult(
        final_content=final_content,
        messages=messages,
        tools_used=turn_result.tools_used or tools_used,
        usage=turn_result.usage,
        stop_reason=turn_result.stop_reason,
        error=turn_result.error,
        tool_events=turn_result.tool_events or tool_events,
        had_injections=False,  # SDK runner doesn't process mid-turn injections
    )

    if hook:
        run_ctx.final_content = final_content
        run_ctx.tools_used = result.tools_used
        run_ctx.usage = result.usage
        run_ctx.stop_reason = result.stop_reason
        run_ctx.error = result.error
        run_ctx.tool_events = result.tool_events
        run_ctx.messages = messages
        await hook.after_run(run_ctx)

    return result
```

### `_save_turn` compatibility (confirmed)

`_save_turn` (loop.py:1690) iterates `all_messages[save_skip:]` where
`save_skip = len(initial_messages)`. SDK runner returns
`all_messages = initial_messages + [{"role": "assistant", "content": final_content}]`.
So `all_messages[save_skip:]` = `[{"role": "assistant", "content": final_content}]`.

`_save_turn` checks:
- `role == "assistant"` and `content` non-empty → passes, persisted ✓
- `role == "tool"` without declared `tool_call_id` → skipped (orphan guard) — not triggered ✓

Phase 1 "只存 user→assistant" 方案完全兼容现有持久化。

### `AgentRunSpec` fields explicitly ignored by SDK runner

These fields from `_run_agent_loop`'s `AgentRunSpec(...)` construction are
deliberately not used by SDKRunner:

| Field | Why ignored |
|-------|------------|
| `tools` | SDK agent manages its own tools (Bash, Edit, Read, etc.) |
| `max_iterations` | SDK has own turn limit (`max_turns` for Claude) |
| `max_tool_result_chars` | SDK manages tool output internally |
| `concurrent_tools` | SDK manages parallelism internally |
| `fail_on_tool_error` | Not applicable — SDK handles tool errors |
| `provider_retry_mode` | Not applicable — SDK manages retries |
| `error_message` | SDK returns own error messages |
| `max_iterations_message` | Not applicable |
| `checkpoint_callback` | Phase 1: no mid-turn checkpoint for SDK turns |
| `injection_callback` (`_drain_pending`) | Phase 1: pending messages queue until turn ends |
| `retry_wait_callback` | Not applicable — SDK manages retry timing |
| `goal_active_predicate` | SDK doesn't see nanobot's sustained goals |
| `goal_continue_message` | SDK doesn't receive mid-turn goal prompts |
| `finalize_on_max_iterations` | SDK manages turn continuation |
| `context_window_tokens` / `context_block_limit` | SDK manages its own context |
| `stream_progress_deltas` | Determined by whether `spec.hook.wants_streaming()` |
| `progress_callback` | Already wrapped in `spec.hook` (AgentProgressHook) |

These are not bugs — they're intentional design choices of the "SDK agent 自治"
strategy. Documented here for code review clarity.

---

## 6. 错误与中断处理

### `/stop` (CancelledError)

When `/stop` is called, `_cancel_active_tasks` cancels the asyncio Task running
the turn. `CancelledError` propagates through `SDKRunner.run()`'s
`await self._run_turn(...)`.

**SDKRunner handling** (in `run()`):
```python
except asyncio.CancelledError:
    await self._interrupt_turn(session_key)
    if hook:
        await hook.emit_reasoning_end()
        await hook.on_stream_end(iter_ctx, resuming=False)
    raise
```

**Subclass `_interrupt_turn`**:
- Codex: find active turn handle for session, call `await turn.interrupt()`
- Claude: call `await client.interrupt()` on the session's client

This tells the SDK subprocess to stop what it's doing. Without this, the
subprocess would continue running even though nanobot has stopped listening.

### SDK subprocess crash

If the SDK subprocess dies mid-turn:
- Codex: `TransportClosedError` is raised by `turn.stream()`
- Claude: `ProcessError` or `CLIConnectionError`

**Handling**: catch in `_run_turn`, return `_TurnResult(stop_reason="error", error="...")`.
The session entry is marked dead. Next turn recreates it automatically.

### SDK package not installed

If `runner_backend` is `"codex-sdk"` or `"claude-sdk"` but the package isn't
installed:

**NOT detected at gateway startup** — avoids crashing the gateway if SDK is
optional. Instead:

**Detected at first turn**: `SDKRunner` subclass does lazy import:
```python
try:
    from openai_codex import AsyncCodex, ...
except ImportError as e:
    return AgentRunResult(
        final_content=None,
        messages=list(spec.initial_messages),
        stop_reason="error",
        error=f"Codex SDK not installed: {e}. Run: pip install openai-codex",
    )
```

User sees a clear error message in the chat, gateway stays up.

### Shutdown cleanup

Add to `close_mcp()` (loop.py:1188) — the existing shutdown hook that drains
background tasks and closes MCP stacks:

```python
async def close_mcp(self) -> None:
    # ... existing cleanup ...
    if self._sdk_runner:
        await self._sdk_runner.shutdown()
```

This ensures SDK subprocesses are terminated when gateway shuts down.

---

## 7. CodexSDKRunner Implementation

```python
class CodexSDKRunner(SDKRunner):
    backend_name = "codex-sdk"

    def __init__(self, config: SDKRunnerConfig):
        self._config = config
        self._codex: AsyncCodex | None = None  # shared instance
        self._codex_lock = asyncio.Lock()       # for lazy init
        self._threads: dict[str, Any] = {}       # session_key → Thread
        self._active_turns: dict[str, Any] = {}  # session_key → TurnHandle
        self._last_activity: dict[str, float] = {}

    async def _ensure_codex(self) -> AsyncCodex:
        """Lazy-create shared AsyncCodex instance."""
        if self._codex is None:
            async with self._codex_lock:
                if self._codex is None:
                    try:
                        from openai_codex import AsyncCodex, CodexConfig
                    except ImportError as e:
                        raise RuntimeError(
                            "openai-codex not installed. Run: pip install openai-codex"
                        ) from e
                    env = {}
                    if self._config.proxy:
                        env["HTTPS_PROXY"] = self._config.proxy
                        env["HTTP_PROXY"] = self._config.proxy
                    self._codex = AsyncCodex(config=CodexConfig(env=env or None))
                    await self._codex.__aenter__()
        return self._codex

    async def _run_turn(self, session_key, prompt, *, model, cwd, on_delta, on_tool_start, on_tool_end, on_reasoning) -> _TurnResult:
        codex = await self._ensure_codex()

        # Get or create thread
        thread = self._threads.get(session_key)
        if thread is None:
            from openai_codex import Sandbox, ApprovalMode
            thread = await codex.thread_start(
                model=model or self._config.codex_model,
                sandbox=Sandbox[self._config.codex_sandbox],  # enum lookup
                approval_mode=ApprovalMode[self._config.codex_approval_mode],
                cwd=cwd,
                base_instructions=self._config.codex_base_instructions,
            )
            self._threads[session_key] = thread

        # Start turn
        turn = await thread.turn(prompt, model=model or self._config.codex_model, cwd=cwd)
        self._active_turns[session_key] = turn

        tools_used: list[str] = []
        usage: dict[str, int] = {}

        try:
            async for event in turn.stream():
                method = event.method
                payload = event.payload

                if method == "agentMessage/delta":
                    await on_delta(payload.delta)

                elif method == "reasoning/text/delta":
                    await on_reasoning(payload.delta)

                elif method == "item/started":
                    item = payload.item.root
                    item_type = getattr(item, "type", "")
                    if item_type == "commandExecution":
                        cmd = getattr(item, "command", "")
                        await on_tool_start("Bash", {"command": cmd})
                    elif item_type == "fileChange":
                        changes = getattr(item, "changes", [])
                        paths = [getattr(c, "path", "") for c in changes if hasattr(c, "path")]
                        await on_tool_start("Edit" if len(changes) == 1 else "MultiEdit", {"files": paths})
                    elif item_type == "mcpToolCall":
                        server = getattr(item, "server", "")
                        tool = getattr(item, "tool", "")
                        args = getattr(item, "arguments", {})
                        await on_tool_start(f"mcp__{server}__{tool}", args)

                elif method == "item/completed":
                    item = payload.item.root
                    item_type = getattr(item, "type", "")
                    if item_type == "commandExecution":
                        cmd = getattr(item, "command", "")
                        exit_code = getattr(item, "exit_code", 1)
                        output = getattr(item, "aggregated_output", "")
                        name = "Bash"
                        tools_used.append(name)
                        await on_tool_end(name, exit_code == 0, output[:500])
                    elif item_type == "fileChange":
                        changes = getattr(item, "changes", [])
                        name = "Edit" if len(changes) == 1 else "MultiEdit"
                        tools_used.append(name)
                        await on_tool_end(name, True, f"{len(changes)} file(s) changed")
                    elif item_type == "mcpToolCall":
                        server = getattr(item, "server", "")
                        tool = getattr(item, "tool", "")
                        status = getattr(item, "status", "")
                        result = getattr(item, "result", "")
                        name = f"mcp__{server}__{tool}"
                        tools_used.append(name)
                        await on_tool_end(name, status == "completed", str(result)[:500])

                elif method == "thread/tokenUsageUpdated":
                    usage_info = getattr(payload, "usage", None)
                    if usage_info:
                        usage = {
                            "prompt_tokens": getattr(usage_info, "input_tokens", 0),
                            "completion_tokens": getattr(usage_info, "output_tokens", 0),
                            "cached_tokens": getattr(usage_info, "cached_tokens", 0),
                        }

            # Get final result
            result = await turn.run()
            final = result.final_response
            status = result.status  # "completed" | "interrupted" | "failed"

            if status == "failed":
                error_info = getattr(result, "error", None)
                error_msg = getattr(error_info, "message", "Codex turn failed") if error_info else "Codex turn failed"
                stop_reason = "error"
                error = error_msg
            elif status == "interrupted":
                stop_reason = "error"
                error = "Turn interrupted"
                final = final or ""
            else:
                stop_reason = "completed"
                error = None

            self._last_activity[session_key] = time.time()
            return _TurnResult(
                final_content=final,
                tools_used=tools_used,
                tool_events=[],  # populated via on_tool_end already
                usage=usage,
                stop_reason=stop_reason,
                error=error,
                messages=[],  # built by base class
            )

        finally:
            self._active_turns.pop(session_key, None)

    async def _interrupt_turn(self, session_key: str) -> None:
        turn = self._active_turns.get(session_key)
        if turn:
            try:
                await turn.interrupt()
            except Exception:
                pass

    async def evict_stale(self, idle_timeout_s: float) -> int:
        now = time.time()
        evicted = 0
        for sk in list(self._threads.keys()):
            last = self._last_activity.get(sk, 0)
            if now - last > idle_timeout_s and sk not in self._active_turns:
                # Thread is idle — codex threads don't need explicit cleanup
                # (they're just state on the app-server). Just forget the reference.
                self._threads.pop(sk, None)
                self._last_activity.pop(sk, None)
                evicted += 1
        return evicted

    async def shutdown(self) -> None:
        if self._codex:
            try:
                await self._codex.__aexit__(None, None, None)
            except Exception:
                pass
            self._codex = None
        self._threads.clear()
        self._active_turns.clear()
```

---

## 8. ClaudeSDKRunner Implementation

```python
class ClaudeSDKRunner(SDKRunner):
    backend_name = "claude-sdk"

    def __init__(self, config: SDKRunnerConfig):
        self._config = config
        self._clients: dict[str, Any] = {}       # session_key → ClaudeSDKClient
        self._client_locks: dict[str, asyncio.Lock] = {}  # per-session init lock
        self._last_activity: dict[str, float] = {}

    async def _ensure_client(self, session_key: str, cwd: str) -> Any:
        """Lazy-create per-session ClaudeSDKClient."""
        if session_key in self._clients:
            return self._clients[session_key]

        lock = self._client_locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            if session_key not in self._clients:
                try:
                    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
                except ImportError as e:
                    raise RuntimeError(
                        "claude-agent-sdk not installed. Run: pip install claude-agent-sdk"
                    ) from e

                env = {}
                if self._config.proxy:
                    env["HTTPS_PROXY"] = self._config.proxy
                    env["HTTP_PROXY"] = self._config.proxy
                if self._config.claude_api_key:
                    env["ANTHROPIC_API_KEY"] = self._config.claude_api_key
                if self._config.claude_base_url:
                    env["ANTHROPIC_BASE_URL"] = self._config.claude_base_url

                options = ClaudeAgentOptions(
                    model=self._config.claude_model,
                    cwd=cwd,
                    permission_mode=self._config.claude_permission_mode,
                    env=env or None,
                    include_partial_messages=True,
                    max_turns=self._config.claude_max_turns,
                    system_prompt=self._config.claude_system_prompt,
                )
                client = ClaudeSDKClient(options=options)
                await client.__aenter__()
                self._clients[session_key] = client

        return self._clients[session_key]

    async def _run_turn(self, session_key, prompt, *, model, cwd, on_delta, on_tool_start, on_tool_end, on_reasoning) -> _TurnResult:
        client = await self._ensure_client(session_key, cwd)

        tools_used: list[str] = []
        usage: dict[str, int] = {}
        final_content_parts: list[str] = []
        stop_reason = "completed"
        error: str | None = None

        try:
            await client.query(prompt)
            async for msg in client.receive_response():
                msg_type = type(msg).__name__

                if msg_type == "AssistantMessage":
                    for block in msg.content:
                        block_type = type(block).__name__
                        if block_type == "TextBlock":
                            text = block.text
                            # Partial messages may arrive incrementally
                            # We detect "new" text by checking length
                            if text and not any(p.endswith(text) for p in final_content_parts):
                                # For streaming: emit delta (approximate)
                                # Claude SDK partial messages contain full text so far,
                                # so we compute the delta from last known state
                                last_len = sum(len(p) for p in final_content_parts)
                                if len(text) > last_len:
                                    delta = text[last_len:]
                                    await on_delta(delta)
                                final_content_parts = [text]
                        elif block_type == "ThinkingBlock":
                            thinking = block.thinking
                            if thinking:
                                await on_reasoning(thinking)
                        elif block_type == "ToolUseBlock":
                            tool_name = block.name
                            tool_input = block.input if isinstance(block.input, dict) else {}
                            tools_used.append(tool_name)
                            await on_tool_start(tool_name, tool_input)

                elif msg_type == "UserMessage":
                    for block in msg.content:
                        block_type = type(block).__name__
                        if block_type == "ToolResultBlock":
                            tool_use_id = block.tool_use_id
                            is_error = block.is_error
                            content = block.content if isinstance(block.content, str) else str(block.content)
                            # Find matching tool name from tools_used
                            # (simplified: last tool used)
                            name = tools_used[-1] if tools_used else "unknown"
                            await on_tool_end(name, not is_error, content[:500])

                elif msg_type == "ResultMessage":
                    stop_reason_map = {
                        "end_turn": "completed",
                        "max_turns": "max_iterations",
                        "stop_sequence": "completed",
                    }
                    sr = getattr(msg, "stop_reason", "end_turn")
                    stop_reason = stop_reason_map.get(sr, "completed")
                    is_error = getattr(msg, "is_error", False)
                    if is_error:
                        stop_reason = "error"
                        error = f"Claude SDK error (stop_reason={sr})"
                    msg_usage = getattr(msg, "usage", None)
                    if msg_usage:
                        # usage shape varies; extract what we can
                        usage = {
                            "prompt_tokens": getattr(msg_usage, "input_tokens", 0),
                            "completion_tokens": getattr(msg_usage, "output_tokens", 0),
                        }

        except Exception as e:
            if "CancelledError" in type(e).__name__ or isinstance(e, asyncio.CancelledError):
                raise
            stop_reason = "error"
            error = str(e)

        self._last_activity[session_key] = time.time()
        final = "".join(final_content_parts) if final_content_parts else None
        return _TurnResult(
            final_content=final,
            tools_used=tools_used,
            tool_events=[],
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            messages=[],
        )

    async def _interrupt_turn(self, session_key: str) -> None:
        client = self._clients.get(session_key)
        if client:
            try:
                await client.interrupt()
            except Exception:
                pass

    async def evict_stale(self, idle_timeout_s: float) -> int:
        now = time.time()
        evicted = 0
        for sk in list(self._clients.keys()):
            last = self._last_activity.get(sk, 0)
            if now - last > idle_timeout_s:
                client = self._clients.pop(sk, None)
                if client:
                    try:
                        await client.__aexit__(None, None, None)
                    except Exception:
                        pass
                self._client_locks.pop(sk, None)
                self._last_activity.pop(sk, None)
                evicted += 1
        return evicted

    async def shutdown(self) -> None:
        for sk in list(self._clients.keys()):
            client = self._clients.pop(sk, None)
            if client:
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass
        self._clients.clear()
        self._client_locks.clear()
        self._last_activity.clear()
```

---

## 9. AgentLoop Seam Injection

### Changes to `nanobot/agent/loop.py`

#### a. `__init__` additions

```python
def __init__(self, ..., runner_backend: str = "native", sdk_runner_config: Any | None = None):
    # ... existing init ...
    self._runner_backend_default = runner_backend
    self._sdk_runner_config = sdk_runner_config
    self._session_runner_backends: dict[str, str] = {}
    self._sdk_runner: SDKRunner | None = None  # lazy
```

#### b. Helper methods

```python
def _effective_runner_backend(self, session_key: str) -> str:
    return self._session_runner_backends.get(session_key, self._runner_backend_default)

def _get_sdk_runner(self) -> SDKRunner:
    if self._sdk_runner is None:
        from nanobot.agent.sdk_runner import CodexSDKRunner, ClaudeSDKRunner
        backend = self._runner_backend_default
        config = self._sdk_runner_config
        if backend == "codex-sdk":
            self._sdk_runner = CodexSDKRunner(config)
        elif backend == "claude-sdk":
            self._sdk_runner = ClaudeSDKRunner(config)
        else:
            raise ValueError(f"Unknown SDK backend: {backend}")
    return self._sdk_runner

def set_session_runner_backend(self, session_key: str, backend: str) -> None:
    self._session_runner_backends[session_key] = backend

def get_session_runner_backend(self, session_key: str) -> str:
    return self._effective_runner_backend(session_key)
```

#### c. `_run_agent_loop` — the seam (at line 874)

```python
runner_backend = self._effective_runner_backend(active_session_key)
if runner_backend == "native":
    runner = self.runner
else:
    runner = self._get_sdk_runner()

result = await runner.run(AgentRunSpec(
    # ... same spec construction as before ...
))
```

Everything else in `_run_agent_loop` stays unchanged: hook construction, context vars,
goal handling, result consumption.

#### d. `from_config` — pass config through (line 399)

```python
return cls(
    # ... existing kwargs ...
    runner_backend=defaults.runner_backend,
    sdk_runner_config=defaults.sdk_runner,
    **extra,
)
```

#### e. `close_mcp` — SDK shutdown (line 1188)

```python
async def close_mcp(self) -> None:
    # ... existing cleanup ...
    if self._sdk_runner:
        try:
            await self._sdk_runner.shutdown()
        except Exception:
            logger.exception("SDK runner shutdown error")
```

#### f. Session eviction in run loop

In the `run()` main loop (line 931), where auto-compact TTL checks happen,
add SDK session eviction. Find the existing `auto_compact.check_expired(...)`
call and add after it:

```python
# Evict stale SDK sessions (runs on same ~60s cadence as compact checks)
if self._sdk_runner:
    idle_s = self._sdk_runner_config.session_idle_timeout_minutes * 60 if self._sdk_runner_config else 3600
    await self._sdk_runner.evict_stale(idle_s)
```

---

## 10. Configuration Schema

### New fields in `nanobot/config/schema.py`

```python
class SDKRunnerConfig(Base):
    """Configuration for SDK-based runner backends (codex-sdk / claude-sdk)."""

    proxy: str | None = None
    # HTTP proxy for SDK subprocess API calls.
    # Example: "http://10.3.42.223:8989"

    # --- Codex SDK settings ---
    codex_model: str = "gpt-4o"
    codex_sandbox: str = "workspace_write"
    # read_only | workspace_write | full_access
    codex_approval_mode: str = "auto_review"
    # auto_review | deny_all
    codex_base_instructions: str | None = None
    # Passed as base_instructions to codex thread_start

    # --- Claude SDK settings ---
    claude_model: str = "claude-sonnet-4-5"
    claude_permission_mode: str = "acceptEdits"
    # default | acceptEdits | plan | bypassPermissions | dontAsk | auto
    claude_api_key: str | None = None
    # Falls back to ANTHROPIC_API_KEY env var
    claude_base_url: str | None = None
    # Falls back to ANTHROPIC_BASE_URL env var
    claude_max_turns: int = 200
    claude_system_prompt: str | None = None

    # --- Session management ---
    session_idle_timeout_minutes: int = Field(default=60, ge=1)


class AgentDefaults(Base):
    # ... all existing fields unchanged ...

    runner_backend: str = "native"
    # "native" | "codex-sdk" | "claude-sdk"

    sdk_runner: SDKRunnerConfig = Field(default_factory=SDKRunnerConfig)
```

### Config example

```json
{
  "agents": {
    "defaults": {
      "runner_backend": "claude-sdk",
      "sdk_runner": {
        "proxy": "http://10.3.42.223:8989",
        "claude_model": "claude-sonnet-4-5",
        "claude_permission_mode": "acceptEdits",
        "session_idle_timeout_minutes": 60
      }
    }
  }
}
```

---

## 11. `/backend` Command

### Changes to `nanobot/command/builtin.py`

Add to `BUILTIN_COMMAND_SPECS` (line 38):
```python
BuiltinCommandSpec(
    name="backend",
    description="Show or switch runner backend for this session",
    usage="/backend [native|codex-sdk|claude-sdk|reset]",
),
```

Handler (registered in `register_builtin_commands`):
```python
async def cmd_backend(ctx: CommandContext) -> OutboundMessage | None:
    loop = ctx.loop
    msg = ctx.msg
    key = ctx.key
    arg = (ctx.arg or "").strip().lower()

    if not arg:
        current = loop.get_session_runner_backend(key)
        default = loop._runner_backend_default
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=(
                f"**Current backend:** `{current}`\n"
                f"**Default:** `{default}`\n\n"
                f"Switch: `/backend codex-sdk` or `/backend native`\n"
                f"Note: model names must match what the SDK accepts."
            ),
        )

    if arg == "reset":
        loop.set_session_runner_backend(key, loop._runner_backend_default)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"Backend reset to default (`{loop._runner_backend_default}`).",
        )

    valid = {"native", "codex-sdk", "claude-sdk"}
    if arg not in valid:
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"Unknown backend: `{arg}`. Valid: {', '.join(f'`{b}`' for b in sorted(valid))}",
        )

    loop.set_session_runner_backend(key, arg)
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id,
        content=f"Backend switched to `{arg}` for this session. Next message will use the new backend.",
    )
```

No `/tmux` command — no tmux.

---

## 12. Compatibility with Existing Nanobot Features

| Feature | SDK backend behavior | Plan |
|---------|---------------------|------|
| **Subagent spawning** | Subagents always use native `AgentRunner` | Phase 1: always native. Phase 2: configurable. |
| **Sustained goals** | `goal_continue_message` not injected mid-turn | Phase 1: goals work at nanobot level (injected into `initial_messages` on next turn), not mid-turn. Accept. |
| **Mid-turn injection** | `injection_callback` ignored; pending messages queue | Phase 1: queued messages dispatched after SDK turn completes. Accept. |
| **Memory consolidation** | Still runs on `session.messages` | Works fine — messages are user+assistant pairs. |
| **`/model` command** | Changes `self.model`, passed via `spec.model` | SDK runner tries `spec.model` first, falls back to config. User must use SDK-accepted names. |
| **MCP servers** | Nanobot MCP not bridged to SDK | Phase 1: MCP configured at SDK level (codex has native MCP support, claude has `mcp_servers` config). Phase 2: bridge. |
| **`/stop`** | `CancelledError` → SDK interrupt → cleanup | Handled in SDKRunner.run() + subclass `_interrupt_turn`. |
| **Session persistence** | `_save_turn` stores user→assistant only | Confirmed compatible. Tool calls not visible in history (Phase 2 concern). |
| **`/new` command** | New session_key → new SDK thread/client | Fresh context on both sides. Works correctly. |
| **Session compaction** | Nanobot compact ≠ SDK compact | SDK manages its own context. Divergence is expected. Document. |
| **Pairing / DM approval** | Unchanged | Channel layer handles this before reaching the runner. |
| **WebUI streaming** | Works via `spec.hook.on_stream` | Confirmed: hook wiring in SDKRunner.run() produces correct stream events. |
| **Feishu card update** | `on_stream` + `on_stream_end` drive card updates | Confirmed: hook wiring produces correct events. |

---

## 13. Risks and Mitigations

### SDK subprocess reliability

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Subprocess crash mid-turn | Turn fails with error | Session entry marked dead, auto-recreated on next turn. User sees error message. |
| Memory leak from idle sessions | Slow growth | `evict_stale()` called on auto-compact cadence. `shutdown()` on gateway stop. |
| Too many Claude subprocesses | Resource usage at scale | Phase 1 accepts one-per-session. Phase 2 explores `query()` + `session_id` stateless mode. |
| Package not installed | Turn fails with clear error | Lazy import + friendly error message. Gateway doesn't crash. |

### Streaming fidelity

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Codex `agentMessage/delta` may include `<think>` | Raw thinking in WebUI | `AgentProgressHook.on_stream` already strips via `IncrementalThinkExtractor`. |
| Claude partial messages contain full accumulated text | Need delta computation | Claude runner tracks `last_len` and computes delta. Imperfect but functional. |
| Tool events don't match nanobot format exactly | WebUI tool cards may not render perfectly | `on_tool_start/end` construct `ToolCallRequest` + call `before_execute_tools/after_iteration`. `format_tool_hints` and `build_tool_event_*` use duck-typing (`getattr`), so it works. |

### Authentication

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Codex not logged in | Turn fails | Document setup: user must run `codex login` or set `OPENAI_API_KEY` before starting gateway. |
| Claude API key missing | Turn fails | `claude_api_key` config or `ANTHROPIC_API_KEY` env. Error message tells user what to do. |
| Proxy not reachable | Turn fails | Proxy is optional. Error message includes proxy URL for debugging. |

### Session divergence

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Nanobot session ≠ SDK thread history | Context mismatch on backend switch | `/backend` help text says "next turn uses new backend." User understands context reset. |
| `/new` creates new nanobot session but SDK thread persists | Stale context | New session_key → new SDK thread/client. Session cache keyed by session_key. |
| SDK thread compacted but nanobot session not | Divergent context windows | Expected behavior. Nanobot session is "gateway log," SDK thread is "agent work log." |

---

## 14. Phase 1 Implementation Steps

| Step | What | Files | Est. lines |
|------|------|-------|-----------|
| 1 | Install SDKs, hello-world with proxy | dev only | — |
| 2 | `SDKRunnerConfig` + `runner_backend` in schema | `config/schema.py` | ~30 |
| 3 | `sdk_runner/__init__.py` + `base.py` (SDKRunner base + _TurnResult) | new files | ~200 |
| 4 | `sdk_runner/codex.py` (CodexSDKRunner) | new file | ~180 |
| 5 | `sdk_runner/claude.py` (ClaudeSDKRunner) | new file | ~180 |
| 6 | AgentLoop seam injection | `agent/loop.py` | ~80 |
| 7 | `/backend` command | `command/builtin.py` | ~50 |
| 8 | `close_mcp` SDK shutdown + run loop evict hook | `agent/loop.py` | ~15 |
| 9 | FakeRunner smoke test (inline in test or test file) | test file | ~40 |
| 10 | End-to-end WebUI smoke with real SDK | runtime | manual |

**Recommended order**: 1 → 2 → 3 → 9 (FakeRunner proves seam) → 6 (seam in loop) → 4 (codex) → 10 (smoke codex) → 5 (claude) → 7 (command) → 8 (cleanup) → 10 (full smoke)

**Why FakeRunner first**: Before writing any real SDK code, implement a minimal
`FakeSDKRunner(SDKRunner)` that returns canned text. This proves:
- The seam in `_run_agent_loop` works
- Hook wiring produces correct WebUI output
- `_save_turn` accepts the returned messages
- `/backend` switching works (if step 7 done earlier)

```python
class FakeSDKRunner(SDKRunner):
    backend_name = "fake"
    async def _run_turn(self, session_key, prompt, **kwargs):
        await kwargs["on_delta"]("Hello from relaybot FakeBackend. ")
        await kwargs["on_delta"](f"You said: {prompt[:50]}")
        return _TurnResult(
            final_content=f"Hello from relaybot FakeBackend. You said: {prompt}",
            tools_used=[], tool_events=[], usage={},
            stop_reason="completed", error=None, messages=[],
        )
    async def evict_stale(self, t): return 0
    async def shutdown(self): pass
    async def _interrupt_turn(self, sk): pass
```

---

## 15. Why "Not First Doing a Large Subtraction"

The core relaybot value is "multi-channel gateway to external coding agents."
Fastest validation path:

1. **Keep nanobot's bus, channels, WebUI, session, transcript intact.**
   These are upstream-compatible foundations we want to track.
2. **Insert the runner backend seam first.** Prove external SDK agent can drive
   a turn end-to-end through existing WebUI. Zero channel/WebUI changes.
3. **Only after seam works** consider trimming native context governance, memory
   consolidation, tool registry, provider marketplace. Those become "disable by
   config" not "delete code" — lower risk, reversible, measurable.

If we started by deleting native agent code, we'd debug two unknowns
simultaneously (does the trim work? does the SDK runner work?). Seam first gives
a working baseline.

---

## 16. Phase 2+ Roadmap (out of scope)

- **Tool call visibility in session history** — reconstruct tool_call/tool_result
  message pairs from SDK events so `_save_turn` stores them
- **Media/image input** — pass image content blocks to SDK (both support it)
- **Per-preset backend** — `ModelPresetConfig.runner_backend` for model-specific backends
- **Subagent backend selection** — allow subagents to use SDK backends
- **SDK MCP bridge** — map nanobot MCP config into SDK initialization
- **Session import/sync** — bridge SDK thread history into nanobot session
- **Stateless Claude SDK** — `query()` + `session_id` instead of per-session client
- **coco/seed/relay-specific tuning** — extra args, custom binary paths via config
- **Mid-turn goal injection** — periodically check goal state and steer SDK turn
- **Model name mapping** — nanobot model name → SDK model name translation table
