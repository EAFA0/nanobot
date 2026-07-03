"""SDKRunner base class — shared logic for all SDK-based runners.

Subclasses (CodexSDKRunner, ClaudeSDKRunner) implement _run_turn() for
SDK-specific event parsing and session management. The base class handles:

- User message extraction from initial_messages
- AgentHook lifecycle wiring (before_run, on_stream, before_execute_tools, etc.)
- AgentRunResult assembly
- CancelledError / stop handling
- _save_turn-compatible message list construction
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.hook import AgentHookContext, AgentRunHookContext
from nanobot.agent.runner import AgentRunResult, AgentRunSpec
from nanobot.providers.base import ToolCallRequest


@dataclass
class _TurnResult:
    """Internal turn result returned by subclass _run_turn()."""

    final_content: str | None
    tools_used: list[str] = field(default_factory=list)
    tool_events: list[dict[str, str]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)


class SDKRunner(ABC):
    """Base for SDK-based runners.

    Implements .run(spec) -> AgentRunResult with the same contract as
    AgentRunner. Subclasses implement _run_turn() for SDK-specific details.
    """

    backend_name: str = "sdk"

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        prompt = self._extract_user_prompt(spec.initial_messages)
        if prompt is None:
            logger.warning("SDK runner: no user message found in initial_messages")
            return AgentRunResult(
                final_content=None,
                messages=list(spec.initial_messages),
                stop_reason="error",
                error="No user message found",
            )

        cwd = str(spec.workspace) if spec.workspace else os.getcwd()
        session_key = spec.session_key or f"sdk-{id(self):x}"
        hook = spec.hook

        run_ctx = AgentRunHookContext(messages=list(spec.initial_messages))
        if hook:
            await hook.before_run(run_ctx)

        iter_ctx = AgentHookContext(
            iteration=0,
            messages=list(spec.initial_messages),
            session_key=session_key,
        )

        all_text_parts: list[str] = []
        tools_used: list[str] = []
        tool_events: list[dict[str, str]] = []
        pending_tool_calls: dict[str, ToolCallRequest] = {}

        async def on_delta(delta: str) -> None:
            all_text_parts.append(delta)
            if hook:
                iter_ctx.streamed_content = True
                await hook.on_stream(iter_ctx, delta)

        async def on_tool_start(name: str, tool_input: dict[str, Any]) -> None:
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
                iter_ctx.tool_results = [
                    output_preview if success else f"Error: {output_preview}"
                ]
                iter_ctx.tool_events = [event]
                await hook.after_iteration(iter_ctx)

        async def on_reasoning(text: str) -> None:
            if hook:
                await hook.emit_reasoning(text)

        try:
            timeout_s = getattr(self, "_turn_timeout_s", 120)
            turn_result = await asyncio.wait_for(
                self._run_turn(
                    session_key=session_key,
                    prompt=prompt,
                    model=spec.model,
                    cwd=cwd,
                    on_delta=on_delta,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    on_reasoning=on_reasoning,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("SDK runner turn timed out after {}s for session {}", timeout_s, session_key)
            await self._interrupt_turn(session_key)
            if hook:
                await hook.emit_reasoning_end()
                await hook.on_stream_end(iter_ctx, resuming=False)
            return AgentRunResult(
                final_content=None,
                messages=list(spec.initial_messages),
                tools_used=tools_used,
                usage={},
                stop_reason="error",
                error=f"SDK turn timed out after {timeout_s}s",
                tool_events=tool_events,
                had_injections=False,
            )
        except asyncio.CancelledError:
            await self._interrupt_turn(session_key)
            if hook:
                await hook.emit_reasoning_end()
                await hook.on_stream_end(iter_ctx, resuming=False)
            raise
        except Exception as exc:
            logger.exception("SDK runner turn error for session {}", session_key)
            if hook:
                await hook.emit_reasoning_end()
                await hook.on_stream_end(iter_ctx, resuming=False)
            return AgentRunResult(
                final_content=None,
                messages=list(spec.initial_messages),
                tools_used=[],
                usage={},
                stop_reason="error",
                error=str(exc),
                tool_events=[],
                had_injections=False,
            )

        final_content = turn_result.final_content or "".join(all_text_parts) or None

        if hook:
            await hook.emit_reasoning_end()
            if iter_ctx.streamed_content or final_content:
                await hook.on_stream_end(iter_ctx, resuming=False)
            final_content = hook.finalize_content(iter_ctx, final_content)

        messages = list(spec.initial_messages)
        if final_content:
            messages.append({"role": "assistant", "content": final_content})

        result = AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=turn_result.tools_used or tools_used,
            usage=turn_result.usage,
            stop_reason=turn_result.stop_reason,
            error=turn_result.error,
            tool_events=turn_result.tool_events or tool_events,
            had_injections=False,
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

    @abstractmethod
    async def _run_turn(
        self,
        *,
        session_key: str,
        prompt: str,
        model: str | None,
        cwd: str,
        on_delta: Callable[[str], Awaitable[None]],
        on_tool_start: Callable[[str, dict[str, Any]], Awaitable[None]],
        on_tool_end: Callable[[str, bool, str], Awaitable[None]],
        on_reasoning: Callable[[str], Awaitable[None]],
    ) -> _TurnResult:
        """Execute one turn. Subclass implements."""
        ...

    @abstractmethod
    async def _interrupt_turn(self, session_key: str) -> None:
        """Stop an in-flight turn (called on /stop)."""
        ...

    @abstractmethod
    async def evict_stale(self, idle_timeout_s: float) -> int:
        """Remove idle sessions. Return count evicted."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up all SDK subprocesses."""
        ...

    @staticmethod
    def _extract_user_prompt(messages: list[dict[str, Any]]) -> str | None:
        """Extract text from last user message.

        Returns None if no user message found. Logs warning if image blocks
        are detected (Phase 1: not supported).
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    images = [
                        b
                        for b in content
                        if isinstance(b, dict)
                        and b.get("type") in ("image_url", "image")
                    ]
                    if images:
                        logger.warning(
                            "SDK runner: image input detected — "
                            "images not supported in Phase 1, ignoring"
                        )
                    return "\n".join(texts) if texts else None
        return None
