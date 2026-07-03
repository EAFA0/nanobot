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
from nanobot.utils.progress_events import invoke_file_edit_progress


@dataclass
class TurnResult:
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
        progress_cb = spec.progress_callback
        pending_file_edits: dict[str, list[dict[str, Any]]] = {}  # call_id → file_edit events

        def _build_file_edit_events(
            *,
            call_id: str,
            tool_name: str,
            paths: list[str],
            phase: str,
            status: str,
        ) -> list[dict[str, Any]]:
            events: list[dict[str, Any]] = []
            for p in paths:
                events.append({
                    "version": 1,
                    "call_id": call_id,
                    "tool": tool_name,
                    "path": p,
                    "absolute_path": p,
                    "phase": phase,
                    "added": 0,
                    "deleted": 0,
                    "approximate": True,
                    "status": status,
                })
            return events

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
            # Emit file_edit events for Edit/MultiEdit so WebUI shows file cards
            if name in ("Edit", "MultiEdit") and progress_cb:
                paths = tool_input.get("files", []) if isinstance(tool_input, dict) else []
                if isinstance(paths, list) and paths:
                    events = _build_file_edit_events(
                        call_id=call_id, tool_name=name, paths=paths,
                        phase="start", status="editing",
                    )
                    pending_file_edits[call_id] = events
                    await invoke_file_edit_progress(progress_cb, events)
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
            # Emit file_edit end events for Edit/MultiEdit
            if name in ("Edit", "MultiEdit") and progress_cb:
                paths = (tc.arguments or {}).get("files", []) if isinstance(tc.arguments, dict) else []
                if isinstance(paths, list) and paths:
                    fe_status = "done" if success else "error"
                    events = _build_file_edit_events(
                        call_id=tc.id, tool_name=name, paths=paths,
                        phase="end", status=fe_status,
                    )
                    pending_file_edits.pop(tc.id, None)
                    await invoke_file_edit_progress(progress_cb, events)
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
            turn_result = await self._run_turn(
                session_key=session_key,
                prompt=prompt,
                model=spec.model,
                cwd=cwd,
                on_delta=on_delta,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                on_reasoning=on_reasoning,
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
    ) -> TurnResult:
        """Execute one turn. Subclass implements."""
        ...

    @abstractmethod
    async def _interrupt_turn(self, session_key: str) -> None:
        """Stop an in-flight turn (called on /stop)."""
        ...

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """List models available from the underlying SDK binary.

        Returns list of dicts with at least: id, name, description, is_default.
        """
        ...

    @abstractmethod
    async def set_model(self, session_key: str, model: str) -> None:
        """Switch model for a session. No-op if not supported."""
        ...

    def get_model(self, session_key: str) -> str | None:
        """Return per-session model override, or None if not set."""
        return None

    @abstractmethod
    async def evict_session(self, session_key: str) -> None:
        """Remove a specific session's SDK resources (thread/client)."""
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
    def _load_agents_md(cwd: str) -> str | None:
        """Read AGENTS.md for SDK system context injection.

        Priority:
        1. ~/.nanobot/workspace/AGENTS.md (user's workspace instructions)
        2. {cwd}/AGENTS.md (project-level instructions)

        Content is passed through directly — no prefix or disclaimer added.
        """
        from pathlib import Path

        parts: list[str] = []

        workspace_agents = Path.home() / ".nanobot" / "workspace" / "AGENTS.md"
        if workspace_agents.is_file():
            try:
                content = workspace_agents.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(content)
            except OSError:
                pass

        for candidate in (Path(cwd) / "AGENTS.md", Path(cwd) / "agents.md"):
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except OSError:
                    pass
                break

        return "\n\n---\n\n".join(parts) if parts else None

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
