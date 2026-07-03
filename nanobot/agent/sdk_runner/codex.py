"""Codex SDK runner — drives the openai-codex AsyncCodex SDK.

One shared AsyncCodex instance, multiple threads multiplexed.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.sdk_runner.base import SDKRunner, TurnResult


class CodexSDKRunner(SDKRunner):
    """SDK runner backed by the Codex / Coco app-server."""

    backend_name = "codex-sdk"

    def __init__(self, config: Any):
        self._config = config
        self._codex: Any = None  # AsyncCodex
        self._codex_lock = asyncio.Lock()
        self._threads: dict[str, Any] = {}  # session_key → Thread
        self._thread_ids: dict[str, str] = {}  # session_key → thread_id (for cross-restart resume)
        self._active_turns: dict[str, Any] = {}  # session_key → TurnHandle
        self._last_activity: dict[str, float] = {}
        self._session_models: dict[str, str] = {}  # session_key → model override

    async def _ensure_codex(self) -> Any:
        if self._codex is None:
            async with self._codex_lock:
                if self._codex is None:
                    try:
                        from openai_codex import AsyncCodex, CodexConfig
                    except ImportError as e:
                        raise RuntimeError(
                            "openai-codex not installed. Run: pip install openai-codex"
                        ) from e

                    env: dict[str, str] = {}
                    proxy = getattr(self._config, "proxy", None)
                    if proxy:
                        env["HTTPS_PROXY"] = proxy
                        env["HTTP_PROXY"] = proxy

                    codex_bin = getattr(self._config, "codex_bin", None)
                    codex_kwargs: dict[str, Any] = dict(
                        codex_bin=codex_bin,
                        cwd=os.getcwd(),
                    )
                    if env:
                        codex_kwargs["env"] = env
                    self._codex = AsyncCodex(config=CodexConfig(**codex_kwargs))
                    await self._codex.__aenter__()
                    logger.info("Codex SDK client initialized")
        return self._codex

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
        codex = await self._ensure_codex()
        from openai_codex import Sandbox, ApprovalMode

        thread = self._threads.get(session_key)
        if thread is None:
            saved_thread_id = self._thread_ids.get(session_key)
            if saved_thread_id:
                # Resume a previously-known thread (e.g. after process restart)
                thread = await codex.thread_resume(saved_thread_id, cwd=cwd)
                self._threads[session_key] = thread
                logger.debug("Resumed codex thread {} for session {}", saved_thread_id, session_key)
            else:
                sandbox_name = getattr(self._config, "codex_sandbox", "workspace_write")
                approval_name = getattr(self._config, "codex_approval_mode", "auto_review")
                base_instructions = getattr(self._config, "codex_base_instructions", None)
                agents_md = self._load_agents_md(cwd)
                if agents_md:
                    merged = f"# AGENTS.md\n\n{agents_md}"
                    if base_instructions:
                        merged = f"{base_instructions}\n\n---\n\n{merged}"
                    base_instructions = merged

                thread_kwargs = dict(
                    sandbox=Sandbox[sandbox_name],
                    approval_mode=ApprovalMode[approval_name],
                    cwd=cwd,
                    base_instructions=base_instructions,
                )
                # Only pass model if explicitly configured — let the codex binary use its own default.
                # Note: we ignore the `model` param from _run_turn because that comes from the native
                # provider config (e.g. "minimax-m3") and is not a valid codex model.
                config_model = getattr(self._config, "codex_model", None)
                if config_model:
                    thread_kwargs["model"] = config_model

                thread = await codex.thread_start(**thread_kwargs)
                self._threads[session_key] = thread
                # Persist thread_id for cross-restart resume
                thread_id = getattr(thread, "id", None) or getattr(thread, "thread_id", None)
                if thread_id:
                    self._thread_ids[session_key] = str(thread_id)
                logger.debug("Created new codex thread for session {}", session_key)

        turn_kwargs = dict(cwd=cwd)
        # Per-session model override takes precedence over global config
        session_model = self._session_models.get(session_key)
        config_model = getattr(self._config, "codex_model", None)
        model = session_model or config_model
        if model:
            turn_kwargs["model"] = model
        turn = await thread.turn(prompt, **turn_kwargs)
        self._active_turns[session_key] = turn

        tools_used: list[str] = []
        usage: dict[str, int] = {}
        all_deltas: list[str] = []
        stop_reason = "completed"
        error: str | None = None

        try:
            async for event in turn.stream():
                method = getattr(event, "method", "")
                payload = getattr(event, "payload", None)

                if method == "item/agentMessage/delta" and payload:
                    delta = getattr(payload, "delta", "")
                    if delta:
                        all_deltas.append(delta)
                        await on_delta(delta)

                elif method == "item/reasoning/text/delta" and payload:
                    delta = getattr(payload, "delta", "")
                    if delta:
                        await on_reasoning(delta)

                elif method == "item/started" and payload:
                    item = self._root_item(payload)
                    await self._handle_item_started(item, on_tool_start)

                elif method == "item/completed" and payload:
                    item = self._root_item(payload)
                    name = await self._handle_item_completed(item, on_tool_end)
                    if name:
                        tools_used.append(name)

                elif method == "thread/tokenUsage/updated" and payload:
                    token_usage = getattr(payload, "token_usage", None)
                    if token_usage is not None:
                        last = getattr(token_usage, "last", None)
                        if last is not None:
                            usage = {
                                "prompt_tokens": getattr(last, "input_tokens", 0),
                                "completion_tokens": getattr(last, "output_tokens", 0),
                                "cached_tokens": getattr(last, "cached_input_tokens", 0),
                            }

                elif method == "turn/completed" and payload:
                    turn_info = getattr(payload, "turn", None)
                    if turn_info is not None:
                        status = getattr(turn_info, "status", None)
                        status_str = str(status) if status else "completed"
                        if "failed" in status_str:
                            stop_reason = "error"
                            err = getattr(turn_info, "error", None)
                            error = getattr(err, "message", "Codex turn failed") if err else "Codex turn failed"
                        elif "interrupted" in status_str:
                            stop_reason = "error"
                            error = "Turn interrupted"

        finally:
            self._active_turns.pop(session_key, None)

        self._last_activity[session_key] = time.time()
        final = "".join(all_deltas) if all_deltas else None
        return TurnResult(
            final_content=final,
            tools_used=tools_used,
            tool_events=[],
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            messages=[],
        )

    @staticmethod
    def _root_item(payload: Any) -> Any:
        """Extract the root item from a payload."""
        item = getattr(payload, "item", None)
        if item is None:
            return None
        return getattr(item, "root", item)

    @staticmethod
    def _command_execution_tool_name(item: Any) -> str:
        """Determine tool name from a commandExecution item's action type."""
        actions = getattr(item, "command_actions", None) or []
        if actions:
            action_type = getattr(actions[0], "type", "")
            if action_type == "read":
                return "Read"
            if action_type == "search":
                return "Grep"
        return "Bash"

    @staticmethod
    async def _handle_item_started(item: Any, on_tool_start: Callable) -> None:
        if item is None:
            return
        item_type = getattr(item, "type", "")
        if item_type == "commandExecution":
            cmd = getattr(item, "command", "")
            name = CodexSDKRunner._command_execution_tool_name(item)
            await on_tool_start(name, {"command": cmd})
        elif item_type == "fileChange":
            changes = getattr(item, "changes", []) or []
            paths = [getattr(c, "path", "") for c in changes if hasattr(c, "path")]
            name = "Edit" if len(changes) == 1 else "MultiEdit"
            await on_tool_start(name, {"files": paths})
        elif item_type == "mcpToolCall":
            server = getattr(item, "server", "")
            tool = getattr(item, "tool", "")
            args = getattr(item, "arguments", {}) or {}
            await on_tool_start(f"mcp__{server}__{tool}", args)

    @staticmethod
    async def _handle_item_completed(item: Any, on_tool_end: Callable) -> str | None:
        if item is None:
            return None
        item_type = getattr(item, "type", "")
        if item_type == "commandExecution":
            cmd = getattr(item, "command", "")
            exit_code = getattr(item, "exit_code", 1)
            output = getattr(item, "aggregated_output", "") or ""
            name = CodexSDKRunner._command_execution_tool_name(item)
            await on_tool_end(name, exit_code == 0, output[:500])
            return name
        elif item_type == "fileChange":
            changes = getattr(item, "changes", []) or []
            name = "Edit" if len(changes) == 1 else "MultiEdit"
            await on_tool_end(name, True, f"{len(changes)} file(s) changed")
            return name
        elif item_type == "mcpToolCall":
            server = getattr(item, "server", "")
            tool = getattr(item, "tool", "")
            status = getattr(item, "status", "")
            result = getattr(item, "result", "") or ""
            name = f"mcp__{server}__{tool}"
            await on_tool_end(name, status == "completed", str(result)[:500])
            return name
        return None

    async def list_models(self) -> list[dict[str, Any]]:
        codex = await self._ensure_codex()
        try:
            sync_client = codex._client._sync
            raw = sync_client._request_raw("model/list", {})
        except Exception:
            logger.exception("Failed to query codex/traex model list")
            return []
        data = raw.get("data", []) if isinstance(raw, dict) else []
        result: list[dict[str, Any]] = []
        for m in data:
            if m.get("hidden"):
                continue
            result.append({
                "id": m.get("id", ""),
                "name": m.get("displayName", m.get("id", "")),
                "description": m.get("description", ""),
                "is_default": m.get("isDefault", False),
                "context_window": m.get("contextWindow"),
                "family": m.get("modelFamily", ""),
            })
        return result

    async def set_model(self, session_key: str, model: str) -> None:
        self._session_models[session_key] = model
        logger.debug("Set per-session model override to {} for session {}", model, session_key)

    def get_model(self, session_key: str) -> str | None:
        return self._session_models.get(session_key)

    def get_thread_id(self, session_key: str) -> str | None:
        return self._thread_ids.get(session_key)

    def set_thread_id(self, session_key: str, thread_id: str) -> None:
        """Restore a thread_id from persisted metadata (e.g. after process restart)."""
        self._thread_ids[session_key] = thread_id

    async def _interrupt_turn(self, session_key: str) -> None:
        turn = self._active_turns.get(session_key)
        if turn:
            try:
                await turn.interrupt()
                logger.debug("Interrupted codex turn for session {}", session_key)
            except Exception:
                logger.debug("Error interrupting codex turn (may have already finished)")

    async def evict_session(self, session_key: str) -> None:
        self._threads.pop(session_key, None)
        self._thread_ids.pop(session_key, None)
        self._last_activity.pop(session_key, None)
        self._session_models.pop(session_key, None)
        logger.debug("Evicted codex session {}", session_key)

    async def evict_stale(self, idle_timeout_s: float) -> int:
        now = time.time()
        evicted = 0
        for sk in list(self._threads.keys()):
            last = self._last_activity.get(sk, 0)
            if now - last > idle_timeout_s and sk not in self._active_turns:
                self._threads.pop(sk, None)
                self._thread_ids.pop(sk, None)
                self._last_activity.pop(sk, None)
                self._session_models.pop(sk, None)
                evicted += 1
        if evicted:
            logger.debug("Evicted {} stale codex threads", evicted)
        return evicted

    async def shutdown(self) -> None:
        if self._codex:
            try:
                await self._codex.__aexit__(None, None, None)
                logger.info("Codex SDK client shut down")
            except Exception:
                logger.exception("Codex SDK shutdown error")
            self._codex = None
        self._threads.clear()
        self._thread_ids.clear()
        self._active_turns.clear()
        self._last_activity.clear()
        self._session_models.clear()
