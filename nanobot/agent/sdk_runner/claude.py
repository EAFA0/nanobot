"""Claude Agent SDK runner — drives the claude-agent-sdk ClaudeSDKClient.

One ClaudeSDKClient per session (stateful — holds conversation context).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.sdk_runner.base import SDKRunner, _TurnResult


class ClaudeSDKRunner(SDKRunner):
    """SDK runner backed by the Claude Code / Relay / Seed CLI."""

    backend_name = "claude-sdk"

    def __init__(self, config: Any):
        self._config = config
        self._clients: dict[str, Any] = {}  # session_key → ClaudeSDKClient
        self._client_locks: dict[str, asyncio.Lock] = {}  # per-session init lock
        self._last_activity: dict[str, float] = {}
        self._active_turns: set[str] = set()

    async def _ensure_client(self, session_key: str, cwd: str) -> Any:
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

                env: dict[str, str] = {}
                proxy = getattr(self._config, "proxy", None)
                if proxy:
                    env["HTTPS_PROXY"] = proxy
                    env["HTTP_PROXY"] = proxy

                api_key = getattr(self._config, "claude_api_key", None)
                if api_key:
                    env["ANTHROPIC_API_KEY"] = api_key

                base_url = getattr(self._config, "claude_base_url", None)
                if base_url:
                    env["ANTHROPIC_BASE_URL"] = base_url

                options_kwargs: dict[str, Any] = dict(
                    cwd=cwd,
                    permission_mode=getattr(self._config, "claude_permission_mode", "acceptEdits"),
                    include_partial_messages=True,
                    max_turns=getattr(self._config, "claude_max_turns", 200),
                )
                config_model = getattr(self._config, "claude_model", None)
                if config_model:
                    options_kwargs["model"] = config_model
                system_prompt = getattr(self._config, "claude_system_prompt", None)
                agents_md = self._load_agents_md(cwd)
                if agents_md:
                    merged = f"# AGENTS.md\n\n{agents_md}"
                    if system_prompt:
                        merged = f"{system_prompt}\n\n---\n\n{merged}"
                    options_kwargs["system_prompt"] = merged
                elif system_prompt:
                    options_kwargs["system_prompt"] = system_prompt
                cli_path = getattr(self._config, "claude_cli_path", None)
                if cli_path:
                    options_kwargs["cli_path"] = cli_path
                if env:
                    options_kwargs["env"] = env
                options = ClaudeAgentOptions(**options_kwargs)
                client = ClaudeSDKClient(options=options)
                await client.__aenter__()
                self._clients[session_key] = client
                self._last_activity[session_key] = time.time()
                logger.debug("Created new claude client for session {}", session_key)

        return self._clients[session_key]

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
        client = await self._ensure_client(session_key, cwd)
        self._active_turns.add(session_key)

        tools_used: list[str] = []
        usage: dict[str, int] = {}
        final_content_parts: list[str] = []
        stop_reason = "completed"
        error: str | None = None
        last_text_len = 0
        pending_tool_calls: list[str] = []  # stack of tool names awaiting results

        try:
            await client.query(prompt)
            async for msg in client.receive_response():
                msg_type = type(msg).__name__

                if msg_type == "AssistantMessage":
                    content = getattr(msg, "content", []) or []
                    for block in content:
                        block_type = type(block).__name__
                        if block_type == "TextBlock":
                            text = getattr(block, "text", "") or ""
                            if text:
                                # Partial messages contain accumulated text; emit delta
                                if len(text) > last_text_len:
                                    delta = text[last_text_len:]
                                    await on_delta(delta)
                                last_text_len = len(text)
                                final_content_parts = [text]
                        elif block_type == "ThinkingBlock":
                            thinking = getattr(block, "thinking", "") or ""
                            if thinking:
                                await on_reasoning(thinking)
                        elif block_type == "ToolUseBlock":
                            tool_name = getattr(block, "name", "unknown")
                            tool_input = getattr(block, "input", {}) or {}
                            if isinstance(tool_input, str):
                                import json
                                try:
                                    tool_input = json.loads(tool_input)
                                except (json.JSONDecodeError, TypeError):
                                    tool_input = {"input": tool_input}
                            tools_used.append(tool_name)
                            pending_tool_calls.append(tool_name)
                            await on_tool_start(tool_name, tool_input)

                elif msg_type == "UserMessage":
                    content = getattr(msg, "content", []) or []
                    for block in content:
                        block_type = type(block).__name__
                        if block_type == "ToolResultBlock":
                            is_error = getattr(block, "is_error", False)
                            block_content = getattr(block, "content", "") or ""
                            if not isinstance(block_content, str):
                                block_content = str(block_content)
                            # Match to last pending tool call
                            name = pending_tool_calls.pop() if pending_tool_calls else "unknown"
                            await on_tool_end(name, not is_error, block_content[:500])

                elif msg_type == "ResultMessage":
                    sr = getattr(msg, "stop_reason", "end_turn")
                    stop_reason_map = {
                        "end_turn": "completed",
                        "max_turns": "max_iterations",
                        "stop_sequence": "completed",
                    }
                    stop_reason = stop_reason_map.get(sr, "completed")
                    is_error = getattr(msg, "is_error", False)
                    if is_error:
                        stop_reason = "error"
                        error = f"Claude SDK error (stop_reason={sr})"
                    msg_usage = getattr(msg, "usage", None)
                    if msg_usage:
                        input_t = getattr(msg_usage, "input_tokens", None)
                        output_t = getattr(msg_usage, "output_tokens", None)
                        if input_t is not None:
                            usage["prompt_tokens"] = input_t
                        if output_t is not None:
                            usage["completion_tokens"] = output_t

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Claude SDK turn error")
            stop_reason = "error"
            error = str(e)
        finally:
            self._active_turns.discard(session_key)

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

    async def list_models(self) -> list[dict[str, Any]]:
        client = await self._ensure_client("__model_query__", os.getcwd())
        try:
            info = await client.get_server_info()
        except Exception:
            logger.exception("Failed to query claude/relay server info for models")
            return []
        models = info.get("models", []) if isinstance(info, dict) else []
        result: list[dict[str, Any]] = []
        for m in models:
            result.append({
                "id": m.get("value", ""),
                "name": m.get("displayName", m.get("value", "")),
                "description": m.get("description", ""),
                "is_default": False,
                "supports_effort": m.get("supportsEffort", False),
                "effort_levels": m.get("supportedEffortLevels", []),
            })
        return result

    async def set_model(self, session_key: str, model: str) -> None:
        client = self._clients.get(session_key)
        if client is None:
            logger.debug("Cannot set_model: no client for session {}", session_key)
            return
        try:
            await client.set_model(model)
            logger.debug("Set model to {} for session {}", model, session_key)
        except Exception:
            logger.exception("Failed to set model {} for session {}", model, session_key)

    async def _interrupt_turn(self, session_key: str) -> None:
        client = self._clients.get(session_key)
        if client:
            try:
                await client.interrupt()
                logger.debug("Interrupted claude turn for session {}", session_key)
            except Exception:
                logger.debug("Error interrupting claude turn (may have already finished)")

    async def evict_stale(self, idle_timeout_s: float) -> int:
        now = time.time()
        evicted = 0
        for sk in list(self._clients.keys()):
            last = self._last_activity.get(sk, 0)
            if now - last > idle_timeout_s and sk not in self._active_turns:
                client = self._clients.pop(sk, None)
                if client:
                    try:
                        await client.__aexit__(None, None, None)
                    except Exception:
                        pass
                self._client_locks.pop(sk, None)
                self._last_activity.pop(sk, None)
                evicted += 1
        if evicted:
            logger.debug("Evicted {} stale claude clients", evicted)
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
        logger.info("Claude SDK clients shut down")
