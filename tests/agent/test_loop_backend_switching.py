"""Tests for AgentLoop backend switching and SDK runner dispatch seam.

Covers:
- _effective_runner_backend resolution (default vs per-session override)
- set/get_session_runner_backend API
- _get_sdk_runner lazy creation per backend name
- Seam dispatch: native vs SDK runner selection
- /backend command handler
- SDKRunner timeout behavior
- Multiple SDK runners coexist independently
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult, AgentRunSpec
from nanobot.agent.sdk_runner import SDKRunner, _TurnResult
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import cmd_backend
from nanobot.command.router import CommandContext
from nanobot.config.schema import SDKRunnerConfig


def _make_provider(default_model: str = "test-model") -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = SimpleNamespace(
        max_tokens=4096, temperature=0.1, reasoning_effort=None,
    )
    provider.estimate_prompt_tokens.return_value = (10_000, "test")
    return provider


def _make_loop(
    tmp_path,
    *,
    runner_backend: str = "native",
    sdk_runner_config: SDKRunnerConfig | None = None,
) -> AgentLoop:
    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub:
        mock_sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        return AgentLoop(
            bus=MessageBus(),
            provider=_make_provider(),
            workspace=tmp_path,
            model="test-model",
            context_window_tokens=128_000,
            runner_backend=runner_backend,
            sdk_runner_config=sdk_runner_config,
        )


# ── Backend resolution ──────────────────────────────────────────────


class TestEffectiveRunnerBackend:
    def test_defaults_to_native(self, tmp_path):
        loop = _make_loop(tmp_path)
        assert loop._effective_runner_backend("sess-1") == "native"

    def test_default_from_config(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="codex-sdk")
        assert loop._effective_runner_backend("sess-1") == "codex-sdk"

    def test_per_session_override(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="native")
        loop.set_session_runner_backend("sess-1", "claude-sdk")
        assert loop._effective_runner_backend("sess-1") == "claude-sdk"

    def test_per_session_does_not_affect_other_sessions(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="codex-sdk")
        loop.set_session_runner_backend("sess-a", "native")
        assert loop._effective_runner_backend("sess-a") == "native"
        assert loop._effective_runner_backend("sess-b") == "codex-sdk"
        assert loop._effective_runner_backend("sess-c") == "codex-sdk"

    def test_get_session_runner_backend_alias(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="codex-sdk")
        loop.set_session_runner_backend("s1", "claude-sdk")
        assert loop.get_session_runner_backend("s1") == "claude-sdk"
        assert loop.get_session_runner_backend("s2") == "codex-sdk"


# ── SDK runner lazy creation ────────────────────────────────────────


class TestGetSDKRunner:
    def test_creates_codex_runner(self, tmp_path):
        loop = _make_loop(tmp_path)
        runner = loop._get_sdk_runner("codex-sdk")
        assert runner.backend_name == "codex-sdk"

    def test_creates_claude_runner(self, tmp_path):
        loop = _make_loop(tmp_path)
        runner = loop._get_sdk_runner("claude-sdk")
        assert runner.backend_name == "claude-sdk"

    def test_caches_by_backend_name(self, tmp_path):
        loop = _make_loop(tmp_path)
        r1 = loop._get_sdk_runner("codex-sdk")
        r2 = loop._get_sdk_runner("codex-sdk")
        assert r1 is r2

    def test_multiple_backends_coexist(self, tmp_path):
        loop = _make_loop(tmp_path)
        codex = loop._get_sdk_runner("codex-sdk")
        claude = loop._get_sdk_runner("claude-sdk")
        assert codex is not claude
        assert codex.backend_name == "codex-sdk"
        assert claude.backend_name == "claude-sdk"
        assert len(loop._sdk_runners) == 2

    def test_unknown_backend_raises(self, tmp_path):
        loop = _make_loop(tmp_path)
        with pytest.raises(ValueError, match="Unknown SDK backend"):
            loop._get_sdk_runner("unknown-sdk")

    def test_passes_config(self, tmp_path):
        cfg = SDKRunnerConfig(proxy="http://proxy:8080")
        loop = _make_loop(tmp_path, sdk_runner_config=cfg)
        runner = loop._get_sdk_runner("codex-sdk")
        assert runner._config.proxy == "http://proxy:8080"

    def test_default_config_when_none(self, tmp_path):
        loop = _make_loop(tmp_path, sdk_runner_config=None)
        runner = loop._get_sdk_runner("codex-sdk")
        assert runner._config is not None


# ── /backend command ────────────────────────────────────────────────


def _cmd_ctx(loop: AgentLoop, raw: str, args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=raw)
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


class TestBackendCommand:
    @pytest.mark.asyncio
    async def test_shows_current_and_default(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="codex-sdk")
        out = await cmd_backend(_cmd_ctx(loop, "/backend"))
        assert "codex-sdk" in out.content
        assert "Current" in out.content

    @pytest.mark.asyncio
    async def test_shows_per_session_override(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="codex-sdk")
        ctx = _cmd_ctx(loop, "/backend")
        loop.set_session_runner_backend(ctx.key, "claude-sdk")
        out = await cmd_backend(ctx)
        assert "claude-sdk" in out.content

    @pytest.mark.asyncio
    async def test_switch_to_native(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="codex-sdk")
        ctx = _cmd_ctx(loop, "/backend native", "native")
        out = await cmd_backend(ctx)
        assert "native" in out.content
        assert loop.get_session_runner_backend(ctx.key) == "native"

    @pytest.mark.asyncio
    async def test_switch_to_codex(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="native")
        ctx = _cmd_ctx(loop, "/backend codex-sdk", "codex-sdk")
        out = await cmd_backend(ctx)
        assert "codex-sdk" in out.content
        assert loop.get_session_runner_backend(ctx.key) == "codex-sdk"

    @pytest.mark.asyncio
    async def test_switch_to_claude(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="native")
        ctx = _cmd_ctx(loop, "/backend claude-sdk", "claude-sdk")
        out = await cmd_backend(ctx)
        assert "claude-sdk" in out.content
        assert loop.get_session_runner_backend(ctx.key) == "claude-sdk"

    @pytest.mark.asyncio
    async def test_reset_returns_to_default(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="codex-sdk")
        ctx = _cmd_ctx(loop, "/backend reset", "reset")
        loop.set_session_runner_backend(ctx.key, "claude-sdk")
        out = await cmd_backend(ctx)
        assert "codex-sdk" in out.content
        assert loop.get_session_runner_backend(ctx.key) == "codex-sdk"

    @pytest.mark.asyncio
    async def test_invalid_backend_rejected(self, tmp_path):
        loop = _make_loop(tmp_path)
        ctx = _cmd_ctx(loop, "/backend bogus", "bogus")
        out = await cmd_backend(ctx)
        assert "Unknown" in out.content or "bogus" in out.content

    @pytest.mark.asyncio
    async def test_switch_is_per_session(self, tmp_path):
        loop = _make_loop(tmp_path, runner_backend="native")
        ctx_a = _cmd_ctx(loop, "/backend codex-sdk", "codex-sdk")
        # Different session key via different chat_id
        msg_b = InboundMessage(channel="cli", sender_id="user", chat_id="other", content="/backend")
        ctx_b = CommandContext(msg=msg_b, session=None, key=msg_b.session_key, raw="/backend", args="", loop=loop)
        # Switch session A
        await cmd_backend(ctx_a)
        assert loop.get_session_runner_backend(ctx_a.key) == "codex-sdk"
        # Session B unaffected
        assert loop.get_session_runner_backend(ctx_b.key) == "native"


# ── SDKRunner timeout ───────────────────────────────────────────────


# ── SDKRunner error handling ────────────────────────────────────────


class _ErrorFakeSDKRunner(SDKRunner):
    backend_name = "error-fake"

    def __init__(self):
        pass

    async def _run_turn(self, **kwargs):
        raise RuntimeError("SDK exploded")

    async def _interrupt_turn(self, session_key: str) -> None:
        pass

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def set_model(self, session_key: str, model: str) -> None:
        pass

    async def evict_stale(self, idle_timeout_s: float) -> int:
        return 0

    async def shutdown(self) -> None:
        pass


class TestSDKRunnerErrorHandling:
    @pytest.mark.asyncio
    async def test_exception_returns_error_result(self):
        runner = _ErrorFakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        result = await runner.run(spec)
        assert result.stop_reason == "error"
        assert "SDK exploded" in (result.error or "")

    @pytest.mark.asyncio
    async def test_no_assistant_appended_on_error(self):
        runner = _ErrorFakeSDKRunner()
        initial = [{"role": "user", "content": "Hi"}]
        spec = AgentRunSpec(
            initial_messages=initial,
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        result = await runner.run(spec)
        assert len(result.messages) == 1
        assert result.messages[0] == {"role": "user", "content": "Hi"}


# ── SDKRunner cancelled error propagation ───────────────────────────


class _CancelledFakeSDKRunner(SDKRunner):
    backend_name = "cancel-fake"

    def __init__(self):
        self.interrupt_called = False

    async def _run_turn(self, **kwargs):
        raise asyncio.CancelledError()

    async def _interrupt_turn(self, session_key: str) -> None:
        self.interrupt_called = True

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def set_model(self, session_key: str, model: str) -> None:
        pass

    async def evict_stale(self, idle_timeout_s: float) -> int:
        return 0

    async def shutdown(self) -> None:
        pass


class TestSDKRunnerCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_error_reraised(self):
        runner = _CancelledFakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        with pytest.raises(asyncio.CancelledError):
            await runner.run(spec)
        assert runner.interrupt_called


# ── SDKRunner lifecycle: evict_stale + shutdown ─────────────────────


class _LifecycleFakeSDKRunner(SDKRunner):
    backend_name = "lifecycle-fake"

    def __init__(self):
        self.evict_count = 0
        self.shutdown_called = False

    async def _run_turn(self, **kwargs):
        return _TurnResult(final_content="ok")

    async def _interrupt_turn(self, session_key: str) -> None:
        pass

    async def list_models(self) -> list[dict[str, Any]]:
        return []

    async def set_model(self, session_key: str, model: str) -> None:
        pass

    async def evict_stale(self, idle_timeout_s: float) -> int:
        self.evict_count += 1
        return 3

    async def shutdown(self) -> None:
        self.shutdown_called = True


class TestSDKRunnerLifecycle:
    @pytest.mark.asyncio
    async def test_evict_stale_returns_count(self):
        runner = _LifecycleFakeSDKRunner()
        count = await runner.evict_stale(60.0)
        assert count == 3
        assert runner.evict_count == 1

    @pytest.mark.asyncio
    async def test_shutdown_called(self):
        runner = _LifecycleFakeSDKRunner()
        await runner.shutdown()
        assert runner.shutdown_called


# ── SDKRunnerConfig defaults ────────────────────────────────────────


class TestSDKRunnerConfig:
    def test_default_values(self):
        cfg = SDKRunnerConfig()
        assert cfg.proxy is None
        assert cfg.codex_bin is None
        assert cfg.codex_model is None
        assert cfg.codex_sandbox == "workspace_write"
        assert cfg.codex_approval_mode == "auto_review"
        assert cfg.codex_base_instructions is None
        assert cfg.claude_model is None
        assert cfg.claude_permission_mode == "acceptEdits"
        assert cfg.claude_api_key is None
        assert cfg.claude_base_url is None
        assert cfg.claude_max_turns == 200
        assert cfg.claude_system_prompt is None
        assert cfg.session_idle_timeout_minutes == 60
