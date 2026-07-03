"""Tests for SDK runner seam — verifies that AgentLoop can dispatch to a
non-native runner via the runner_backend config.

Uses a FakeSDKRunner that returns canned responses, proving the seam works
without requiring real SDK packages.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import pytest

from nanobot.agent.runner import AgentRunResult, AgentRunSpec
from nanobot.agent.sdk_runner import SDKRunner, _TurnResult


class FakeSDKRunner(SDKRunner):
    """Minimal SDK runner that returns canned text for testing the seam."""

    backend_name = "fake"

    def __init__(self, response: str = "Hello from relaybot FakeBackend."):
        self._response = response
        self.last_prompt: str | None = None
        self.last_session_key: str | None = None
        self.last_cwd: str | None = None
        self.last_model: str | None = None
        self.turn_count = 0
        self.interrupt_called = False
        self.evict_called = False
        self.shutdown_called = False

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
        self.turn_count += 1
        self.last_prompt = prompt
        self.last_session_key = session_key
        self.last_cwd = cwd
        self.last_model = model
        await on_delta(self._response)
        return _TurnResult(
            final_content=self._response,
            tools_used=[],
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            stop_reason="completed",
        )

    async def _interrupt_turn(self, session_key: str) -> None:
        self.interrupt_called = True

    async def list_models(self) -> list[dict[str, Any]]:
        return [
            {"id": "fake-model-1", "name": "Fake Model 1", "description": "Test model", "is_default": True},
            {"id": "fake-model-2", "name": "Fake Model 2", "description": "Another test model", "is_default": False},
        ]

    async def set_model(self, session_key: str, model: str) -> None:
        self.last_set_model = model

    async def evict_stale(self, idle_timeout_s: float) -> int:
        self.evict_called = True
        return 0

    async def shutdown(self) -> None:
        self.shutdown_called = True


class TestFakeSDKRunner:
    """Unit tests for FakeSDKRunner directly."""

    @pytest.mark.asyncio
    async def test_returns_canned_response(self):
        runner = FakeSDKRunner("Hello test")
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test-model",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        result = await runner.run(spec)
        assert result.final_content == "Hello test"
        assert result.stop_reason == "completed"
        assert result.tools_used == []
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}

    @pytest.mark.asyncio
    async def test_extracts_user_prompt(self):
        runner = FakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        await runner.run(spec)
        assert runner.last_prompt == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_appends_assistant_to_messages(self):
        runner = FakeSDKRunner("Reply text")
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        result = await runner.run(spec)
        assert len(result.messages) == 2
        assert result.messages[0] == {"role": "user", "content": "Hi"}
        assert result.messages[1] == {"role": "assistant", "content": "Reply text"}

    @pytest.mark.asyncio
    async def test_no_user_message_returns_error(self):
        runner = FakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "system", "content": "System only"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        result = await runner.run(spec)
        assert result.stop_reason == "error"
        assert "No user message" in (result.error or "")

    @pytest.mark.asyncio
    async def test_handles_cancelled_error(self):
        runner = FakeSDKRunner()

        async def _cancelled_turn(**kwargs):
            raise asyncio.CancelledError()

        runner._run_turn = _cancelled_turn  # type: ignore[assignment]
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

    @pytest.mark.asyncio
    async def test_session_key_passed_through(self):
        runner = FakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
            session_key="my-session-123",
        )
        await runner.run(spec)
        assert runner.last_session_key == "my-session-123"

    @pytest.mark.asyncio
    async def test_model_passed_through(self):
        runner = FakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="gpt-4o",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        await runner.run(spec)
        assert runner.last_model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_workspace_as_cwd(self):
        runner = FakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
            workspace="/tmp/test-workspace",
        )
        await runner.run(spec)
        assert runner.last_cwd == "/tmp/test-workspace"

    @pytest.mark.asyncio
    async def test_multimodal_content_text_extracted(self):
        runner = FakeSDKRunner()
        spec = AgentRunSpec(
            initial_messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        result = await runner.run(spec)
        assert runner.last_prompt == "Describe this image"
        assert result.stop_reason == "completed"


class TestSDKRunnerSeamIntegration:
    """Integration tests: verify hook lifecycle and error handling in SDKRunner."""

    @pytest.mark.asyncio
    async def test_hook_lifecycle_called(self):
        """Verify before_run, on_stream, on_stream_end, after_run, emit_reasoning_end are all called."""

        class CaptureHook:
            def __init__(self):
                self.deltas: list[str] = []
                self.before_run_called = False
                self.after_run_called = False
                self.stream_end_called = False
                self.reasoning_end_called = False

            async def before_run(self, ctx):
                self.before_run_called = True

            async def on_stream(self, ctx, delta):
                self.deltas.append(delta)

            async def on_stream_end(self, ctx, *, resuming):
                self.stream_end_called = True

            async def after_run(self, ctx):
                self.after_run_called = True

            async def emit_reasoning_end(self):
                self.reasoning_end_called = True

            def finalize_content(self, ctx, content):
                return content

        hook = CaptureHook()
        runner = FakeSDKRunner("Hook test reply")
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
            hook=hook,
        )
        result = await runner.run(spec)

        assert hook.before_run_called
        assert hook.after_run_called
        assert hook.stream_end_called
        assert hook.reasoning_end_called
        assert len(hook.deltas) > 0
        assert result.final_content == "Hook test reply"

    @pytest.mark.asyncio
    async def test_sdk_error_gracefully_handled(self):
        """If _run_turn raises, return error result, don't crash."""

        class ErrorRunner(FakeSDKRunner):
            async def _run_turn(self, **kwargs):
                raise RuntimeError("SDK connection failed")

        runner = ErrorRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "Hi"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
        )
        result = await runner.run(spec)
        assert result.stop_reason == "error"
        assert "SDK connection failed" in (result.error or "")
        # messages should still contain initial_messages (no assistant appended)
        assert len(result.messages) == len(spec.initial_messages)

    @pytest.mark.asyncio
    async def test_tool_events_flow_through_hook(self):
        """Verify tool start/end callbacks reach the hook."""

        class ToolCaptureHook:
            def __init__(self):
                self.tool_starts: list[str] = []
                self.tool_ends: list[str] = []

            async def before_run(self, ctx):
                pass

            async def after_run(self, ctx):
                pass

            async def before_execute_tools(self, ctx):
                for tc in ctx.tool_calls:
                    self.tool_starts.append(tc.name)

            async def after_iteration(self, ctx):
                for tc in ctx.tool_calls:
                    self.tool_ends.append(tc.name)

            async def on_stream(self, ctx, delta):
                pass

            async def on_stream_end(self, ctx, *, resuming):
                pass

            async def emit_reasoning_end(self):
                pass

            def finalize_content(self, ctx, content):
                return content

        class ToolRunner(FakeSDKRunner):
            async def _run_turn(self, **kwargs):
                await kwargs["on_tool_start"]("Bash", {"command": "ls"})
                await kwargs["on_tool_end"]("Bash", True, "file1.txt\nfile2.txt")
                await kwargs["on_delta"]("Done.")
                return _TurnResult(
                    final_content="Done.",
                    tools_used=["Bash"],
                    stop_reason="completed",
                )

        hook = ToolCaptureHook()
        runner = ToolRunner()
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "List files"}],
            tools=None,  # type: ignore[arg-type]
            model="test",
            max_iterations=10,
            max_tool_result_chars=1000,
            hook=hook,
        )
        result = await runner.run(spec)
        assert "Bash" in hook.tool_starts
        assert "Bash" in hook.tool_ends
        assert result.tools_used == ["Bash"]
        assert result.final_content == "Done."
