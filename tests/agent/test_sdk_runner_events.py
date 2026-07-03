"""Tests for CodexSDKRunner and ClaudeSDKRunner event parsing.

These tests mock the SDK internals and feed events with the ACTUAL method names
that openai-codex / claude-agent-sdk emit. If the event name checks in the
runners drift from reality, these tests catch it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.sdk_runner.codex import CodexSDKRunner
from nanobot.agent.sdk_runner.claude import ClaudeSDKRunner
from nanobot.config.schema import SDKRunnerConfig


# ── Helpers ─────────────────────────────────────────────────────────


def _make_event(method: str, payload):
    """Create a mock SDK event object."""
    return SimpleNamespace(method=method, payload=payload)


async def _async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item


# ── CodexSDKRunner event parsing ────────────────────────────────────


class _MockTurn:
    def __init__(self, events):
        self._events = events
        self.interrupt = AsyncMock()

    def stream(self):
        return _async_iter(self._events)


def _setup_codex_runner(events, config=None):
    """Create a CodexSDKRunner with mocked internals, bypassing _ensure_codex."""
    if config is None:
        config = SDKRunnerConfig()

    runner = CodexSDKRunner(config)

    # Bypass _ensure_codex by directly setting _codex
    mock_codex = MagicMock()
    mock_codex.__aenter__ = AsyncMock(return_value=mock_codex)
    mock_codex.__aexit__ = AsyncMock()
    runner._codex = mock_codex

    # Mock thread.turn to return our mock turn
    mock_thread = MagicMock()
    mock_thread.turn = AsyncMock(return_value=_MockTurn(events))
    runner._threads["test"] = mock_thread

    # Also mock _ensure_codex to return mock_codex (in case it's called)
    runner._ensure_codex = AsyncMock(return_value=mock_codex)  # type: ignore[assignment]

    return runner, mock_thread


class TestCodexEventParsing:
    """Verify CodexSDKRunner correctly maps SDK events to callbacks."""

    @pytest.mark.asyncio
    async def test_agent_message_delta_triggers_on_delta(self):
        """`item/agentMessage/delta` events must reach on_delta."""
        events = [
            _make_event("turn/started", None),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Hello ")),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="world!")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        deltas = []
        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(side_effect=lambda d: deltas.append(d)),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert deltas == ["Hello ", "world!"]
        assert result.final_content == "Hello world!"
        assert result.stop_reason == "completed"

    @pytest.mark.asyncio
    async def test_command_execution_item_triggers_tool_callbacks(self):
        """`item/started` + `item/completed` with commandExecution must fire on_tool_start/end."""
        events = [
            _make_event("turn/started", None),
            _make_event("item/started", SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(
                    type="commandExecution",
                    command="/bin/bash -lc 'echo hi'",
                ))
            )),
            _make_event("item/completed", SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(
                    type="commandExecution",
                    command="/bin/bash -lc 'echo hi'",
                    exit_code=0,
                    aggregated_output="hi\n",
                ))
            )),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Done.")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        tool_starts = []
        tool_ends = []

        result = await runner._run_turn(
            session_key="test",
            prompt="run echo hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(side_effect=lambda n, i: tool_starts.append((n, i))),
            on_tool_end=AsyncMock(side_effect=lambda n, s, o: tool_ends.append((n, s, o))),
            on_reasoning=AsyncMock(),
        )

        assert len(tool_starts) == 1
        assert tool_starts[0][0] == "Bash"
        assert "echo hi" in tool_starts[0][1].get("command", "")

        assert len(tool_ends) == 1
        assert tool_ends[0][0] == "Bash"
        assert tool_ends[0][1] is True  # success
        assert "hi" in tool_ends[0][2]

        assert "Bash" in result.tools_used

    @pytest.mark.asyncio
    async def test_file_change_item_maps_to_edit_tool(self):
        """fileChange item must map to Edit/MultiEdit tool name."""
        events = [
            _make_event("turn/started", None),
            _make_event("item/started", SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(
                    type="fileChange",
                    changes=[SimpleNamespace(path="/tmp/test.py")],
                ))
            )),
            _make_event("item/completed", SimpleNamespace(
                item=SimpleNamespace(root=SimpleNamespace(
                    type="fileChange",
                    changes=[SimpleNamespace(path="/tmp/test.py")],
                ))
            )),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Edited.")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        tool_starts = []
        result = await runner._run_turn(
            session_key="test",
            prompt="edit file",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(side_effect=lambda n, i: tool_starts.append(n)),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert "Edit" in tool_starts
        assert "Edit" in result.tools_used

    @pytest.mark.asyncio
    async def test_token_usage_populated(self):
        """`thread/tokenUsage/updated` must populate usage dict."""
        events = [
            _make_event("turn/started", None),
            _make_event("thread/tokenUsage/updated", SimpleNamespace(
                token_usage=SimpleNamespace(
                    last=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=50,
                        cached_input_tokens=10,
                    )
                )
            )),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Reply")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert result.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 10,
        }

    @pytest.mark.asyncio
    async def test_turn_failed_status_maps_to_error(self):
        """turn/completed with failed status must return stop_reason='error'."""
        events = [
            _make_event("turn/started", None),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(
                    status="failed",
                    error=SimpleNamespace(message="Model unavailable"),
                    items=[],
                )
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert result.stop_reason == "error"
        assert "Model unavailable" in (result.error or "")

    @pytest.mark.asyncio
    async def test_turn_interrupted_status(self):
        """turn/completed with interrupted status must return stop_reason='error'."""
        events = [
            _make_event("turn/started", None),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="interrupted", error=None, items=[])
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert result.stop_reason == "error"
        assert "interrupted" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_no_double_run_call(self):
        """After consuming turn.stream(), turn.run() must NOT be called (would hang)."""
        events = [
            _make_event("turn/started", None),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Hi")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        mock_turn = _MockTurn(events)
        mock_turn.run = AsyncMock()  # track if run() is called

        runner, mock_thread = _setup_codex_runner([])
        mock_thread.turn = AsyncMock(return_value=mock_turn)

        await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        mock_turn.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_from_spec_not_passed_to_codex(self):
        """The model param from AgentRunSpec (native provider model) must NOT be
        passed to codex — it's usually something like 'minimax-m3' which codex
        doesn't understand. Only codex_model from SDKRunnerConfig should be used."""
        events = [
            _make_event("turn/started", None),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Hi")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        runner, mock_thread = _setup_codex_runner(events)

        await runner._run_turn(
            session_key="test",
            prompt="hi",
            model="minimax-m3",  # This is the native provider model
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        # turn should NOT have been called with model="minimax-m3"
        turn_kwargs = mock_thread.turn.call_args
        assert "model" not in (turn_kwargs.kwargs or {})

    @pytest.mark.asyncio
    async def test_config_codex_model_passed_through(self):
        """If codex_model is set in config, it should be passed to codex."""
        events = [
            _make_event("turn/started", None),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Hi")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        config = SDKRunnerConfig(codex_model="gpt-5.5-2026-04-24")
        runner, mock_thread = _setup_codex_runner(events, config=config)

        await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        turn_kwargs = mock_thread.turn.call_args
        assert turn_kwargs.kwargs.get("model") == "gpt-5.5-2026-04-24"

    @pytest.mark.asyncio
    async def test_active_turns_tracking(self):
        """_active_turns must be populated during turn and cleared after."""
        events = [
            _make_event("turn/started", None),
            _make_event("item/agentMessage/delta", SimpleNamespace(delta="Hi")),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        assert "test" not in runner._active_turns
        await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )
        assert "test" not in runner._active_turns  # cleared after

    @pytest.mark.asyncio
    async def test_wrong_event_names_not_matched(self):
        """If event names drift (e.g. missing 'item/' prefix), callbacks must NOT fire.
        This test would PASS with the old buggy code (names were wrong so nothing fired)."""
        events = [
            _make_event("turn/started", None),
            # These are the WRONG names (what our code used to check)
            _make_event("agentMessage/delta", SimpleNamespace(delta="Hello")),
            _make_event("reasoning/text/delta", SimpleNamespace(delta="thinking")),
            _make_event("thread/tokenUsageUpdated", SimpleNamespace(usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cached_tokens=0
            ))),
            _make_event("turn/completed", SimpleNamespace(
                turn=SimpleNamespace(status="completed", error=None, items=[])
            )),
        ]
        runner, _ = _setup_codex_runner(events)

        deltas = []
        reasoning = []
        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(side_effect=lambda d: deltas.append(d)),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(side_effect=lambda t: reasoning.append(t)),
        )

        # With wrong event names, nothing should have been captured
        assert deltas == []
        assert reasoning == []
        assert result.final_content is None
        # This test documents: if SDK changes event names, our runner silently fails.
        # The correct names are tested in test_agent_message_delta_triggers_on_delta above.


class TestCodexReasoningEffortPatch:
    """The codex binary returns reasoningEffort='max' but the SDK enum only
    accepts none/minimal/low/medium/high/xhigh. Our patch adds 'max' → xhigh."""

    def test_patch_adds_max_to_enum(self):
        from openai_codex.generated.v2_all import ReasoningEffort

        # Save original state
        had_max = "max" in ReasoningEffort._value2member_map_
        original_map = dict(ReasoningEffort._value2member_map_)

        try:
            if had_max:
                del ReasoningEffort._value2member_map_["max"]

            assert "max" not in ReasoningEffort._value2member_map_

            CodexSDKRunner._patch_reasoning_effort_enum()

            assert "max" in ReasoningEffort._value2member_map_
            assert ReasoningEffort._value2member_map_["max"] == ReasoningEffort.xhigh
        finally:
            if not had_max and "max" in ReasoningEffort._value2member_map_:
                del ReasoningEffort._value2member_map_["max"]
            else:
                ReasoningEffort._value2member_map_.clear()
                ReasoningEffort._value2member_map_.update(original_map)

    def test_patch_is_idempotent(self):
        """Calling patch twice must not raise or duplicate."""
        from openai_codex.generated.v2_all import ReasoningEffort

        original_map = dict(ReasoningEffort._value2member_map_)
        try:
            CodexSDKRunner._patch_reasoning_effort_enum()
            count_after_first = len(ReasoningEffort._value2member_map_)
            CodexSDKRunner._patch_reasoning_effort_enum()
            assert len(ReasoningEffort._value2member_map_) == count_after_first
        finally:
            ReasoningEffort._value2member_map_.clear()
            ReasoningEffort._value2member_map_.update(original_map)


# ── ClaudeSDKRunner event parsing ───────────────────────────────────


class _MockClient:
    def __init__(self, responses):
        self._responses = responses
        self.__aenter__ = AsyncMock(return_value=self)
        self.__aexit__ = AsyncMock(return_value=None)
        self.query = AsyncMock()
        self.interrupt = AsyncMock()

    def receive_response(self):
        return _async_iter(self._responses)


def _setup_claude_runner(responses, config=None):
    """Create a ClaudeSDKRunner with mocked client, bypassing _ensure_client."""
    if config is None:
        config = SDKRunnerConfig()

    runner = ClaudeSDKRunner(config)
    mock_client = _MockClient(responses)
    runner._clients["test"] = mock_client
    runner._last_activity["test"] = 0  # prevent eviction during test
    return runner, mock_client


# Real classes for Claude message type checks (type(msg).__name__)
class AssistantMessage:
    def __init__(self, content):
        self.content = content


class UserMessage:
    def __init__(self, content):
        self.content = content


class ResultMessage:
    def __init__(self, stop_reason="end_turn", is_error=False, usage=None):
        self.stop_reason = stop_reason
        self.is_error = is_error
        self.usage = usage


class TextBlock:
    def __init__(self, text):
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking):
        self.thinking = thinking


class ToolUseBlock:
    def __init__(self, name, input):
        self.name = name
        self.input = input


class ToolResultBlock:
    def __init__(self, is_error=False, content=""):
        self.is_error = is_error
        self.content = content


class TestClaudeEventParsing:
    """Verify ClaudeSDKRunner correctly maps SDK message types to callbacks."""

    @pytest.mark.asyncio
    async def test_text_block_emits_delta(self):
        """AssistantMessage with TextBlock must trigger on_delta."""
        responses = [
            AssistantMessage(content=[TextBlock(text="Hello world")]),
            ResultMessage(
                stop_reason="end_turn",
                is_error=False,
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            ),
        ]
        runner, _ = _setup_claude_runner(responses)

        deltas = []
        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(side_effect=lambda d: deltas.append(d)),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert len(deltas) == 1
        assert deltas[0] == "Hello world"
        assert result.final_content == "Hello world"

    @pytest.mark.asyncio
    async def test_partial_messages_accumulate_deltas(self):
        """Partial messages with growing text must emit only new deltas."""
        responses = [
            AssistantMessage(content=[TextBlock(text="Hel")]),
            AssistantMessage(content=[TextBlock(text="Hello")]),
            AssistantMessage(content=[TextBlock(text="Hello world")]),
            ResultMessage(stop_reason="end_turn", is_error=False, usage=None),
        ]
        runner, _ = _setup_claude_runner(responses)

        deltas = []
        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(side_effect=lambda d: deltas.append(d)),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert deltas == ["Hel", "lo", " world"]
        assert result.final_content == "Hello world"

    @pytest.mark.asyncio
    async def test_thinking_block_triggers_reasoning(self):
        """ThinkingBlock must trigger on_reasoning."""
        responses = [
            AssistantMessage(content=[ThinkingBlock(thinking="Let me think about this...")]),
            ResultMessage(stop_reason="end_turn", is_error=False, usage=None),
        ]
        runner, _ = _setup_claude_runner(responses)

        reasoning = []
        await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(side_effect=lambda t: reasoning.append(t)),
        )

        assert reasoning == ["Let me think about this..."]

    @pytest.mark.asyncio
    async def test_tool_use_block_triggers_on_tool_start(self):
        """ToolUseBlock must trigger on_tool_start."""
        responses = [
            AssistantMessage(content=[ToolUseBlock(name="Bash", input={"command": "echo hi"})]),
            UserMessage(content=[ToolResultBlock(is_error=False, content="hi\n")]),
            ResultMessage(stop_reason="end_turn", is_error=False, usage=None),
        ]
        runner, _ = _setup_claude_runner(responses)

        tool_starts = []
        tool_ends = []
        await runner._run_turn(
            session_key="test",
            prompt="run echo",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(side_effect=lambda n, i: tool_starts.append((n, i))),
            on_tool_end=AsyncMock(side_effect=lambda n, s, o: tool_ends.append((n, s, o))),
            on_reasoning=AsyncMock(),
        )

        assert len(tool_starts) == 1
        assert tool_starts[0][0] == "Bash"
        assert tool_starts[0][1] == {"command": "echo hi"}

        assert len(tool_ends) == 1
        assert tool_ends[0][0] == "Bash"
        assert tool_ends[0][1] is True

    @pytest.mark.asyncio
    async def test_result_message_stop_reason_mapping(self):
        """ResultMessage stop_reason must map correctly."""
        test_cases = [
            ("end_turn", "completed"),
            ("max_turns", "max_iterations"),
            ("stop_sequence", "completed"),
        ]
        for sdk_reason, expected in test_cases:
            responses = [
                ResultMessage(
                    stop_reason=sdk_reason,
                    is_error=False,
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                ),
            ]
            runner, _ = _setup_claude_runner(responses)
            result = await runner._run_turn(
                session_key="test",
                prompt="hi",
                model=None,
                cwd="/tmp",
                on_delta=AsyncMock(),
                on_tool_start=AsyncMock(),
                on_tool_end=AsyncMock(),
                on_reasoning=AsyncMock(),
            )
            assert result.stop_reason == expected, f"stop_reason={sdk_reason} -> {expected}"

    @pytest.mark.asyncio
    async def test_result_message_is_error(self):
        """ResultMessage with is_error=True must set stop_reason='error'."""
        responses = [
            ResultMessage(stop_reason="end_turn", is_error=True, usage=None),
        ]
        runner, _ = _setup_claude_runner(responses)
        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert result.stop_reason == "error"

    @pytest.mark.asyncio
    async def test_usage_extracted(self):
        """ResultMessage usage must be extracted into prompt/completion tokens."""
        responses = [
            ResultMessage(
                stop_reason="end_turn",
                is_error=False,
                usage=SimpleNamespace(input_tokens=42, output_tokens=7),
            ),
        ]
        runner, _ = _setup_claude_runner(responses)
        result = await runner._run_turn(
            session_key="test",
            prompt="hi",
            model=None,
            cwd="/tmp",
            on_delta=AsyncMock(),
            on_tool_start=AsyncMock(),
            on_tool_end=AsyncMock(),
            on_reasoning=AsyncMock(),
        )

        assert result.usage == {"prompt_tokens": 42, "completion_tokens": 7}

    @pytest.mark.asyncio
    async def test_active_turns_protected_from_evict(self):
        """Sessions with active turns must NOT be evicted by evict_stale."""
        mock_client = MagicMock()
        mock_client.__aexit__ = AsyncMock()

        runner = ClaudeSDKRunner(SDKRunnerConfig())
        runner._clients["sess-active"] = mock_client
        runner._last_activity["sess-active"] = 0  # very stale
        runner._active_turns.add("sess-active")

        runner._clients["sess-idle"] = MagicMock()
        runner._clients["sess-idle"].__aexit__ = AsyncMock()
        runner._last_activity["sess-idle"] = 0  # very stale, no active turn

        evicted = await runner.evict_stale(idle_timeout_s=1.0)
        assert evicted == 1  # only idle session evicted
        assert "sess-active" in runner._clients
        assert "sess-idle" not in runner._clients

    @pytest.mark.asyncio
    async def test_client_creation_sets_last_activity(self):
        """_ensure_client must set _last_activity to prevent immediate eviction."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("claude_agent_sdk.ClaudeSDKClient", return_value=mock_client), \
             patch("claude_agent_sdk.ClaudeAgentOptions", return_value=MagicMock()):
            runner = ClaudeSDKRunner(SDKRunnerConfig())
            client = await runner._ensure_client("test-sess", "/tmp")

        assert client is mock_client
        assert "test-sess" in runner._last_activity
        assert runner._last_activity["test-sess"] > 0
