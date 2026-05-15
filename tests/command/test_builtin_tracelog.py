"""Tests for /log and /think commands (no nanobot core imports)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import cmd_log, cmd_think
from nanobot.command.router import CommandContext
from nanobot.session.manager import SessionManager


def _make_ctx(
    raw: str,
    workspace: Path,
    key: str = "cli:direct",
    *,
    args: str = "",
) -> CommandContext:
    msg = InboundMessage(
        channel="cli", sender_id="u1", chat_id="direct", content=raw,
    )
    sessions = SimpleNamespace(workspace=workspace)
    loop = SimpleNamespace(sessions=sessions)
    return CommandContext(
        msg=msg, session=None, key=key, raw=raw, args=args, loop=loop,
    )


def _write_sidecar(workspace: Path, key: str, entries: list[dict]) -> Path:
    safe_key = SessionManager.safe_key(key)
    path = workspace / "sessions" / f"{safe_key}.fulltraces.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


# ── /log tests ──────────────────────────────────────────────────────────


class TestCmdLog:
    @pytest.mark.asyncio
    async def test_no_sidecar_file(self, tmp_path: Path) -> None:
        ctx = _make_ctx("/log", tmp_path)
        out = await cmd_log(ctx)
        assert "No traces found" in out.content

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [])
        ctx = _make_ctx("/log", tmp_path)
        out = await cmd_log(ctx)
        assert "No more entries" in out.content

    @pytest.mark.asyncio
    async def test_single_entry(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {
                "timestamp": "2026-05-15T10:30:00",
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "",
                "tool_calls": [],
            },
        ])
        ctx = _make_ctx("/log", tmp_path)
        out = await cmd_log(ctx)
        assert "[#1]" in out.content
        assert "10:30" in out.content
        assert "Hello" in out.content

    @pytest.mark.asyncio
    async def test_entry_with_tool_calls(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {
                "timestamp": "2026-05-15T10:31:00",
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {"function": {"name": "read_file"}},
                    {"function": {"name": "bash"}},
                ],
            },
        ])
        ctx = _make_ctx("/log", tmp_path)
        out = await cmd_log(ctx)
        assert "[#1]" in out.content
        assert "10:31" in out.content
        assert "read_file,bash" in out.content

    @pytest.mark.asyncio
    async def test_multi_entry_reverse_order(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {"timestamp": "2026-05-15T10:00:00", "role": "assistant", "content": "first", "tool_calls": []},
            {"timestamp": "2026-05-15T10:01:00", "role": "assistant", "content": "second", "tool_calls": []},
            {"timestamp": "2026-05-15T10:02:00", "role": "assistant", "content": "third", "tool_calls": []},
        ])
        ctx = _make_ctx("/log", tmp_path)
        out = await cmd_log(ctx)
        lines = [l for l in out.content.split("\n") if l.startswith("[#")]
        assert len(lines) == 3
        assert "[#3]" in lines[0]
        assert "[#2]" in lines[1]
        assert "[#1]" in lines[2]

    @pytest.mark.asyncio
    async def test_pagination_footer(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {"timestamp": "2026-05-15T10:00:00", "role": "assistant", "content": "a", "tool_calls": []},
            {"timestamp": "2026-05-15T10:01:00", "role": "assistant", "content": "b", "tool_calls": []},
            {"timestamp": "2026-05-15T10:02:00", "role": "assistant", "content": "c", "tool_calls": []},
        ])
        ctx = _make_ctx("/log 2", tmp_path, args="2")
        out = await cmd_log(ctx)
        assert "共 3 条" in out.content
        assert "/log 2 2" in out.content

    @pytest.mark.asyncio
    async def test_start_offset(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {"timestamp": "2026-05-15T10:00:00", "role": "assistant", "content": "first", "tool_calls": []},
            {"timestamp": "2026-05-15T10:01:00", "role": "assistant", "content": "second", "tool_calls": []},
            {"timestamp": "2026-05-15T10:02:00", "role": "assistant", "content": "third", "tool_calls": []},
        ])
        ctx = _make_ctx("/log 2 1", tmp_path, args="2 1")
        out = await cmd_log(ctx)
        lines = [l for l in out.content.split("\n") if l.startswith("[#")]
        assert len(lines) == 2
        assert "[#2]" in lines[0]
        assert "[#1]" in lines[1]

    @pytest.mark.asyncio
    async def test_start_beyond_total(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {"timestamp": "2026-05-15T10:00:00", "role": "assistant", "content": "x", "tool_calls": []},
        ])
        ctx = _make_ctx("/log 5 10", tmp_path, args="5 10")
        out = await cmd_log(ctx)
        assert "No more entries" in out.content

    @pytest.mark.asyncio
    async def test_long_content_truncated(self, tmp_path: Path) -> None:
        long = "x" * 100
        _write_sidecar(tmp_path, "cli:direct", [
            {"timestamp": "2026-05-15T10:00:00", "role": "assistant", "content": long, "tool_calls": []},
        ])
        ctx = _make_ctx("/log", tmp_path)
        out = await cmd_log(ctx)
        assert long[:50] in out.content
        assert "…" in out.content


# ── /think tests ─────────────────────────────────────────────────────────


class TestCmdThink:
    @pytest.mark.asyncio
    async def test_no_sidecar_file(self, tmp_path: Path) -> None:
        ctx = _make_ctx("/think", tmp_path)
        out = await cmd_think(ctx)
        assert "No traces found" in out.content

    @pytest.mark.asyncio
    async def test_no_reasoning(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {
                "timestamp": "2026-05-15T10:30:00",
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "",
                "tool_calls": [],
            },
        ])
        ctx = _make_ctx("/think", tmp_path)
        out = await cmd_think(ctx)
        assert "no reasoning_content" in out.content

    @pytest.mark.asyncio
    async def test_shows_reasoning(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {
                "timestamp": "2026-05-15T10:30:00",
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "Because 1+1=2",
                "tool_calls": [],
            },
        ])
        ctx = _make_ctx("/think", tmp_path)
        out = await cmd_think(ctx)
        assert "[#1] 2026-05-15T10:30" in out.content
        assert "Because 1+1=2" in out.content

    @pytest.mark.asyncio
    async def test_index_from_tail(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {"timestamp": "2026-05-15T10:00:00", "role": "assistant", "content": "a", "reasoning_content": "r1", "tool_calls": []},
            {"timestamp": "2026-05-15T10:01:00", "role": "assistant", "content": "b", "reasoning_content": "r2", "tool_calls": []},
            {"timestamp": "2026-05-15T10:02:00", "role": "assistant", "content": "c", "reasoning_content": "r3", "tool_calls": []},
        ])
        ctx = _make_ctx("/think 2", tmp_path, args="2")
        out = await cmd_think(ctx)
        assert "[#2]" in out.content
        assert "r2" in out.content

    @pytest.mark.asyncio
    async def test_reasoning_overflow_footer(self, tmp_path: Path) -> None:
        long_reasoning = "x" * 2500
        _write_sidecar(tmp_path, "cli:direct", [
            {
                "timestamp": "2026-05-15T10:00:00",
                "role": "assistant",
                "content": "c",
                "reasoning_content": long_reasoning,
                "tool_calls": [],
            },
        ])
        ctx = _make_ctx("/think", tmp_path)
        out = await cmd_think(ctx)
        assert "/think 1 1500 继续" in out.content
        assert "[1500/2500 chars" in out.content

    @pytest.mark.asyncio
    async def test_reasoning_char_offset(self, tmp_path: Path) -> None:
        reasoning = "0123456789" + "x" * 2000
        _write_sidecar(tmp_path, "cli:direct", [
            {
                "timestamp": "2026-05-15T10:00:00",
                "role": "assistant",
                "content": "c",
                "reasoning_content": reasoning,
                "tool_calls": [],
            },
        ])
        ctx = _make_ctx("/think 1 5", tmp_path, args="1 5")
        out = await cmd_think(ctx)
        assert "56789" in out.content

    @pytest.mark.asyncio
    async def test_idx_out_of_range(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "cli:direct", [
            {"timestamp": "2026-05-15T10:00:00", "role": "assistant", "content": "a", "reasoning_content": "r", "tool_calls": []},
        ])
        ctx = _make_ctx("/think 99", tmp_path, args="99")
        out = await cmd_think(ctx)
        assert "Index out of range" in out.content


# ── key format tests ─────────────────────────────────────────────────────


class TestCmdLogKeyFormat:
    @pytest.mark.asyncio
    async def test_colon_in_session_key(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path, "telegram:12345", [
            {
                "timestamp": "2026-05-15T12:00:00",
                "role": "assistant",
                "content": "hi",
                "tool_calls": [],
            },
        ])
        ctx = _make_ctx("/log", tmp_path, key="telegram:12345")
        out = await cmd_log(ctx)
        assert "[#1]" in out.content
        assert "hi" in out.content
