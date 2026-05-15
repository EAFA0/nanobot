"""thinklog.py — sidecar trace logger for nanobot.

Loaded via PYTHONSTARTUP.  Patches ONLY SessionManager.save to write a
``<session_key>.fulltraces.jsonl`` sidecar file alongside each session.

Does NOT touch get_history, get_messages, or any other method.
Does NOT strip or alter any data in the main flow.
"""

import json
import re
import sys
from datetime import datetime, timezone


def _extract_thinking_blocks(content: str) -> list[str]:
    """Extract <thinking>...</thinking> blocks from message content."""
    if not content or not isinstance(content, str):
        return []
    return re.findall(r"<thinking>(.*?)</thinking>", content, re.DOTALL)


def _patch():
    from nanobot.session.manager import SessionManager

    _original_save = SessionManager.save

    def _save_with_sidecar(self, session, *, fsync=False):
        # ── sidecar: write new qualifying assistant messages ──────────
        try:
            session_path = self._get_session_path(session.key)
            sidecar_path = session_path.with_suffix(".fulltraces.jsonl")

            # Determine the starting index.  Use the session-attached tracker
            # for the fast path, but fall back to counting sidecar lines on
            # cold start (after service restart the tracker is lost).
            last_idx = getattr(session, "_thinklog_last_idx", 0)
            if last_idx == 0 and sidecar_path.exists():
                # Count qualifying messages already persisted in the sidecar.
                try:
                    sidecar_lines = 0
                    with open(sidecar_path, "r", encoding="utf-8") as sc:
                        for _line in sc:
                            if _line.strip():
                                sidecar_lines += 1
                except Exception:
                    sidecar_lines = 0

                # Walk session.messages and skip the first sidecar_lines
                # qualifying messages to avoid duplicates.
                q_seen = 0
                for i, msg in enumerate(session.messages):
                    if msg.get("role") != "assistant":
                        continue
                    reasoning = msg.get("reasoning_content", "")
                    tool_calls = msg.get("tool_calls")
                    if reasoning or tool_calls:
                        q_seen += 1
                        if q_seen == sidecar_lines:
                            last_idx = i + 1
                            break

            new_messages = session.messages[last_idx:]

            if new_messages:
                with open(sidecar_path, "a", encoding="utf-8") as f:
                    for msg in new_messages:
                        if msg.get("role") != "assistant":
                            continue
                        reasoning = msg.get("reasoning_content", "")
                        tool_calls = msg.get("tool_calls")
                        if not reasoning and not tool_calls:
                            continue

                        content = msg.get("content", "")
                        record = {
                            "timestamp": msg.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat(),
                            ),
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning,
                            "thinking_blocks": _extract_thinking_blocks(content),
                            "tool_calls": tool_calls,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Update the tracker even when there are no new messages so the
            # next save starts from the correct position.
            session._thinklog_last_idx = len(session.messages)
        except Exception:
            # Sidecar is best-effort; never let it break the main save path.
            pass

        # ── always call the original save ─────────────────────────────
        return _original_save(self, session, fsync=fsync)

    SessionManager.save = _save_with_sidecar


try:
    _patch()
    print("[thinklog] sidecar trace logger active", file=sys.stderr)
except Exception:
    # If nanobot isn't available yet (e.g. during uv tool install), fail
    # silently.  The script is harmless to load even without nanobot.
    pass
