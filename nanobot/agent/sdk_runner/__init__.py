"""SDK-based runner backends for relaybot.

Delegates turn execution to external coding agent SDKs instead of the native
AgentRunner. Two backends cover all target CLIs:

- ``codex-sdk`` — Codex / Coco (via openai-codex)
- ``claude-sdk`` — Claude Code / Relay / Seed (via claude-agent-sdk)
"""

from nanobot.agent.sdk_runner.base import SDKRunner, _TurnResult

__all__ = ["SDKRunner", "_TurnResult"]
