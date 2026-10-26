"""Claude via Claude Code headless mode (`claude -p`) on a Claude subscription.

Limitation (stated in the README/report): Claude Code adds its own system
prompt, so these are not raw-API results, and temperature cannot be set.

Flags follow the Claude Code CLI docs (checked 2026-10-26, CLI 2.1.283):
  -p                       non-interactive, single response
  --output-format json     full JSON result (saved verbatim)
  --model <name>           explicit model
  --tools ""               disable all built-in tools
  --strict-mcp-config      no MCP servers (none passed with --mcp-config)
  --disable-slash-commands no skills / commands
  --max-turns 1            single agentic turn
  --no-session-persistence nothing saved to disk
and isolation mode "bare" (--bare) or "safe" (--safe-mode). The docs say bare
mode never reads OAuth credentials, so it cannot use a subscription login;
"safe" disables CLAUDE.md, skills, plugins, hooks, MCP and auto memory while
keeping normal authentication. Every call runs from an empty directory.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .. import config
from .base import CallResult, FatalError, Provider, ProviderUnavailable, QuotaExhausted, RetryableError

ISOLATION_FLAGS = {"bare": ["--bare"], "safe": ["--safe-mode"]}


def build_command(exe: str, model: str, isolation: str) -> list[str]:
    return [exe, "-p", *ISOLATION_FLAGS[isolation], "--tools", "", "--strict-mcp-config",
            "--disable-slash-commands", "--max-turns", "1", "--no-session-persistence",
            "--output-format", "json", "--model", model]


def classify(raw: dict) -> None:
    """Raise the right error for a JSON result with is_error=true."""
    if not raw.get("is_error"):
        return
    msg = str(raw.get("result", ""))
    low = msg.lower()
    if "not logged in" in low or "authenticate" in low or "/login" in low:
        raise ProviderUnavailable(f"claude auth: {msg[:200]}")
    if "limit" in low and ("usage" in low or "reach" in low or "reset" in low):
        raise QuotaExhausted(f"claude usage limit: {msg[:200]}")
    if raw.get("terminal_reason") == "api_error" or "overloaded" in low or "rate limit" in low:
        raise RetryableError(f"claude api error: {msg[:200]}")
    raise FatalError(f"claude error: {msg[:200]}")


class ClaudeCodeProvider(Provider):
    family = "claude"
    supports_temperature = False

    def __init__(self, model, limits, isolation: str = "safe",
                 sandbox: Path = config.CLAUDE_SANDBOX_DIR, **kw):
        kw.setdefault("timeout", 300.0)
        super().__init__(model, limits, **kw)
        self.isolation = isolation
        self.sandbox = sandbox
        self.exe = shutil.which("claude") or "claude"

    def _ensure_empty_sandbox(self) -> None:
        self.sandbox.mkdir(parents=True, exist_ok=True)
        if any(self.sandbox.iterdir()):
            raise RuntimeError(f"Claude sandbox dir is not empty: {self.sandbox}")

    def _call(self, prompt, temperature):
        self._ensure_empty_sandbox()
        cmd = build_command(self.exe, self.model, self.isolation)
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  encoding="utf-8", cwd=self.sandbox, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise RetryableError("claude -p timed out") from None
        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RetryableError(f"claude -p exit {proc.returncode}, non-JSON stdout: "
                                 f"{proc.stdout[:200]!r} stderr: {proc.stderr[:200]!r}") from None
        classify(raw)
        return self.parse(raw)

    @staticmethod
    def parse(raw: dict) -> CallResult:
        usage = raw.get("usage") or {}
        models = list((raw.get("modelUsage") or {}).keys())
        inp = sum(usage.get(k) or 0 for k in
                  ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        return CallResult(
            text=raw.get("result"), raw=raw,
            model_reported=",".join(models) or None,
            input_tokens=inp or None, output_tokens=usage.get("output_tokens"),
            meta={"total_cost_usd_estimate": raw.get("total_cost_usd"),
                  "num_turns": raw.get("num_turns")},
        )
