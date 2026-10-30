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
keeping normal authentication. --system-prompt replaces Claude Code's default
system prompt so Claude sees (close to) what the API models see; check the
input token count of a live call to confirm the replacement took effect.
Every call runs from an empty directory.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .. import config
from .base import CallResult, FatalError, Provider, ProviderUnavailable, QuotaExhausted, RetryableError

ISOLATION_FLAGS = {"bare": ["--bare"], "safe": ["--safe-mode"]}


def build_command(exe: str, model: str, isolation: str, system_prompt: str | None = None,
                  persist: bool = False, resume: str | None = None) -> list[str]:
    """system_prompt=None keeps Claude Code's default system prompt; a string
    (possibly empty) replaces it entirely via --system-prompt.

    Sessions are not saved (--no-session-persistence) except when a later turn
    must resume them: persist=True keeps a first turn's session, and
    resume=<session_id> continues it. Neither changes what the model is sent."""
    cmd = [exe, "-p", *ISOLATION_FLAGS[isolation], "--tools", "", "--strict-mcp-config",
           "--disable-slash-commands", "--max-turns", "1",
           "--output-format", "json", "--model", model]
    if not persist and not resume:
        cmd.append("--no-session-persistence")
    if resume:
        cmd += ["--resume", resume]
    if system_prompt is not None:
        cmd += ["--system-prompt", system_prompt]
    return cmd


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
    if raw.get("api_error_status") == 400 or "api error: 400" in low:
        raise FatalError(f"claude bad request: {msg[:200]}")  # retrying cannot fix a 400
    if raw.get("terminal_reason") == "api_error" or "overloaded" in low or "rate limit" in low:
        raise RetryableError(f"claude api error: {msg[:200]}")
    raise FatalError(f"claude error: {msg[:200]}")


class ClaudeCodeProvider(Provider):
    family = "claude"
    supports_temperature = False
    resumes_sessions = True  # a follow-up turn continues the first turn's saved session

    def __init__(self, model, limits, isolation: str = "safe", system_prompt: str | None = None,
                 sandbox: Path = config.CLAUDE_SANDBOX_DIR, **kw):
        kw.setdefault("timeout", 300.0)
        super().__init__(model, limits, **kw)
        self.isolation = isolation
        self.system_prompt = system_prompt
        self.sandbox = sandbox
        self.exe = shutil.which("claude") or "claude"

    def _ensure_empty_sandbox(self) -> None:
        self.sandbox.mkdir(parents=True, exist_ok=True)
        if any(self.sandbox.iterdir()):
            raise RuntimeError(f"Claude sandbox dir is not empty: {self.sandbox}")

    def _call(self, prompt, temperature, history=None, session=None):
        # Claude Code keeps conversation state itself: a follow-up resumes the first turn's
        # saved session instead of replaying `history` (which is kept only for the record).
        session = session or {}
        if history and not session.get("resume"):
            raise ValueError("a Claude follow-up turn needs session={'resume': <session_id>}")
        self._ensure_empty_sandbox()
        cmd = build_command(self.exe, self.model, self.isolation, self.system_prompt,
                            persist=bool(session.get("persist")), resume=session.get("resume"))
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
                  "num_turns": raw.get("num_turns"), "session_id": raw.get("session_id")},
        )
