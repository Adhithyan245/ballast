"""Models served through the Kiro CLI (`kiro-cli chat --no-interactive`), paid with Kiro credits.

Added after the preregistration as an exploratory family (see DEVIATIONS.md section B).

Limitation, as for Claude Code: Kiro is an agent harness and adds its own context; temperature
cannot be set. Every call runs from an empty directory with no tools trusted
(`--trust-tools=`), and the full event stream is saved. Flags checked against
`kiro-cli chat --help` on 2026-10-30:
  --no-interactive --model <m> --trust-tools= --output-format stream-json --wrap never
  --resume-id <session> (pushback turn only)
The stream's `runFinished.finalText` is the model's reply. Each turn reports its credit cost in
`metadata.meteringUsage`. Kiro reports no token counts.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .. import config
from .base import CallResult, FatalError, Provider, ProviderUnavailable, QuotaExhausted, RetryableError

KIRO_SANDBOX_DIR = config.ROOT / ".kiro_sandbox"


def build_command(exe: str, model: str, prompt: str, resume: str | None = None) -> list[str]:
    cmd = [exe, "chat", "--no-interactive", "--model", model, "--trust-tools=",
           "--output-format", "stream-json", "--wrap", "never"]
    if resume:
        cmd += ["--resume-id", resume]
    return cmd + [prompt]


def parse_stream(stdout: str) -> dict:
    """Collect what the study needs from Kiro's JSON Lines event stream."""
    events, session, credits, finished = [], None, 0.0, None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(e)
        d = e.get("data") or {}
        session = d.get("sessionId") or session
        for m in d.get("meteringUsage") or []:
            if m.get("unit") == "credit":
                credits += float(m.get("value") or 0)
        if e.get("type") == "runFinished":
            finished = d
    return {"events": events, "session_id": session, "credits": credits, "finished": finished}


def classify(proc_returncode: int, parsed: dict, stderr: str) -> None:
    fin = parsed["finished"]
    if fin is not None and fin.get("status") == "success":
        return  # never scan a successful reply: abstracts contain words like "rate" and "limit"
    fin_info = {k: v for k, v in (fin or {}).items() if k != "finalText"}
    text = f"{stderr} {json.dumps(fin_info)}".lower()
    if "login" in text or "not authenticated" in text or "unauthorized" in text:
        raise ProviderUnavailable(f"kiro auth: {stderr[:200]}")
    if "credit" in text and ("insufficient" in text or "exhaust" in text or "limit" in text):
        raise QuotaExhausted(f"kiro credits: {text[:200]}")
    if ("rate" in text and "limit" in text) or "throttl" in text:
        raise RetryableError(f"kiro throttled: {text[:200]}")
    if fin is None:
        raise RetryableError(f"kiro exit {proc_returncode}, no runFinished event: {stderr[:200]!r}")
    if fin.get("status") != "success":
        raise FatalError(f"kiro run status {fin.get('status')!r}: {str(fin)[:200]}")


class KiroProvider(Provider):
    family = "kiro"
    supports_temperature = False
    resumes_sessions = True  # a follow-up turn continues the first turn's saved session

    def __init__(self, model, limits, sandbox: Path = KIRO_SANDBOX_DIR, **kw):
        kw.setdefault("timeout", 300.0)
        super().__init__(model, limits, **kw)
        self.sandbox = sandbox
        self.exe = shutil.which("kiro-cli") or "kiro-cli"

    def _call(self, prompt, temperature, history=None, session=None):
        session = session or {}
        if history and not session.get("resume"):
            raise ValueError("a Kiro follow-up turn needs session={'resume': <session_id>}")
        self.sandbox.mkdir(parents=True, exist_ok=True)
        if any(self.sandbox.iterdir()):
            raise RuntimeError(f"Kiro sandbox dir is not empty: {self.sandbox}")
        cmd = build_command(self.exe, self.model, prompt, session.get("resume"))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                  cwd=self.sandbox, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise RetryableError("kiro-cli timed out") from None
        parsed = parse_stream(proc.stdout)
        classify(proc.returncode, parsed, proc.stderr)
        return self.parse(parsed)

    def parse(self, parsed: dict) -> CallResult:
        fin = parsed["finished"] or {}
        return CallResult(
            text=fin.get("finalText"), raw={"events": parsed["events"]},
            model_reported=None,  # the stream names no model; the requested --model is recorded
            meta={"session_id": parsed["session_id"], "credits": round(parsed["credits"], 6),
                  "stop_reason": fin.get("stopReason")},
        )

    def list_models(self) -> list[str]:
        proc = subprocess.run([self.exe, "chat", "--list-models", "--format", "json"],
                              capture_output=True, text=True, encoding="utf-8", timeout=120)
        data = json.loads(proc.stdout)
        models = data.get("models", data) if isinstance(data, dict) else data
        return sorted(m.get("id") or m.get("name") if isinstance(m, dict) else str(m) for m in models)
