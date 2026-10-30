"""Kiro CLI adapter: command flags, event-stream parsing and error classification."""
import json

import pytest

from aisci.providers import kiro
from aisci.providers.base import FatalError, ProviderUnavailable, QuotaExhausted, RetryableError
from aisci.ratelimit import Limits


def stream(final_text, status="success", credits=0.07, session="s-1"):
    events = [
        {"type": "runStarted", "data": {"engine": "v2"}},
        {"type": "metadata", "data": {"sessionId": session}},
        {"type": "sessionUpdate", "data": {"sessionId": session, "update": {
            "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": final_text}}}},
        {"type": "metadata", "data": {"sessionId": session,
                                      "meteringUsage": [{"value": credits, "unit": "credit"}]}},
        {"type": "runFinished", "data": {"sessionId": session, "status": status,
                                         "stopReason": "end_turn", "finalText": final_text}},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_command_flags_and_resume():
    cmd = kiro.build_command("kiro-cli", "deepseek-3.2", "PROMPT")
    assert cmd[:3] == ["kiro-cli", "chat", "--no-interactive"]
    assert cmd[cmd.index("--model") + 1] == "deepseek-3.2"
    assert "--trust-tools=" in cmd  # no tools trusted
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[-1] == "PROMPT" and "--resume-id" not in cmd
    r = kiro.build_command("kiro-cli", "m", "PUSH", resume="abc")
    assert r[r.index("--resume-id") + 1] == "abc" and r[-1] == "PUSH"


def test_parse_stream_text_session_credits():
    p = kiro.parse_stream(stream('{"answer":"yes","confidence":80}') + "not json\n")
    assert p["finished"]["finalText"] == '{"answer":"yes","confidence":80}'
    assert p["session_id"] == "s-1" and p["credits"] == pytest.approx(0.07)
    res = kiro.KiroProvider("m", Limits()).parse(p)
    assert res.text == '{"answer":"yes","confidence":80}' and res.meta["session_id"] == "s-1"
    assert res.meta["credits"] == pytest.approx(0.07)


def test_success_is_never_misread_as_an_error():
    # medical text full of "rate", "limit", "credit" and "login" words must not trip the checks
    text = '{"answer":"no","confidence":70,"quotes":["Heart rate stayed within the upper limit; login credit"]}'
    kiro.classify(0, kiro.parse_stream(stream(text)), "")


@pytest.mark.parametrize("stdout,stderr,exc", [
    ("", "error: not authenticated, please run kiro-cli login", ProviderUnavailable),
    ("", "Insufficient credits remaining", QuotaExhausted),
    ("", "request throttled", RetryableError),
    ("", "", RetryableError),  # no runFinished at all
])
def test_errors(stdout, stderr, exc):
    with pytest.raises(exc):
        kiro.classify(1, kiro.parse_stream(stdout), stderr)


def test_failed_run_status_is_fatal():
    with pytest.raises(FatalError):
        kiro.classify(0, kiro.parse_stream(stream("", status="error")), "")


def test_followup_requires_resume(tmp_path):
    p = kiro.KiroProvider("m", Limits(), sandbox=tmp_path / "sb", sleep=lambda _: None)
    with pytest.raises(ValueError):
        p.call("PUSH", history=[{"user": "Q", "assistant": "A"}])
    with pytest.raises(ValueError):
        p.call("x", temperature=0.0)  # temperature cannot be set
