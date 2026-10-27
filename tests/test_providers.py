"""Adapter tests with mocked transports. No live API calls."""
import json

import httpx
import pytest

from aisci.cache import Cache
from aisci.data import Item
from aisci.providers import claude_code
from aisci.providers.base import CallResult, FatalError, Provider, ProviderUnavailable, QuotaExhausted, RetryableError
from aisci.providers.gemini import GeminiProvider
from aisci.providers.openai_compat import GroqProvider, MistralProvider
from aisci.ratelimit import Limits
from aisci.runner import conditions_for, run_provider


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("aisci.config.USAGE_DIR", tmp_path / "usage")
    monkeypatch.setattr("aisci.config.LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def nosleep(_):
    pass


def test_gemini_parses_text_skips_thoughts_and_sends_temperature(tmp_path):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["key"] = req.headers["x-goog-api-key"]
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                  {"text": '{"answer":"yes","confidence":90}'}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 50},
            "modelVersion": "gemini-x-001"})

    p = GeminiProvider("gemini-x", Limits(), sleep=nosleep)
    p.client = mock_client(handler)
    r = p.call("hello", temperature=0.0)
    assert r.text == '{"answer":"yes","confidence":90}'
    assert (r.input_tokens, r.output_tokens, r.model_reported) == (100, 60, "gemini-x-001")
    assert seen["body"]["generationConfig"] == {"temperature": 0.0}
    assert seen["key"] == "test-key"
    p.call("hello")
    assert "generationConfig" not in seen["body"]  # default sampling sends no temperature


def test_gemini_daily_quota_stops():
    def handler(req):
        return httpx.Response(429, json={"error": {"code": 429, "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})

    p = GeminiProvider("g", Limits(), sleep=nosleep)
    p.client = mock_client(handler)
    with pytest.raises(QuotaExhausted):
        p.call("x")


def test_groq_retries_per_minute_429_then_succeeds():
    calls = {"n": 0}
    waits = []

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "7"},
                                  json={"error": {"message": "Rate limit reached ... requests per minute"}})
        return httpx.Response(200, headers={"x-ratelimit-remaining-requests": "999"}, json={
            "model": "llama-x", "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    p = GroqProvider("llama-x", Limits(), sleep=waits.append)
    p.client = mock_client(handler)
    r = p.call("x")
    assert calls["n"] == 3 and r.text == "{}" and r.model_reported == "llama-x"
    assert all(w >= 7 for w in waits)
    assert r.meta["ratelimit_headers"]["x-ratelimit-remaining-requests"] == "999"


def test_groq_per_day_429_stops():
    p = GroqProvider("m", Limits(), sleep=nosleep)
    p.client = mock_client(lambda r: httpx.Response(429, json={"error": {"message": "Rate limit reached on requests per day (RPD)"}}))
    with pytest.raises(QuotaExhausted):
        p.call("x")


def test_retry_exhaustion_raises():
    p = MistralProvider("m", Limits(), max_attempts=2, sleep=nosleep)
    p.client = mock_client(lambda r: httpx.Response(503, text="busy"))
    with pytest.raises(RetryableError):
        p.call("x")


def test_status_codes_classified():
    p = MistralProvider("m", Limits(), sleep=nosleep)
    p.client = mock_client(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(ProviderUnavailable):
        p.call("x")
    p.client = mock_client(lambda r: httpx.Response(400, text="bad request"))
    with pytest.raises(FatalError):
        p.call("x")


def test_mistral_chunked_content():
    raw = {"model": "mistral-x", "choices": [{"message": {"content": [
        {"type": "thinking", "thinking": []}, {"type": "text", "text": '{"a":1}'}]}}]}
    assert MistralProvider.parse(raw).text == '{"a":1}'


def test_missing_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    p = GroqProvider("m", Limits(), sleep=nosleep)
    with pytest.raises(ProviderUnavailable):
        p.call("x")


def test_claude_command_flags():
    cmd = claude_code.build_command("claude", "sonnet", "safe")
    assert cmd[:3] == ["claude", "-p", "--safe-mode"]
    assert cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert claude_code.build_command("claude", "m", "bare")[2] == "--bare"


@pytest.mark.parametrize("result,exc", [
    ("Not logged in · Please run /login", ProviderUnavailable),
    ("Failed to authenticate: OAuth session expired", ProviderUnavailable),
    ("Claude usage limit reached. Your limit will reset at 5pm", QuotaExhausted),
    ("Something odd", FatalError),
])
def test_claude_error_classification(result, exc):
    with pytest.raises(exc):
        claude_code.classify({"is_error": True, "result": result})
    claude_code.classify({"is_error": False, "result": result})  # not an error -> no raise


def test_claude_parse():
    raw = {"is_error": False, "result": '{"answer":"no"}', "num_turns": 1, "total_cost_usd": 0.01,
           "usage": {"input_tokens": 10, "cache_read_input_tokens": 5000, "output_tokens": 20},
           "modelUsage": {"claude-x-1": {}}}
    r = claude_code.ClaudeCodeProvider.parse(raw)
    assert (r.text, r.model_reported, r.input_tokens, r.output_tokens) == ('{"answer":"no"}', "claude-x-1", 5010, 20)


def test_claude_rejects_temperature(tmp_path):
    p = claude_code.ClaudeCodeProvider("sonnet", Limits(), sandbox=tmp_path / "sb", sleep=nosleep)
    with pytest.raises(ValueError):
        p.call("x", temperature=0.0)
    assert conditions_for(p, include_t0=True) == ["closed_default", "context_default"]


# --- runner ---------------------------------------------------------------
class FakeProvider(Provider):
    family = "fake"

    def __init__(self, script):
        super().__init__("m", Limits(), sleep=nosleep)
        self.script = list(script)
        self.n = 0

    def _call(self, prompt, temperature):
        self.n += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return CallResult(text=step, raw={"t": step}, model_reported="m-1", input_tokens=1, output_tokens=1)


ITEMS = [Item("1", "q1?", ("c",), ("",), "yes"), Item("2", "q2?", ("c",), ("",), "no")]


def test_runner_caches_skips_and_logs_failures(tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache")
    ok = '{"answer":"yes","confidence":90}'
    p = FakeProvider([ok, FatalError("400 bad"), ok, ok])
    s = run_provider(p, ITEMS, ["closed_default"], "pilot", runs=2, cache=cache, log=lambda *_: None)
    assert (s.done, s.failed, s.cached) == (3, 1, 0)
    assert len(list(cache.iter_records())) == 3  # the failed call is not cached as an answer
    assert "400 bad" in (tmp_path / "logs" / "errors.jsonl").read_text()
    # resume: only the failed call is re-requested
    p2 = FakeProvider([ok])
    s2 = run_provider(p2, ITEMS, ["closed_default"], "pilot", runs=2, cache=cache, log=lambda *_: None)
    assert (s2.done, s2.cached, p2.n) == (1, 3, 1)


def test_runner_stops_cleanly_on_quota(tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache")
    p = FakeProvider(['{"answer":"yes","confidence":1}', QuotaExhausted("daily")])
    s = run_provider(p, ITEMS, ["closed_default"], "pilot", runs=2, cache=cache, log=lambda *_: None)
    assert s.done == 1 and s.stopped.startswith("QuotaExhausted")


def test_claude_system_prompt_override():
    assert "--system-prompt" not in claude_code.build_command("claude", "m", "safe")
    cmd = claude_code.build_command("claude", "m", "safe", system_prompt="")
    assert cmd[cmd.index("--system-prompt") + 1] == ""


def test_vllm_provider_url_key_and_reasoning(monkeypatch):
    from aisci.providers.openai_compat import VLLMProvider

    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:9999/v1/")
    monkeypatch.setenv("VLLM_API_KEY", "local-secret")
    seen = {}

    def handler(req):
        seen["url"], seen["auth"] = str(req.url), req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"model": "Qwen/Q", "choices": [{"message": {
            "content": '{"answer":"no","confidence":60}', "reasoning_content": "hmm"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3}})

    p = VLLMProvider("Qwen/Q", Limits(), sleep=nosleep)
    p.client = mock_client(handler)
    r = p.call("x", temperature=0.0)
    assert seen["url"] == "http://localhost:9999/v1/chat/completions"
    assert seen["auth"] == "Bearer local-secret" and seen["body"]["temperature"] == 0.0
    assert r.text == '{"answer":"no","confidence":60}' and r.meta["has_reasoning"]
    # a 429 from our own server is never a daily quota
    p.client = mock_client(lambda req: httpx.Response(429, text="too many requests per day"))
    p.max_attempts = 1
    with pytest.raises(RetryableError):
        p.call("x")


def test_runner_parallel_matches_sequential(tmp_path):
    ok = '{"answer":"yes","confidence":90}'
    cache = Cache(tmp_path / "cache")
    p = FakeProvider([ok] * 8)
    s = run_provider(p, ITEMS, ["closed_default", "context_default"], "pilot", runs=2,
                     cache=cache, workers=4, log=lambda *_: None)
    assert (s.done, s.failed, p.n) == (8, 0, 8)
    keys = sorted((r["key"]["condition"], r["key"]["qid"], r["key"]["run"]) for r in cache.iter_records())
    assert len(set(keys)) == 8


def test_runner_max_calls(tmp_path):
    p = FakeProvider(['{"answer":"yes","confidence":90}'] * 3)
    s = run_provider(p, ITEMS, ["closed_default"], "pilot", runs=2, cache=Cache(tmp_path / "c"),
                     max_calls=3, log=lambda *_: None)
    assert s.done == 3 and s.stopped == "max_calls=3 reached"


def test_study_entries_share_an_adapter(tmp_path):
    from aisci.providers import adapter_class, make_provider, study_keys
    from aisci.providers.openai_compat import VLLMProvider

    f = tmp_path / "sm.json"
    f.write_text(json.dumps({"models": {
        "qwen_vllm": {"adapter": "vllm", "model": "Qwen/A", "limits": {}},
        "mistral_vllm": {"adapter": "vllm", "model": "mistralai/B", "limits": {}},
        "mistral": {"enabled": False, "model": "m", "limits": {}}}}))
    assert study_keys(path=f) == ["qwen_vllm", "mistral_vllm"]
    assert "mistral" in study_keys(enabled_only=False, path=f)
    spec = json.loads(f.read_text())["models"]
    a, b = make_provider("qwen_vllm", spec["qwen_vllm"]), make_provider("mistral_vllm", spec["mistral_vllm"])
    assert isinstance(a, VLLMProvider) and isinstance(b, VLLMProvider)
    assert a.study_id != b.study_id  # separate cache directories
    assert adapter_class("mistral", spec["mistral"]) is MistralProvider


def test_nvidia_provider_and_402_stops():
    from aisci.providers.openai_compat import NvidiaProvider

    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"model": "mistralai/m", "choices": [{"message": {"content": "{}"}}]})

    import os
    os.environ["NVIDIA_API_KEY"] = "test-key"
    p = NvidiaProvider("mistralai/m", Limits(), sleep=nosleep)
    p.client = mock_client(handler)
    assert p.call("x").model_reported == "mistralai/m"
    assert seen["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    p.client = mock_client(lambda r: httpx.Response(402, text="credits exhausted"))
    with pytest.raises(QuotaExhausted):
        p.call("x")
    del os.environ["NVIDIA_API_KEY"]


def test_daily_budget_is_per_entry(tmp_path):
    from aisci.providers import make_provider

    a = make_provider("groq", {"model": "openai/gpt-oss-120b", "limits": {}})
    b = make_provider("qwen_groq", {"adapter": "groq", "model": "qwen/q", "limits": {}})
    assert (a.limiter.name, b.limiter.name) == ("groq", "qwen_groq")


def test_claude_400_is_not_retried():
    with pytest.raises(FatalError):
        claude_code.classify({"is_error": True, "terminal_reason": "api_error", "api_error_status": 400,
                              "result": "API Error: 400 system: text content blocks must contain non-whitespace text"})
