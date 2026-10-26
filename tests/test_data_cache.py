import json

import pytest

from aisci import data
from aisci.cache import Cache, Key
from aisci.ratelimit import Limits, QuotaExhausted, RateLimiter


def fake_items(n_yes=55, n_no=34, n_maybe=11):
    items, i = {}, 1000
    for lab, n in (("yes", n_yes), ("no", n_no), ("maybe", n_maybe)):
        for _ in range(n):
            items[str(i)] = data.Item(str(i), f"q{i}?", ("c1", "c2"), ("BACKGROUND", "RESULTS"), lab)
            i += 1
    return items


def test_stratified_counts_match_population():
    assert data.stratified_counts({"yes": 552, "no": 338, "maybe": 110}, 20) == {"yes": 11, "no": 7, "maybe": 2}
    assert data.stratified_counts({"yes": 541, "no": 331, "maybe": 108}, 100) == {"yes": 55, "no": 34, "maybe": 11}


def test_sample_is_deterministic_and_disjoint():
    items = fake_items(552, 338, 110)
    a, b = data.make_samples(items), data.make_samples(items)
    assert a == b
    assert len(a["pilot"]) == 20 and len(a["main"]) == 100
    assert not set(a["pilot"]) & set(a["main"])


def test_abstract_excludes_nothing_but_long_answer():
    it = data.Item("1", "q?", ("First.", "Second."), ("BACKGROUND", "RESULTS"), "yes")
    assert it.abstract == "BACKGROUND: First.\n\nRESULTS: Second."


def test_git_blob_sha():
    assert data.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_cache_roundtrip(tmp_path):
    c = Cache(tmp_path)
    k = Key("gemini:m/1", "closed_default", "123", 2)
    assert not c.has(k)
    c.put(k, {"text": "hi"})
    assert c.has(k) and c.get(k)["text"] == "hi"
    assert c.get(k)["key"] == {"model": "gemini:m/1", "condition": "closed_default", "qid": "123", "run": 2}
    assert [r["text"] for r in c.iter_records()] == ["hi"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_rate_limiter_paces_and_stops(tmp_path):
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    rl = RateLimiter("x", Limits(rpm=6, rpd=10, safety=1.0), tmp_path, clock=lambda: t[0], sleep=sleep)
    for _ in range(10):
        rl.acquire()
    assert slept and all(s == pytest.approx(10.0) for s in slept)  # 60/6 seconds apart
    with pytest.raises(QuotaExhausted):
        rl.acquire()
    day_file = next(tmp_path.glob("x_*.json"))
    assert json.loads(day_file.read_text())["requests"] == 10


def test_rate_limiter_tpm(tmp_path):
    t = [0.0]

    def sleep(s):
        t[0] += s

    rl = RateLimiter("y", Limits(tpm=1000, safety=1.0), tmp_path, clock=lambda: t[0], sleep=sleep)
    rl.acquire(est_tokens=600)
    rl.record_tokens(900)
    rl.acquire(est_tokens=600)  # must wait for the 900 tokens to leave the 60 s window
    assert t[0] > 60
