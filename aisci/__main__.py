"""Command line: python -m aisci <command>

  download              fetch PubMedQA PQA-L at the pinned commit and verify it
  sample                write data/samples/{pilot,main}.json
  list-models ENTRY     print the provider's live model list
  check [ENTRY ...]     readiness: login / key presence and live model lists (no quota used)
  smoke [ENTRY ...]     one test call per provider (saved under results/smoke/)
  run SAMPLE            run the study on a sample (resumable from cache)
  status [SAMPLE]       calls, missing and parse failures per model/condition
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from . import config, data
from .providers import ProviderUnavailable, adapter_class, load_study_models, make_provider, study_keys


def cmd_download(a):
    p = data.download()
    print(f"downloaded {p.relative_to(config.ROOT)} (git blob {config.PUBMEDQA_GIT_BLOB_SHA} verified)")


def cmd_sample(a):
    samples = data.write_samples(data.load())
    for name, ids in samples.items():
        print(f"{name}: {len(ids)} questions")


def cmd_list_models(a):
    p = make_provider(a.family)
    for m in p.list_models():
        print(m)


def cmd_check(a):
    """Readiness check that spends no quota: login/key presence (never values)
    and each provider's live model list, saved under results/provider_checks/."""
    import os
    import subprocess

    import shutil as _sh

    out_dir = config.ROOT / "results" / "provider_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for fam in a.families or study_keys():
        cls = adapter_class(fam)
        if cls.family == "claude":
            exe = _sh.which("claude") or "claude"
            try:
                st = json.loads(subprocess.run([exe, "auth", "status"], capture_output=True,
                                               text=True, timeout=60).stdout)
                ver = subprocess.run([exe, "--version"], capture_output=True, text=True,
                                     timeout=60).stdout.strip()
            except Exception as e:
                print(f"claude   NOT READY  {type(e).__name__}: {e}")
                continue
            ready = st.get("loggedIn")
            print(f"claude   {'READY' if ready else 'NOT READY'}  loggedIn={st.get('loggedIn')} "
                  f"authMethod={st.get('authMethod')} cli={ver}")
            continue
        if cls.key_var and not os.environ.get(cls.key_var, "").strip():
            print(f"{fam:8s} NOT READY  {cls.key_var} is empty in .env")
            continue
        try:
            models = make_provider(fam, {**load_study_models()["models"][fam], "limits": {}}).list_models()
        except Exception as e:
            print(f"{fam:8s} NOT READY  {type(e).__name__}: {str(e)[:200]}")
            continue
        path = out_dir / f"{fam}_models_{day}.json"
        path.write_text(json.dumps({"family": fam, "checked_utc": datetime.now(timezone.utc).isoformat(),
                                    "models": models}, indent=1) + "\n", encoding="utf-8")
        print(f"{fam:8s} READY  key works, {len(models)} models listed -> {path.relative_to(config.ROOT)}")


def cmd_smoke(a):
    from .parsing import ParseError, parse_response
    from .prompts import build_prompt

    items = data.load()
    item = items[data.load_sample("pilot")[0]]
    prompt = build_prompt("context", item.question, item.abstract)
    out_dir = config.ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for fam in a.families or study_keys():
        p = make_provider(fam)
        try:
            res = p.call(prompt)
        except Exception as e:  # report and continue with the next provider
            print(f"{fam:8s} FAILED  {type(e).__name__}: {str(e)[:200]}")
            ok = False
            continue
        try:
            parsed = parse_response(res.text, require_quotes=True)
            parse_note = f"parsed answer={parsed.answer} conf={parsed.confidence:g} quotes={len(parsed.quotes)} mode={parsed.mode}"
        except ParseError as e:
            parse_note = f"UNPARSEABLE ({e})"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (out_dir / f"{fam}_{stamp}.json").write_text(json.dumps({
            "study_model": p.study_id, "model_reported": res.model_reported, "qid": item.qid,
            "text": res.text, "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
            "latency_s": res.latency_s, "meta": res.meta, "raw": res.raw}, indent=1), encoding="utf-8")
        print(f"{fam:8s} OK  requested={p.model} reported={res.model_reported} "
              f"tokens in/out={res.input_tokens}/{res.output_tokens} {res.latency_s:.1f}s  {parse_note}")
    sys.exit(0 if ok else 1)


def cmd_run(a):
    from .runner import conditions_for, run_provider

    items = data.load()
    ids = data.load_sample(a.sample)
    if a.limit:
        ids = ids[: a.limit]
    chosen = [items[q] for q in ids]
    spec = load_study_models()
    for fam in a.families or study_keys():
        if not spec["models"].get(fam, {}).get("enabled", True):
            print(f"{fam}: disabled in study_models.json, skipping")
            continue
        try:
            p = make_provider(fam)
        except ProviderUnavailable as e:
            print(f"{fam}: unavailable ({e})")
            continue
        conds = a.conditions or conditions_for(p, include_t0=not a.no_t0)
        workers = spec["models"][fam].get("workers", 1)
        s = run_provider(p, chosen, conds, a.sample, max_calls=a.max_calls, workers=workers)
        print(f"{s.model}: new={s.done} cached={s.cached} failed={s.failed} "
              f"tokens in/out={s.input_tokens}/{s.output_tokens} {s.seconds/60:.1f} min"
              + (f"  STOPPED: {s.stopped}" if s.stopped else ""))


def cmd_status(a):
    from .results import status_table

    for row in status_table(a.sample):
        print(json.dumps(row))


def main(argv=None):
    load_dotenv(config.ROOT / ".env", override=True)  # project keys win over stale system env vars
    ap = argparse.ArgumentParser(prog="aisci", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    sub.add_parser("sample").set_defaults(fn=cmd_sample)
    p = sub.add_parser("list-models"); p.add_argument("family")
    p.set_defaults(fn=cmd_list_models)
    p = sub.add_parser("check"); p.add_argument("families", nargs="*", help="entries of study_models.json")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("smoke"); p.add_argument("families", nargs="*", help="entries of study_models.json")
    p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("run"); p.add_argument("sample", choices=["pilot", "main"])
    p.add_argument("--families", nargs="*", help="entries of study_models.json")
    p.add_argument("--conditions", nargs="*")
    p.add_argument("--no-t0", action="store_true", help="skip the temperature-0 conditions")
    p.add_argument("--max-calls", type=int, help="stop each provider after this many new calls")
    p.add_argument("--limit", type=int, help="only the first N questions (debugging)")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("status"); p.add_argument("sample", nargs="?")
    p.set_defaults(fn=cmd_status)
    a = ap.parse_args(argv)
    known = study_keys(enabled_only=False)
    bad = [f for f in (getattr(a, "families", None) or []) + [getattr(a, "family", None)]
           if f and f not in known]
    if bad:
        ap.error(f"unknown entries {bad}; choose from {known}")
    a.fn(a)


if __name__ == "__main__":
    main()
