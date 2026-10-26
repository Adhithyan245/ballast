"""On-disk cache of raw responses: one JSON file per call, keyed by
(model, condition, question, run). Only calls that returned a response are
cached; transport failures go to the error log and are retried on resume."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


@dataclass(frozen=True)
class Key:
    model: str       # study model id, e.g. "gemini:gemini-2.5-flash"
    condition: str   # e.g. "closed_default"
    qid: str
    run: int

    def path(self, root: Path) -> Path:
        return root / slug(self.model) / self.condition / self.qid / f"run{self.run}.json"


class Cache:
    def __init__(self, root: Path | None = None):
        self.root = root or config.CACHE_DIR

    def has(self, key: Key) -> bool:
        return key.path(self.root).exists()

    def get(self, key: Key) -> dict | None:
        p = key.path(self.root)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, key: Key, record: dict) -> Path:
        p = key.path(self.root)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"key": key.__dict__, "cached_at": datetime.now(timezone.utc).isoformat(), **record}
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)  # atomic: a crash never leaves a half-written record
        return p

    def iter_records(self, model: str | None = None):
        base = self.root / slug(model) if model else self.root
        for p in sorted(base.rglob("run*.json")):
            yield json.loads(p.read_text(encoding="utf-8"))


def log_error(entry: dict, log_dir: Path | None = None) -> None:
    log_dir = log_dir or config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {"at": datetime.now(timezone.utc).isoformat(), **entry}
    with open(log_dir / "errors.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
