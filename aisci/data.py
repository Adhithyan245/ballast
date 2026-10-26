"""Download, verify, load and sample the PubMedQA expert-labeled subset.

Labels are read as-is and never modified.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import config


@dataclass(frozen=True)
class Item:
    qid: str
    question: str
    contexts: tuple[str, ...]
    context_labels: tuple[str, ...]
    label: str  # expert final_decision: yes / no / maybe

    @property
    def abstract(self) -> str:
        """Abstract text shown to the model. LONG_ANSWER (the conclusion) is
        deliberately excluded because it usually states the answer."""
        parts = []
        for lab, text in zip(self.context_labels, self.contexts):
            parts.append(f"{lab}: {text}" if lab else text)
        return "\n\n".join(parts)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(dest: Path = config.PUBMEDQA_FILE) -> Path:
    """Fetch ori_pqal.json at the pinned commit and verify its git blob hash."""
    resp = httpx.get(config.PUBMEDQA_URL, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    data = resp.content
    got = git_blob_sha(data)
    if got != config.PUBMEDQA_GIT_BLOB_SHA:
        raise RuntimeError(f"PubMedQA file hash mismatch: {got}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def load(path: Path = config.PUBMEDQA_FILE) -> dict[str, Item]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = {}
    for qid, rec in raw.items():
        items[qid] = Item(
            qid=qid,
            question=rec["QUESTION"],
            contexts=tuple(rec["CONTEXTS"]),
            context_labels=tuple(rec.get("LABELS") or [""] * len(rec["CONTEXTS"])),
            label=rec["final_decision"],
        )
    return items


def stratified_counts(label_counts: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation by largest remainder (ties broken by label order)."""
    total = sum(label_counts.values())
    exact = {lab: n * c / total for lab, c in label_counts.items()}
    counts = {lab: int(v) for lab, v in exact.items()}
    order = sorted(exact, key=lambda lab: (-(exact[lab] - counts[lab]), config.LABELS.index(lab)))
    for lab in order[: n - sum(counts.values())]:
        counts[lab] += 1
    return counts


def stratified_sample(
    items: dict[str, Item], n: int, seed: int, exclude: set[str] = frozenset()
) -> list[str]:
    """Deterministic label-stratified sample of question ids."""
    pool = {lab: sorted(q for q, it in items.items() if it.label == lab and q not in exclude)
            for lab in config.LABELS}
    counts = stratified_counts({lab: len(v) for lab, v in pool.items()}, n)
    rng = random.Random(seed)
    chosen = []
    for lab in config.LABELS:
        chosen.extend(rng.sample(pool[lab], counts[lab]))
    return sorted(chosen, key=int)


def make_samples(items: dict[str, Item]) -> dict[str, list[str]]:
    pilot = stratified_sample(items, config.PILOT_SIZE, config.SEED)
    main = stratified_sample(items, config.MAIN_SIZE, config.SEED + 1, exclude=set(pilot))
    assert not set(pilot) & set(main)
    return {"pilot": pilot, "main": main}


def write_samples(items: dict[str, Item], out_dir: Path = config.SAMPLES_DIR) -> dict:
    samples = make_samples(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in samples.items():
        meta = {
            "name": name,
            "seed": config.SEED if name == "pilot" else config.SEED + 1,
            "source_commit": config.PUBMEDQA_COMMIT,
            "label_counts": {lab: sum(items[q].label == lab for q in ids) for lab in config.LABELS},
            "ids": ids,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return samples


def load_sample(name: str) -> list[str]:
    return json.loads((config.SAMPLES_DIR / f"{name}.json").read_text(encoding="utf-8"))["ids"]
