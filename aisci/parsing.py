"""Deterministic parsing of model output into (answer, confidence, quotes).

Rule: after removing a surrounding markdown code fence, the text must contain
exactly one JSON object that decodes and validates. Anything else is
unparseable and counted as missing, never as an answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

VALID_ANSWERS = ("yes", "no", "maybe")
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Parsed:
    answer: str
    confidence: float
    quotes: tuple[str, ...] | None = None
    mode: str = "strict"  # "strict": the whole text is the JSON object; "embedded": surrounded by other text
    extra: dict = field(default_factory=dict, compare=False)


def _json_objects(text: str) -> list[dict]:
    dec = json.JSONDecoder()
    found, i = [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = end
    return found


def parse_response(text: str | None, require_quotes: bool) -> Parsed:
    if text is None or not text.strip():
        raise ParseError("empty response")
    body = text.strip()
    m = _FENCE.match(body)
    if m:
        body = m.group(1).strip()
    try:
        obj = json.loads(body)
        mode = "strict"
    except json.JSONDecodeError:
        objs = _json_objects(body)
        if len(objs) != 1:
            raise ParseError(f"expected exactly one JSON object, found {len(objs)}")
        obj, mode = objs[0], "embedded"
    if not isinstance(obj, dict):
        raise ParseError("top-level JSON is not an object")

    ans = obj.get("answer")
    if not isinstance(ans, str) or ans.strip().lower() not in VALID_ANSWERS:
        raise ParseError(f"invalid answer: {ans!r}")
    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 100:
        raise ParseError(f"invalid confidence: {conf!r}")

    quotes = None
    if require_quotes:
        q = obj.get("quotes")
        if not isinstance(q, list) or not all(isinstance(s, str) for s in q):
            raise ParseError("quotes missing or not a list of strings")
        quotes = tuple(q)
    extra = {k: v for k, v in obj.items() if k not in ("answer", "confidence", "quotes")}
    return Parsed(ans.strip().lower(), float(conf), quotes, mode, extra)
