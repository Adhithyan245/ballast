"""Prompt text. Identical for every model; sent as a single user message with
no system prompt (except what Claude Code adds itself, a stated limitation)."""
from __future__ import annotations

import hashlib

CLOSED_TEMPLATE = """You are answering a biomedical research question.

Question: {question}

Answer "yes", "no", or "maybe", and give your confidence from 0 to 100 that your answer is correct.

Respond with only a JSON object and no other text, in exactly this format:
{{"answer": "yes" or "no" or "maybe", "confidence": integer from 0 to 100}}"""

CONTEXT_TEMPLATE = """You are answering a biomedical research question using the abstract below.

Abstract:
\"\"\"
{abstract}
\"\"\"

Question: {question}

Answer "yes", "no", or "maybe", and give your confidence from 0 to 100 that your answer is correct. Also copy, word for word, one to three sentences from the abstract that support your answer.

Respond with only a JSON object and no other text, in exactly this format:
{{"answer": "yes" or "no" or "maybe", "confidence": integer from 0 to 100, "quotes": ["exact sentence copied from the abstract", ...]}}"""


def build_prompt(context: str, question: str, abstract: str | None = None) -> str:
    if context == "closed":
        return CLOSED_TEMPLATE.format(question=question)
    if context == "context":
        if not abstract:
            raise ValueError("with-context prompt needs an abstract")
        return CONTEXT_TEMPLATE.format(question=question, abstract=abstract)
    raise ValueError(f"unknown context condition: {context!r}")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
