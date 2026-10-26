import pytest

from aisci.parsing import ParseError, parse_response
from aisci.prompts import build_prompt, prompt_hash


def test_closed_prompt_has_question_and_no_quotes_field():
    p = build_prompt("closed", "Does X cause Y?")
    assert "Question: Does X cause Y?" in p
    assert '"confidence"' in p and '"quotes"' not in p
    assert "Abstract" not in p


def test_context_prompt_includes_abstract_and_quotes_field():
    p = build_prompt("context", "Does X cause Y?", "BACKGROUND: X was studied.")
    assert "BACKGROUND: X was studied." in p
    assert '"quotes"' in p and "word for word" in p


def test_prompt_is_deterministic_and_model_independent():
    a = build_prompt("context", "Q?", "A.")
    b = build_prompt("context", "Q?", "A.")
    assert a == b and prompt_hash(a) == prompt_hash(b)


def test_prompt_errors():
    with pytest.raises(ValueError):
        build_prompt("context", "Q?", None)
    with pytest.raises(ValueError):
        build_prompt("open", "Q?")


def test_parse_strict_json():
    p = parse_response('{"answer": "Yes", "confidence": 85}', require_quotes=False)
    assert (p.answer, p.confidence, p.quotes, p.mode) == ("yes", 85.0, None, "strict")


def test_parse_code_fence():
    p = parse_response('```json\n{"answer": "no", "confidence": 40, "quotes": ["a b"]}\n```',
                       require_quotes=True)
    assert p.answer == "no" and p.quotes == ("a b",) and p.mode == "strict"


def test_parse_embedded_single_object():
    p = parse_response('Here you go: {"answer": "maybe", "confidence": 50} thanks', False)
    assert p.answer == "maybe" and p.mode == "embedded"


@pytest.mark.parametrize("text", [
    None, "", "   ", "yes", "not json at all",
    '{"answer": "probably", "confidence": 50}',
    '{"answer": "yes", "confidence": 101}',
    '{"answer": "yes", "confidence": -1}',
    '{"answer": "yes", "confidence": "high"}',
    '{"answer": "yes", "confidence": true}',
    '{"confidence": 50}',
    '[{"answer": "yes", "confidence": 50}]',
    '{"answer": "yes", "confidence": 50} {"answer": "no", "confidence": 50}',
])
def test_parse_rejects(text):
    with pytest.raises(ParseError):
        parse_response(text, require_quotes=False)


@pytest.mark.parametrize("text", [
    '{"answer": "yes", "confidence": 50}',
    '{"answer": "yes", "confidence": 50, "quotes": "one string"}',
    '{"answer": "yes", "confidence": 50, "quotes": [1, 2]}',
])
def test_parse_requires_quote_list_in_context(text):
    with pytest.raises(ParseError):
        parse_response(text, require_quotes=True)


def test_parse_empty_quote_list_is_valid_answer():
    p = parse_response('{"answer": "yes", "confidence": 70, "quotes": []}', True)
    assert p.quotes == ()
