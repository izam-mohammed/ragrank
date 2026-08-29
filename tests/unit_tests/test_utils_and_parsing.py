"""Tests for the small pieces everything else depends on."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from ragrank.exceptions import (
    EvaluationError,
    RagRankError,
    ValidationError,
)
from ragrank.metric.parse import ParsedScore, parse_score
from ragrank.utils.common import eval_cell
from ragrank.utils.llm import get_env_var
from ragrank.utils.optional import require

# --------------------------- parse_score ---------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.8", 0.8),
        ("  0.8  ", 0.8),
        ("0", 0.0),
        ("1", 1.0),
        ("I would say 0.8 overall", 0.8),
        ("```\n0.5\n```", 0.5),
        ("```json\n0.25\n```", 0.25),
    ],
)
def test_parses_a_number(text: str, expected: float) -> None:
    """Numbers are found even when wrapped in prose or fences."""
    assert parse_score(text).score == pytest.approx(expected)


@pytest.mark.parametrize(
    "text", ["47", "-1", "1.5", "banana", "", "   ", "\n"]
)
def test_rejects_what_is_not_a_valid_score(text: str) -> None:
    """Out of range and unparseable answers abstain with a reason."""
    parsed = parse_score(text)
    assert parsed.score is None
    assert parsed.error


def test_none_input_is_handled() -> None:
    """A model that returned nothing at all must not crash the parser."""
    parsed = parse_score(None)
    assert parsed.score is None
    assert "no content" in parsed.error


def test_custom_score_range_is_respected() -> None:
    """A metric may declare bounds other than 0 to 1."""
    assert parse_score("47", score_range=(0.0, 100.0)).score == 47.0
    assert parse_score("47", score_range=(0.0, 10.0)).score is None


RUBRIC = {"A": 1.0, "B": 0.5, "C": 0.0}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", 1.0),
        ("b", 0.5),
        ("  C  ", 0.0),
        ("Answer: B", 0.5),
        ("(A)", 1.0),
        ("C) the passage is unrelated", 0.0),
        ("'A'", 1.0),
    ],
)
def test_parses_a_rubric_choice(text: str, expected: float) -> None:
    """Labels are matched exactly, then as a standalone token."""
    assert parse_score(text, rubric=RUBRIC).score == expected


@pytest.mark.parametrize("text", ["Z", "0.5", "AB", "banana"])
def test_rejects_an_off_rubric_answer(text: str) -> None:
    """A number is not a valid answer when a rubric is in force."""
    parsed = parse_score(text, rubric=RUBRIC)
    assert parsed.score is None
    assert "expected one of" in parsed.error


def test_rubric_matching_is_not_fooled_by_substrings() -> None:
    """A label inside a word must not count as the label."""
    parsed = parse_score("BANANA", rubric={"A": 1.0, "N": 0.0})
    assert parsed.score is None


def test_word_rubric_labels_work() -> None:
    """Rubrics are not limited to single letters."""
    rubric = {"pass": 1.0, "fail": 0.0}
    assert parse_score("pass", rubric=rubric).score == 1.0
    assert parse_score("FAIL", rubric=rubric).score == 0.0


def test_parsed_score_is_a_named_tuple() -> None:
    """Callers may unpack it."""
    score, error = ParsedScore(0.5, None)
    assert score == 0.5
    assert error is None


# --------------------------- eval_cell ---------------------------


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("['a', 'b']", ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        ("[]", []),
        ("[1, 2]", ["1", "2"]),
        ("plain text", "plain text"),
        ("[unclosed", "[unclosed"),
        ("[1, 2", "[1, 2"),
        ("{'a': 1}", "{'a': 1}"),
        (["already", "list"], ["already", "list"]),
        ("", ""),
    ],
)
def test_eval_cell(
    cell: str | list[str], expected: str | list[str]
) -> None:
    """List literals are parsed; everything else passes through."""
    assert eval_cell(cell) == expected


def test_eval_cell_rejects_a_non_list_literal() -> None:
    """A dict literal is not a context list."""
    assert eval_cell("[1, 2][0]") == "[1, 2][0]"


# --------------------------- env vars ---------------------------


def test_get_env_var_returns_the_value() -> None:
    """A set variable comes back."""
    with patch.dict(os.environ, {"RAGRANK_TEST_VAR": "value"}):
        assert get_env_var("RAGRANK_TEST_VAR") == "value"


def test_get_env_var_error_is_actionable() -> None:
    """The message must say what to do, not just what went wrong."""
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="is not set") as caught,
    ):
        get_env_var("RAGRANK_MISSING_VAR")
    assert "evaluate()" in str(caught.value)


# --------------------- optional dependencies ---------------------


def test_require_returns_an_installed_module() -> None:
    """The happy path returns the module itself."""
    assert require("json", "core").dumps({}) == "{}"


def test_require_explains_how_to_install() -> None:
    """A missing extra must name the extra."""
    with pytest.raises(ModuleNotFoundError) as caught:
        require("definitely_not_installed_xyz", "someextra")
    message = str(caught.value)
    assert "someextra" in message
    assert "pip install" in message


# --------------------------- exceptions ---------------------------


def test_exception_hierarchy() -> None:
    """Callers can catch everything with one base class."""
    assert issubclass(EvaluationError, RagRankError)
    assert issubclass(ValidationError, RagRankError)


def test_exceptions_carry_a_default_message() -> None:
    """Raising bare still produces something readable."""
    assert str(EvaluationError())
    assert str(ValidationError())
    assert str(EvaluationError("custom")) == "custom"
