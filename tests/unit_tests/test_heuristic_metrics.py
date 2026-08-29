"""Tests for the deterministic text metrics -- no LLM involved."""

from __future__ import annotations

import pytest
from ragrank.dataset import DataNode
from ragrank.metric import (
    exact_match,
    json_valid,
    levenshtein_ratio,
    rouge_l,
    string_presence,
    token_f1,
)
from ragrank.metric._heuristic.text import (
    lcs_length,
    levenshtein,
    normalize,
    tokenize,
)

ALL_REFERENCE_METRICS = [
    exact_match,
    string_presence,
    levenshtein_ratio,
    token_f1,
    rouge_l,
]


def node(response: str, reference: str | None = None) -> DataNode:
    """Build a one-off data node."""
    return DataNode(
        question="q",
        context=["c"],
        response=response,
        reference=reference,
    )


# --------------------------- helpers ---------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Paris.", "paris"),
        ("  a  CAT  ", "cat"),
        ("an apple, the pear!", "apple pear"),
        ("", ""),
        ("...", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    """Normalisation strips case, punctuation and articles."""
    assert normalize(raw) == expected


def test_tokenize_of_empty_string_is_empty() -> None:
    """An empty string has no tokens, not one empty token."""
    assert tokenize("") == []
    assert tokenize("  ") == []


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("kitten", "sitting", 3),
        ("", "abc", 3),
        ("abc", "", 3),
        ("same", "same", 0),
        ("ab", "ba", 2),
    ],
)
def test_levenshtein(left: str, right: str, expected: int) -> None:
    """Edit distance matches known values."""
    assert levenshtein(left, right) == expected


def test_levenshtein_is_symmetric() -> None:
    """Distance does not depend on argument order."""
    assert levenshtein("abcdef", "azced") == levenshtein(
        "azced", "abcdef"
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (["a", "b", "c"], ["a", "c"], 2),
        (["a", "b"], ["c", "d"], 0),
        ([], ["a"], 0),
        (["a", "b", "c"], ["a", "b", "c"], 3),
    ],
)
def test_lcs_length(
    left: list[str], right: list[str], expected: int
) -> None:
    """Longest common subsequence matches known values."""
    assert lcs_length(left, right) == expected


# --------------------------- metrics ---------------------------


def test_exact_match_normalises_before_comparing() -> None:
    """Casing, punctuation and articles must not matter."""
    assert (
        exact_match.score(node("The Paris.", "paris")).score == 1.0
    )
    assert exact_match.score(node("London", "paris")).score == 0.0


def test_string_presence_finds_a_substring() -> None:
    """The reference need only appear somewhere."""
    assert (
        string_presence.score(
            node("I think Paris is", "Paris")
        ).score
        == 1.0
    )
    assert (
        string_presence.score(node("I think Rome is", "Paris")).score
        == 0.0
    )


def test_levenshtein_ratio_is_bounded_and_ordered() -> None:
    """Closer strings score higher, and everything stays in [0, 1]."""
    near = levenshtein_ratio.score(node("Paris", "Pariss")).score
    far = levenshtein_ratio.score(node("Paris", "Tokyo")).score
    assert near > far
    assert 0.0 <= far <= near <= 1.0


def test_levenshtein_ratio_of_identical_text_is_one() -> None:
    """Identical strings are maximally similar."""
    assert levenshtein_ratio.score(node("same", "same")).score == 1.0


def test_token_f1_ignores_word_order() -> None:
    """Reordering the same words is still a perfect match."""
    score = token_f1.score(
        node("the capital is Paris", "Paris is the capital")
    ).score
    assert score == pytest.approx(1.0)


def test_token_f1_penalises_missing_and_extra_tokens() -> None:
    """Partial overlap lands strictly between 0 and 1."""
    score = token_f1.score(node("Paris France", "Paris")).score
    assert 0.0 < score < 1.0


def test_token_f1_with_no_overlap_is_zero() -> None:
    """Disjoint token sets score zero."""
    assert token_f1.score(node("alpha beta", "gamma")).score == 0.0


def test_rouge_l_rewards_ordered_overlap() -> None:
    """ROUGE-L is sensitive to subsequence order."""
    ordered = rouge_l.score(node("a b c d", "a b c")).score
    shuffled = rouge_l.score(node("d c b a", "a b c")).score
    assert ordered > shuffled


def test_json_valid_detects_parseable_output() -> None:
    """Valid JSON scores 1, anything else scores 0."""
    assert json_valid.score(node('{"a": 1}')).score == 1.0
    assert json_valid.score(node("[1, 2]")).score == 1.0
    assert json_valid.score(node("not json")).score == 0.0


def test_json_valid_needs_no_reference() -> None:
    """It only looks at the response."""
    assert json_valid.required_columns == {"response"}


@pytest.mark.parametrize("metric", ALL_REFERENCE_METRICS)
def test_reference_metrics_abstain_without_a_reference(
    metric: object,
) -> None:
    """No ground truth means None, never a fabricated zero."""
    result = metric.score(node("anything"))
    assert result.score is None
    assert result.error is not None


@pytest.mark.parametrize("metric", ALL_REFERENCE_METRICS)
def test_reference_metrics_declare_their_columns(
    metric: object,
) -> None:
    """The runner needs this to validate before spending anything."""
    assert metric.required_columns == {"response", "reference"}


@pytest.mark.parametrize("metric", ALL_REFERENCE_METRICS)
def test_reference_metrics_stay_in_range(metric: object) -> None:
    """Every heuristic metric is bounded to [0, 1]."""
    score = metric.score(
        node("some words here", "other words")
    ).score
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("metric", ALL_REFERENCE_METRICS)
def test_reference_metrics_need_no_llm(metric: object) -> None:
    """These must work with no model and no credentials."""
    assert metric.llm is None
    assert metric.prompt is None
    assert metric.score(node("x", "x")).score is not None
