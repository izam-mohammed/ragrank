"""Checking the judge against human labels."""

from __future__ import annotations

import pytest

from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.evaluation import align
from ragrank.evaluation.alignment import (
    cohens_kappa,
    pearson,
    rank,
    spearman,
)
from ragrank.llm import FakeLLM
from ragrank.metric import (
    exact_match,
    response_conciseness,
    response_relevancy,
)

# ---------------------------- statistics ----------------------------


def test_pearson_of_a_perfect_line_is_one() -> None:
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_pearson_of_a_reversed_line_is_minus_one() -> None:
    assert pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_is_undefined_for_a_constant_series() -> None:
    assert pearson([1, 1, 1], [1, 2, 3]) is None


def test_pearson_needs_two_points() -> None:
    assert pearson([1], [2]) is None


def test_ranks_average_over_ties() -> None:
    assert rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_survives_a_compressed_scale() -> None:
    """Right ordering, wrong spread: rank correlation still sees it."""
    human = [0.0, 0.25, 0.5, 0.75, 1.0]
    judge = [0.40, 0.42, 0.44, 0.46, 0.48]

    assert spearman(judge, human) == pytest.approx(1.0)
    assert pearson(judge, human) == pytest.approx(1.0)


def test_spearman_is_one_for_any_monotonic_relation() -> None:
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(
        1.0
    )


def test_kappa_of_total_agreement_is_one() -> None:
    assert cohens_kappa(
        [True, False, True, False], [True, False, True, False]
    ) == pytest.approx(1.0)


def test_kappa_discounts_chance_agreement() -> None:
    """Two raters who always say yes agree, but learn nothing."""
    assert cohens_kappa([True] * 5, [True] * 5) is None


def test_kappa_of_pure_disagreement_is_negative() -> None:
    value = cohens_kappa(
        [True, True, False, False], [False, False, True, True]
    )
    assert value is not None
    assert value < 0


def test_kappa_needs_something_to_compare() -> None:
    assert cohens_kappa([], []) is None


# ----------------------------- align -----------------------------


def test_a_judge_that_matches_humans_exactly() -> None:
    labels = [0.0, 0.5, 1.0, 0.5]
    report = align(labels, labels, metric="Judge")

    assert report.pairs == 4
    assert report.pearson == pytest.approx(1.0)
    assert report.spearman == pytest.approx(1.0)
    assert report.mean_absolute_error == pytest.approx(0.0)
    assert report.bias == pytest.approx(0.0)


def test_a_generous_judge_shows_positive_bias() -> None:
    report = align([0.7, 0.8, 0.9], [0.5, 0.6, 0.7])

    assert report.bias == pytest.approx(0.2)
    assert report.mean_absolute_error == pytest.approx(0.2)
    assert report.pearson == pytest.approx(1.0)


def test_a_harsh_judge_shows_negative_bias() -> None:
    assert align([0.3], [0.5, 0.9][:1]).bias == pytest.approx(-0.2)


def test_missing_values_are_dropped_not_zeroed() -> None:
    report = align([1.0, None, 0.0], [1.0, 0.5, 0.0])

    assert report.pairs == 2
    assert report.dropped == 1
    assert report.mean_absolute_error == pytest.approx(0.0)


def test_a_missing_human_label_drops_the_pair_too() -> None:
    report = align([1.0, 0.5], [1.0, None])
    assert report.pairs == 1
    assert report.dropped == 1


def test_nothing_usable_yields_an_empty_report() -> None:
    report = align([None, None], [0.5, 0.5])

    assert report.pairs == 0
    assert report.dropped == 2
    assert report.pearson is None
    assert report.trustworthy is None


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="line up"):
        align([1.0, 0.0], [1.0])


def test_a_threshold_adds_agreement_and_kappa() -> None:
    judge = [0.9, 0.8, 0.2, 0.1]
    human = [1.0, 0.0, 1.0, 0.0]
    report = align(judge, human, threshold=0.5)

    assert report.agreement == pytest.approx(0.5)
    assert report.kappa is not None


def test_without_a_threshold_there_is_no_agreement_number() -> None:
    report = align([0.9, 0.1], [1.0, 0.0])
    assert report.agreement is None
    assert report.kappa is None


def test_trustworthy_needs_both_correlation_and_evidence() -> None:
    aligned = list(range(30))
    strong = align(
        [float(item) for item in aligned],
        [float(item) for item in aligned],
    )
    assert strong.trustworthy is True

    inverted = align(
        [float(item) for item in aligned],
        [float(-item) for item in aligned],
    )
    assert inverted.trustworthy is False


def test_too_few_pairs_refuses_to_conclude() -> None:
    report = align([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert report.spearman == pytest.approx(1.0)
    assert report.trustworthy is None
    assert "too few" in repr(report)


def test_the_report_reads_as_text() -> None:
    text = str(align([1.0, 0.0], [1.0, 0.0], metric="Faithfulness"))
    assert "Faithfulness vs human labels" in text
    assert "pearson" in text


# -------------------------- with a run --------------------------


def run_with(metrics: list) -> object:
    data = from_dict(
        {
            "question": ["a", "b"],
            "context": [["x"], ["y"]],
            "response": ["1", "2"],
        },
        return_as_dataset=True,
    )
    return evaluate(
        data,
        metrics=metrics,
        llm=FakeLLM(responses=["0.8"]),
    )


def test_a_single_metric_run_needs_no_metric_argument() -> None:
    report = align(run_with([response_relevancy]), [0.9, 0.7])

    assert report.metric == "Response Relevancy"
    assert report.pairs == 2


def test_a_multi_metric_run_must_say_which() -> None:
    result = run_with([response_relevancy, response_conciseness])

    with pytest.raises(ValueError, match="pass metric="):
        align(result, [0.9, 0.7])


def test_a_metric_can_be_named_or_passed() -> None:
    result = run_with([response_relevancy, response_conciseness])

    by_name = align(result, [0.9, 0.7], metric="Response Relevancy")
    by_object = align(result, [0.9, 0.7], metric=response_relevancy)

    assert by_name.metric == by_object.metric
    assert by_name.pearson == by_object.pearson


def test_an_unknown_metric_lists_what_the_run_had() -> None:
    result = run_with([response_relevancy])

    with pytest.raises(ValueError, match="Response Relevancy"):
        align(result, [0.9, 0.7], metric=exact_match)
