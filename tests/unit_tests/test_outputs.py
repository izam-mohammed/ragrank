"""Tests for EvalResult reporting, and the field-name constants."""

from __future__ import annotations

import json

import pytest
from ragrank import constants
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation import EvalResult, MetricSummary
from ragrank.llm import FakeLLM
from ragrank.metric import exact_match, response_relevancy


@pytest.fixture
def dataset() -> Dataset:
    """Two trivial rows."""
    return Dataset(
        question=["q1", "q2"],
        context=[["c"], ["c"]],
        response=["r1", "r2"],
    )


def result_with(
    dataset: Dataset, scores: list[list[float | None]]
) -> EvalResult:
    """Build an EvalResult directly from scores."""
    return EvalResult(
        llm=FakeLLM(),
        metrics=[response_relevancy],
        dataset=dataset,
        scores=scores,
        response_time=0.1,
    )


# --------------------------- validation ---------------------------


def test_rejects_mismatched_metric_and_score_counts(
    dataset: Dataset,
) -> None:
    """Every metric needs its own score list."""
    with pytest.raises(
        ValueError, match="metrics and number of scores"
    ):
        EvalResult(
            llm=FakeLLM(),
            metrics=[response_relevancy, exact_match],
            dataset=dataset,
            scores=[[1.0, 1.0]],
            response_time=0.1,
        )


def test_rejects_scores_that_do_not_match_the_dataset(
    dataset: Dataset,
) -> None:
    """A score list must be as long as the dataset."""
    with pytest.raises(ValueError, match="not balanced"):
        result_with(dataset, [[1.0]])


def test_rejects_a_non_positive_response_time(
    dataset: Dataset,
) -> None:
    """A run always takes some time."""
    with pytest.raises(ValueError):
        EvalResult(
            llm=FakeLLM(),
            metrics=[response_relevancy],
            dataset=dataset,
            scores=[[1.0, 1.0]],
            response_time=0.0,
        )


# --------------------------- summary ---------------------------


def test_summary_statistics(dataset: Dataset) -> None:
    """Mean, spread and extremes."""
    summary = result_with(dataset, [[0.2, 0.8]]).summary()[0]
    assert isinstance(summary, MetricSummary)
    assert summary.count == 2
    assert summary.scored == 2
    assert summary.failed == 0
    assert summary.value == pytest.approx(0.5)
    assert summary.minimum == pytest.approx(0.2)
    assert summary.maximum == pytest.approx(0.8)
    assert summary.stderr == pytest.approx(0.3)


def test_summary_counts_and_excludes_failures(
    dataset: Dataset,
) -> None:
    """Unscored rows are counted, not averaged in as zero."""
    summary = result_with(dataset, [[0.4, None]]).summary()[0]
    assert summary.scored == 1
    assert summary.failed == 1
    assert summary.value == pytest.approx(0.4)


def test_stderr_needs_more_than_one_point(dataset: Dataset) -> None:
    """A single observation has no spread."""
    summary = result_with(dataset, [[0.5, None]]).summary()[0]
    assert summary.stderr is None


def test_summary_of_an_all_failed_metric(dataset: Dataset) -> None:
    """Everything None means no aggregate at all."""
    summary = result_with(dataset, [[None, None]]).summary()[0]
    assert summary.value is None
    assert summary.minimum is None
    assert summary.passed is None


def test_failed_count_totals_across_metrics(
    dataset: Dataset,
) -> None:
    """Counts (row, metric) pairs, not rows."""
    result = EvalResult(
        llm=FakeLLM(),
        metrics=[response_relevancy, exact_match],
        dataset=dataset,
        scores=[[1.0, None], [None, None]],
        response_time=0.1,
    )
    assert result.failed_count == 3


# --------------------------- thresholds ---------------------------


def test_pass_rate_and_verdict(dataset: Dataset) -> None:
    """Threshold turns scores into a decision."""
    metric = response_relevancy.model_copy(update={"threshold": 0.5})
    result = EvalResult(
        llm=FakeLLM(),
        metrics=[metric],
        dataset=dataset,
        scores=[[0.4, 0.6]],
        response_time=0.1,
    )
    summary = result.summary()[0]
    assert summary.pass_rate == pytest.approx(0.5)
    assert summary.passed is True  # mean 0.5 >= 0.5
    assert result.passed is True


def test_passed_is_none_when_no_metric_gates(
    dataset: Dataset,
) -> None:
    """Without a threshold there is nothing to gate on."""
    assert result_with(dataset, [[0.1, 0.2]]).passed is None


def test_passed_requires_every_gating_metric(
    dataset: Dataset,
) -> None:
    """One failing metric fails the run."""
    good = response_relevancy.model_copy(update={"threshold": 0.1})
    bad = exact_match.model_copy(update={"threshold": 0.9})
    result = EvalResult(
        llm=FakeLLM(),
        metrics=[good, bad],
        dataset=dataset,
        scores=[[0.5, 0.5], [0.0, 0.0]],
        response_time=0.1,
    )
    assert result.passed is False


# --------------------------- output ---------------------------


def test_to_dict_merges_scores_into_the_data(
    dataset: Dataset,
) -> None:
    """The dict carries both the data and the scores."""
    payload = result_with(dataset, [[0.1, 0.2]]).to_dict()
    assert payload["question"] == ["q1", "q2"]
    assert payload["Response Relevancy"] == [0.1, 0.2]


def test_to_json_is_parseable_and_complete(
    dataset: Dataset,
) -> None:
    """JSON output must not need pandas and must round-trip."""
    payload = json.loads(
        result_with(dataset, [[0.1, 0.2]]).to_json()
    )
    assert payload["llm"] == "Fake LLM"
    assert payload["summary"][0]["name"] == "Response Relevancy"
    assert payload["data"]["Response Relevancy"] == [0.1, 0.2]


def test_to_json_accepts_json_dumps_kwargs(
    dataset: Dataset,
) -> None:
    """Indentation and friends pass through."""
    text = result_with(dataset, [[0.1, 0.2]]).to_json(indent=2)
    assert "\n" in text


def test_repr_shows_value_and_error_bar(dataset: Dataset) -> None:
    """The default rendering is a readable summary."""
    text = repr(result_with(dataset, [[0.2, 0.8]]))
    assert "Response Relevancy" in text
    assert "0.500" in text
    assert "+/-" in text


def test_repr_marks_unscored_rows(dataset: Dataset) -> None:
    """Failures must be visible in the headline output."""
    assert "1 unscored" in repr(result_with(dataset, [[0.5, None]]))


def test_repr_handles_a_metric_with_no_scores(
    dataset: Dataset,
) -> None:
    """An all-failed metric renders as n/a, not a crash."""
    assert "n/a" in repr(result_with(dataset, [[None, None]]))


def test_str_matches_repr(dataset: Dataset) -> None:
    """print() and repr() agree."""
    result = result_with(dataset, [[0.2, 0.8]])
    assert str(result) == repr(result)


def test_to_dataframe(dataset: Dataset) -> None:
    """Scores land as a column alongside the data."""
    frame = result_with(dataset, [[0.1, 0.2]]).to_dataframe()
    assert list(frame["Response Relevancy"]) == [0.1, 0.2]
    assert len(frame) == 2


# --------------------------- constants ---------------------------


def test_constants_cannot_drift_from_the_model() -> None:
    """The field lists are derived, not duplicated by hand."""
    assert list(DataNode.model_fields) == constants.DATA_FIELDS
    assert set(
        constants.REQUIRED_FIELDS + constants.OPTIONAL_FIELDS
    ) == set(DataNode.model_fields)


def test_constants_name_the_right_fields() -> None:
    """The individual names are still importable and correct."""
    for name in (
        constants.QUESTION_FIELD,
        constants.CONTEXT_FIELD,
        constants.RESPONSE_FIELD,
        constants.REFERENCE_FIELD,
        constants.RETRIEVED_IDS_FIELD,
        constants.REFERENCE_IDS_FIELD,
    ):
        assert name in DataNode.model_fields
