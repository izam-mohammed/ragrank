"""Tests for comparing two evaluation runs."""

from __future__ import annotations

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.evaluation import Comparison, RunConfig, compare
from ragrank.llm import FakeLLM
from ragrank.metric import (
    exact_match,
    response_conciseness,
    response_relevancy,
)

SERIAL = RunConfig(show_progress=False, max_workers=1)


@pytest.fixture
def dataset() -> Dataset:
    """Six rows."""
    return Dataset(
        question=[f"q{i}" for i in range(6)],
        context=[["c"]] * 6,
        response=["r"] * 6,
        reference=["r"] * 6,
    )


def run(
    dataset: Dataset,
    scores: list[str],
    metrics: list | None = None,
) -> object:
    """Evaluate with a scripted judge."""
    return evaluate(
        dataset,
        llm=FakeLLM(responses=scores),
        metrics=metrics or [response_relevancy],
        run_config=SERIAL,
    )


STEADY = ["0.5", "0.5", "0.6", "0.5", "0.5", "0.5"]


def test_a_large_improvement_is_significant(
    dataset: Dataset,
) -> None:
    """A real win must be reported as one."""
    diff = compare(run(dataset, STEADY), run(dataset, ["0.9"]))
    assert diff.deltas[0].delta > 0
    assert diff.deltas[0].significant is True
    assert len(diff.improved) == 1


def test_a_tiny_change_is_flagged_as_noise(
    dataset: Dataset,
) -> None:
    """The point of the feature: 0.003 is not an improvement."""
    nudged = ["0.5", "0.5", "0.6", "0.5", "0.5", "0.52"]
    diff = compare(run(dataset, STEADY), run(dataset, nudged))
    assert diff.deltas[0].delta > 0
    assert diff.deltas[0].significant is False
    assert diff.improved == []


def test_a_regression_is_caught(dataset: Dataset) -> None:
    """Getting worse must be visible and attributed to rows."""
    diff = compare(run(dataset, STEADY), run(dataset, ["0.1"]))
    delta = diff.deltas[0]
    assert delta.delta < 0
    assert delta.significant is True
    assert delta.regressed_rows == [0, 1, 2, 3, 4, 5]
    assert len(diff.regressed) == 1


def test_row_level_movement_is_reported(dataset: Dataset) -> None:
    """Which rows moved, not just the aggregate."""
    before = run(dataset, ["0.5", "0.5", "0.5", "0.5", "0.5", "0.5"])
    after = run(dataset, ["0.9", "0.5", "0.1", "0.5", "0.5", "0.5"])
    delta = compare(before, after).deltas[0]
    assert delta.improved_rows == [0]
    assert delta.regressed_rows == [2]


def test_identical_runs_show_no_movement(dataset: Dataset) -> None:
    """Same in, same out."""
    diff = compare(run(dataset, ["0.5"]), run(dataset, ["0.5"]))
    delta = diff.deltas[0]
    assert delta.delta == pytest.approx(0.0)
    assert delta.improved_rows == []
    assert delta.regressed_rows == []
    assert bool(diff) is False


def test_only_shared_metrics_are_compared(dataset: Dataset) -> None:
    """Adding a metric is not a regression in the existing ones."""
    before = run(dataset, ["0.5"], [response_relevancy])
    after = run(
        dataset, ["0.5"], [response_relevancy, response_conciseness]
    )
    diff = compare(before, after)
    assert [item.name for item in diff.deltas] == [
        "Response Relevancy"
    ]


def test_no_shared_metrics_gives_an_empty_comparison(
    dataset: Dataset,
) -> None:
    """Nothing in common is not an error."""
    before = run(dataset, ["0.5"], [response_relevancy])
    after = evaluate(
        dataset,
        llm=FakeLLM(),
        metrics=[exact_match],
        run_config=SERIAL,
    )
    diff = compare(before, after)
    assert diff.deltas == []
    assert bool(diff) is False
    assert "no shared metrics" in repr(diff)


def test_unscored_rows_are_skipped_not_counted_as_movement(
    dataset: Dataset,
) -> None:
    """A row that failed in one run has not improved or regressed."""
    before = run(dataset, ["0.5"])
    after = evaluate(
        dataset,
        llm=FakeLLM(
            responses=["0.5", "banana", "0.5", "0.5", "0.5", "0.5"]
        ),
        metrics=[
            response_relevancy.model_copy(update={"max_retries": 0})
        ],
        run_config=SERIAL,
    )
    delta = compare(before, after).deltas[0]
    assert 1 not in delta.improved_rows
    assert 1 not in delta.regressed_rows


def test_significance_is_none_without_enough_data() -> None:
    """One row has no spread, so no claim can be made."""
    single = Dataset(question=["q"], context=[["c"]], response=["r"])
    diff = compare(run(single, ["0.5"]), run(single, ["0.9"]))
    assert diff.deltas[0].significant is None


def test_repr_reads_like_a_report(dataset: Dataset) -> None:
    """The default rendering should be usable in a terminal."""
    text = repr(compare(run(dataset, STEADY), run(dataset, ["0.1"])))
    assert "Response Relevancy" in text
    assert "->" in text
    assert "significant" in text
    assert "rows regressed" in text


def test_noise_is_labelled_in_the_repr(dataset: Dataset) -> None:
    """A change within noise must not read like a win."""
    nudged = ["0.5", "0.5", "0.6", "0.5", "0.5", "0.52"]
    text = repr(compare(run(dataset, STEADY), run(dataset, nudged)))
    assert "within noise" in text


def test_str_matches_repr(dataset: Dataset) -> None:
    """print() and repr() agree."""
    diff = compare(run(dataset, ["0.5"]), run(dataset, ["0.9"]))
    assert str(diff) == repr(diff)
    assert str(diff.deltas[0]) == repr(diff.deltas[0])


def test_comparison_is_usable_as_a_ci_gate(
    dataset: Dataset,
) -> None:
    """The workflow this exists for."""
    diff = compare(run(dataset, STEADY), run(dataset, ["0.1"]))
    assert diff.regressed, "a real regression must fail the gate"

    clean = compare(run(dataset, ["0.5"]), run(dataset, ["0.5"]))
    assert not clean.regressed


def test_comparison_is_frozen(dataset: Dataset) -> None:
    """A record of a diff is a value."""
    diff = compare(run(dataset, ["0.5"]), run(dataset, ["0.5"]))
    assert isinstance(diff, Comparison)
    with pytest.raises(Exception):  # noqa: B017, PT011
        diff.deltas = []
