"""Tests for repeated scoring, which exposes judge variance."""

from __future__ import annotations

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.evaluation import RunConfig
from ragrank.llm import FakeLLM
from ragrank.metric import exact_match, response_relevancy


@pytest.fixture
def one_row() -> Dataset:
    """A single datapoint."""
    return Dataset(
        question=["q"],
        context=[["c"]],
        response=["r"],
        reference=["r"],
    )


def run(
    dataset: Dataset, responses: list[str], **config: object
) -> object:
    """Evaluate with a scripted judge."""
    return evaluate(
        dataset,
        llm=FakeLLM(responses=responses),
        metrics=[response_relevancy],
        run_config=RunConfig(
            show_progress=False, max_workers=1, **config
        ),
    )


def test_defaults_to_a_single_pass(one_row: Dataset) -> None:
    """No surprise extra spend."""
    assert RunConfig().repetitions == 1
    assert run(one_row, ["0.5"]).usage.calls == 1


def test_repetitions_call_the_judge_repeatedly(
    one_row: Dataset,
) -> None:
    """Five repetitions is five calls."""
    result = run(one_row, ["0.5"], repetitions=5)
    assert result.usage.calls == 5


def test_mean_is_the_default_reducer(one_row: Dataset) -> None:
    """The average of the samples."""
    result = run(
        one_row, ["0.9", "0.5", "0.7", "0.6", "0.8"], repetitions=5
    )
    assert result.scores[0][0] == pytest.approx(0.7)


def test_samples_and_spread_are_recorded(one_row: Dataset) -> None:
    """The point: variance is visible, not hidden behind one sample."""
    result = run(
        one_row, ["0.9", "0.5", "0.7", "0.6", "0.8"], repetitions=5
    )
    meta = result.results[0][0].metadata
    assert meta["repetitions"] == [0.9, 0.5, 0.7, 0.6, 0.8]
    assert meta["repetition_spread"] == pytest.approx(
        0.158, abs=1e-3
    )


def test_a_steady_judge_shows_zero_spread(
    one_row: Dataset,
) -> None:
    """No variance is also a finding."""
    result = run(one_row, ["0.5"], repetitions=4)
    assert result.results[0][0].metadata[
        "repetition_spread"
    ] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("reducer", "expected"),
    [("mean", 0.6), ("median", 0.6), ("mode", 0.6)],
)
def test_reducers(
    one_row: Dataset, reducer: str, expected: float
) -> None:
    """Each named reducer works."""
    result = run(
        one_row,
        ["0.6", "0.6", "0.6"],
        repetitions=3,
        reducer=reducer,
    )
    assert result.scores[0][0] == pytest.approx(expected)


def test_median_ignores_an_outlier(one_row: Dataset) -> None:
    """Why you would choose median over mean."""
    result = run(
        one_row,
        ["0.5", "0.5", "1.0"],
        repetitions=3,
        reducer="median",
    )
    assert result.scores[0][0] == pytest.approx(0.5)


def test_partial_failures_still_reduce(one_row: Dataset) -> None:
    """One unusable answer does not lose the other samples."""
    result = evaluate(
        one_row,
        llm=FakeLLM(responses=["0.6", "banana", "0.8"]),
        metrics=[
            response_relevancy.model_copy(update={"max_retries": 0})
        ],
        run_config=RunConfig(
            show_progress=False, max_workers=1, repetitions=3
        ),
    )
    assert result.scores[0][0] == pytest.approx(0.7)
    assert result.results[0][0].metadata["repetitions"] == [0.6, 0.8]


def test_all_repetitions_failing_abstains(one_row: Dataset) -> None:
    """No usable sample means no score."""
    result = evaluate(
        one_row,
        llm=FakeLLM(responses=["banana"]),
        metrics=[
            response_relevancy.model_copy(update={"max_retries": 0})
        ],
        run_config=RunConfig(
            show_progress=False, max_workers=1, repetitions=3
        ),
    )
    assert result.scores[0][0] is None
    assert result.results[0][0].error is not None


def test_deterministic_metrics_are_not_repeated_wastefully(
    one_row: Dataset,
) -> None:
    """Repeating a pure function still costs nothing."""
    result = evaluate(
        one_row,
        llm=FakeLLM(),
        metrics=[exact_match],
        run_config=RunConfig(
            show_progress=False, max_workers=1, repetitions=3
        ),
    )
    assert result.usage.calls == 0
    assert result.scores[0][0] == 1.0


# --------------------- RunConfig strictness ---------------------


def test_a_typo_in_run_config_is_an_error() -> None:
    """Silently ignoring an option the user set is a trap."""
    with pytest.raises(Exception):  # noqa: B017, PT011
        RunConfig(max_worker=8)


def test_an_unknown_reducer_is_caught_at_config_time() -> None:
    """Not part way through a paid run."""
    with pytest.raises(Exception):  # noqa: B017, PT011
        RunConfig(reducer="avarage")


def test_repetitions_must_be_positive() -> None:
    """Zero repetitions is meaningless."""
    with pytest.raises(Exception):  # noqa: B017, PT011
        RunConfig(repetitions=0)
