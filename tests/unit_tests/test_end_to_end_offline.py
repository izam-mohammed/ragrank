"""End to end evaluation with no language model at all.

This is the property that makes the library testable, demonstrable and
cheap: a complete run using only deterministic metrics.
"""

from __future__ import annotations

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.evaluation import RunConfig
from ragrank.exceptions import ValidationError
from ragrank.llm import FakeLLM
from ragrank.metric import (
    RETRIEVAL_METRICS,
    exact_match,
    hit_rate,
    metric,
    recall_at_k,
    response_relevancy,
    token_f1,
)

SERIAL = RunConfig(show_progress=False, max_workers=1)


@pytest.fixture
def graded_dataset() -> Dataset:
    """Three rows with references and retrieval ids."""
    return Dataset(
        question=["q1", "q2", "q3"],
        context=[["c1"], ["c2"], ["c3"]],
        response=["Paris", "London", "Rome"],
        reference=["Paris", "Berlin", "Rome"],
        retrieved_ids=[["d1"], ["d9"], ["d3", "d4"]],
        reference_ids=[["d1"], ["d2"], ["d4"]],
    )


def test_full_run_with_zero_llm_calls(
    graded_dataset: Dataset,
) -> None:
    """A complete evaluation that never touches a model."""
    llm = FakeLLM()
    result = evaluate(
        graded_dataset,
        llm=llm,
        metrics=[exact_match, token_f1, hit_rate, recall_at_k],
        run_config=SERIAL,
    )

    assert llm.prompts == [], "a deterministic run must not prompt"
    assert result.scores[0] == [1.0, 0.0, 1.0]
    assert result.failed_count == 0
    assert len(result.summary()) == 4


def test_retrieval_preset_runs(graded_dataset: Dataset) -> None:
    """RETRIEVAL_METRICS is usable as a one-liner."""
    result = evaluate(
        graded_dataset,
        llm=FakeLLM(),
        metrics=RETRIEVAL_METRICS,
        run_config=SERIAL,
    )
    assert len(result.summary()) == len(RETRIEVAL_METRICS)
    assert result.summary()[0].value is not None


def test_mixed_llm_and_deterministic_metrics(
    graded_dataset: Dataset,
) -> None:
    """The two tiers compose in one run."""
    llm = FakeLLM(responses=["0.8"])
    result = evaluate(
        graded_dataset,
        llm=llm,
        metrics=[exact_match, response_relevancy],
        run_config=SERIAL,
    )
    assert result.scores[0] == [1.0, 0.0, 1.0]
    assert result.scores[1] == [0.8, 0.8, 0.8]
    assert len(llm.prompts) == 3, "only the judged metric prompts"


def test_summary_reports_per_metric_aggregates(
    graded_dataset: Dataset,
) -> None:
    """Each metric gets its own row with an error bar."""
    result = evaluate(
        graded_dataset,
        llm=FakeLLM(),
        metrics=[exact_match, hit_rate],
        run_config=SERIAL,
    )
    by_name = {item.name: item for item in result.summary()}
    assert by_name["Exact Match"].value == pytest.approx(2 / 3)
    assert by_name["Exact Match"].stderr is not None
    assert by_name["Hit Rate"].value == pytest.approx(2 / 3)


def test_threshold_gating_across_tiers(
    graded_dataset: Dataset,
) -> None:
    """Deterministic metrics gate a build just like judged ones."""
    strict = exact_match.model_copy(update={"threshold": 0.9})
    lenient = exact_match.model_copy(update={"threshold": 0.5})

    assert (
        evaluate(
            graded_dataset,
            llm=FakeLLM(),
            metrics=[strict],
            run_config=SERIAL,
        ).passed
        is False
    )
    assert (
        evaluate(
            graded_dataset,
            llm=FakeLLM(),
            metrics=[lenient],
            run_config=SERIAL,
        ).passed
        is True
    )


def test_decorated_metric_runs_in_a_real_evaluation(
    graded_dataset: Dataset,
) -> None:
    """A one-expression metric is a first class citizen."""

    @metric(name="Starts with capital")
    def starts_upper(response: str) -> bool:
        return response[:1].isupper()

    result = evaluate(
        graded_dataset,
        llm=FakeLLM(),
        metrics=[starts_upper],
        run_config=SERIAL,
    )
    assert result.scores == [[1.0, 1.0, 1.0]]


# ------------------- validation before spending -------------------


def test_missing_reference_column_fails_before_any_work() -> None:
    """A reference metric on a dataset without references fails fast."""
    plain = Dataset(question=["q"], context=[["c"]], response=["r"])
    with pytest.raises(ValidationError, match="reference"):
        evaluate(
            plain,
            llm=FakeLLM(),
            metrics=[exact_match],
            run_config=SERIAL,
        )


def test_missing_id_columns_fail_before_any_work() -> None:
    """Ranking metrics need ids, and say so up front."""
    plain = Dataset(question=["q"], context=[["c"]], response=["r"])
    with pytest.raises(ValidationError, match="retrieved_ids"):
        evaluate(
            plain,
            llm=FakeLLM(),
            metrics=[hit_rate],
            run_config=SERIAL,
        )


def test_validation_names_the_offending_metric() -> None:
    """The error must be actionable, not just 'invalid'."""
    plain = Dataset(question=["q"], context=[["c"]], response=["r"])
    with pytest.raises(ValidationError, match="Exact Match"):
        evaluate(
            plain,
            llm=FakeLLM(),
            metrics=[exact_match],
            run_config=SERIAL,
        )


def test_populated_optional_column_passes_validation(
    graded_dataset: Dataset,
) -> None:
    """The inverse: when the data is there, validation allows it."""
    result = evaluate(
        graded_dataset,
        llm=FakeLLM(),
        metrics=[exact_match],
        run_config=SERIAL,
    )
    assert result.failed_count == 0


def test_deterministic_run_is_reproducible(
    graded_dataset: Dataset,
) -> None:
    """Same input, same output, every time."""
    runs = [
        evaluate(
            graded_dataset,
            llm=FakeLLM(),
            metrics=[exact_match, token_f1, hit_rate],
            run_config=SERIAL,
        ).scores
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_parallel_matches_serial_for_deterministic_metrics(
    graded_dataset: Dataset,
) -> None:
    """Concurrency must not change the answer."""
    serial = evaluate(
        graded_dataset,
        llm=FakeLLM(),
        metrics=[exact_match, hit_rate],
        run_config=SERIAL,
    )
    parallel = evaluate(
        graded_dataset,
        llm=FakeLLM(),
        metrics=[exact_match, hit_rate],
        run_config=RunConfig(show_progress=False, max_workers=4),
    )
    assert serial.scores == parallel.scores
