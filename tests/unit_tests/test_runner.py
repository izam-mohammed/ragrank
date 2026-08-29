"""Tests for the evaluation runner and result aggregation."""

from __future__ import annotations

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.evaluation import EvalResult, RunConfig
from ragrank.exceptions import ValidationError
from ragrank.llm import BaseLLM, FakeLLM, LLMResult
from ragrank.metric import (
    InstructConfig,
    MetricType,
    response_conciseness,
    response_relevancy,
)

SERIAL = RunConfig(show_progress=False, max_workers=1)


@pytest.fixture
def dataset() -> Dataset:
    """Three rows of trivial data."""
    return Dataset(
        question=["q1", "q2", "q3"],
        context=[["c1"], ["c2"], ["c3"]],
        response=["r1", "r2", "r3"],
    )


def test_issue_46_evaluate_honours_the_llm_argument(
    dataset: Dataset,
) -> None:
    """github.com/izam-mohammed/ragrank/issues/46.

    Passing an LLM to evaluate() must actually reach the metrics. It
    previously did not, so evaluate(llm=...) still demanded an OpenAI
    key -- reported in 2024, closed, never fixed.
    """
    llm = FakeLLM(responses=["0.5"])
    result = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert result.scores == [[0.5, 0.5, 0.5]]
    assert len(llm.prompts) == 3


def test_metric_keeps_its_own_llm_over_the_runs(
    dataset: Dataset,
) -> None:
    """An explicit LLM on a metric wins over the run's LLM."""
    own = FakeLLM(responses=["0.1"])
    metric = response_relevancy.model_copy(update={"llm": own})
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.9"]),
        metrics=[metric],
        run_config=SERIAL,
    )
    assert result.scores == [[0.1, 0.1, 0.1]]


def test_with_llm_does_not_mutate_the_shared_metric() -> None:
    """Binding an LLM must copy, so module-level metrics stay clean."""
    bound = response_relevancy.with_llm(FakeLLM())
    assert bound is not response_relevancy
    assert response_relevancy.llm is None


def test_one_bad_row_does_not_destroy_the_run(
    dataset: Dataset,
) -> None:
    """A row the judge fluffs is recorded, not raised."""
    llm = FakeLLM(
        response_fn=lambda prompt: (
            "banana" if "q2" in prompt else "0.7"
        )
    )
    result = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert result.scores == [[0.7, None, 0.7]]
    assert result.failed_count == 1
    assert result.results is not None
    assert result.results[0][1].error is not None


def test_transport_failure_is_recorded_not_raised(
    dataset: Dataset,
) -> None:
    """An LLM that throws yields an error-carrying result."""

    class BrokenLLM(BaseLLM):
        @property
        def name(self) -> str:
            return "Broken LLM"

        def generate_text(self, text: str) -> LLMResult:
            raise ConnectionError("upstream is down")

    result = evaluate(
        dataset,
        llm=BrokenLLM(),
        metrics=[response_relevancy],
        run_config=RunConfig(
            show_progress=False,
            max_workers=1,
            max_retries=1,
            backoff=0.0,
        ),
    )
    assert result.scores == [[None, None, None]]
    assert result.results is not None
    assert "upstream is down" in result.results[0][0].error


def test_raise_on_error_aborts_the_run(dataset: Dataset) -> None:
    """Opting in to fail-fast propagates the original exception."""

    class BrokenLLM(BaseLLM):
        @property
        def name(self) -> str:
            return "Broken LLM"

        def generate_text(self, text: str) -> LLMResult:
            raise ConnectionError("upstream is down")

    with pytest.raises(ConnectionError):
        evaluate(
            dataset,
            llm=BrokenLLM(),
            metrics=[response_relevancy],
            run_config=RunConfig(
                show_progress=False,
                max_workers=1,
                raise_on_error=True,
            ),
        )


def test_concurrency_matches_serial_results(
    dataset: Dataset,
) -> None:
    """Parallel and serial runs must agree."""
    scripted = {"q1": "0.1", "q2": "0.5", "q3": "0.9"}

    def answer(prompt: str) -> str:
        return next(v for k, v in scripted.items() if k in prompt)

    serial = evaluate(
        dataset,
        llm=FakeLLM(response_fn=answer),
        metrics=[response_relevancy, response_conciseness],
        run_config=SERIAL,
    )
    parallel = evaluate(
        dataset,
        llm=FakeLLM(response_fn=answer),
        metrics=[response_relevancy, response_conciseness],
        run_config=RunConfig(show_progress=False, max_workers=4),
    )
    assert (
        serial.scores
        == parallel.scores
        == [
            [0.1, 0.5, 0.9],
            [0.1, 0.5, 0.9],
        ]
    )


def test_empty_dataset_is_rejected_before_spending_tokens() -> None:
    """Validation happens up front, not part way through a paid run."""
    llm = FakeLLM()
    with pytest.raises(ValidationError):
        evaluate(
            Dataset(question=[], context=[], response=[]),
            llm=llm,
            run_config=SERIAL,
        )
    assert llm.prompts == []


def test_missing_column_is_rejected_before_spending_tokens(
    dataset: Dataset,
) -> None:
    """A metric needing a field the data lacks fails fast and cheap."""
    from ragrank.metric import CustomInstruct

    metric = CustomInstruct(
        config=InstructConfig(
            metric_type=MetricType.BINARY,
            name="Needs more",
            instructions="...",
            input_fields=["question", "not_a_field"],
        )
    )
    llm = FakeLLM()
    with pytest.raises(ValidationError, match="not_a_field"):
        evaluate(
            dataset, llm=llm, metrics=[metric], run_config=SERIAL
        )
    assert llm.prompts == []


def test_summary_reports_mean_stderr_and_failures(
    dataset: Dataset,
) -> None:
    """Aggregates exist, and carry an error bar."""
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.2", "0.4", "0.6"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    summary = result.summary()[0]
    assert summary.name == "Response Relevancy"
    assert summary.count == 3
    assert summary.scored == 3
    assert summary.failed == 0
    assert summary.value == pytest.approx(0.4)
    assert summary.stderr == pytest.approx(0.11547, rel=1e-3)
    assert summary.minimum == pytest.approx(0.2)
    assert summary.maximum == pytest.approx(0.6)


def test_threshold_gives_a_pass_fail_verdict(
    dataset: Dataset,
) -> None:
    """A score nobody can gate on is a score nobody acts on."""
    strict = response_relevancy.model_copy(update={"threshold": 0.8})
    lenient = response_relevancy.model_copy(
        update={"threshold": 0.1}
    )

    passing = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.9"]),
        metrics=[lenient],
        run_config=SERIAL,
    )
    failing = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[strict],
        run_config=SERIAL,
    )
    assert passing.passed is True
    assert failing.passed is False
    assert failing.summary()[0].pass_rate == pytest.approx(0.0)


def test_no_threshold_means_no_verdict(dataset: Dataset) -> None:
    """Metrics without a threshold must not fabricate a pass/fail."""
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert result.passed is None


def test_results_keep_reasons_and_timings(
    dataset: Dataset,
) -> None:
    """evaluate() used to discard everything except the bare float."""
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert result.results is not None
    first = result.results[0][0]
    assert first.datanode.question == "q1"
    assert first.process_time is not None
    assert first.process_time >= 0


def test_to_json_works_without_pandas(dataset: Dataset) -> None:
    """JSON output must not need an optional dependency."""
    import json

    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    payload = json.loads(result.to_json())
    assert payload["summary"][0]["value"] == pytest.approx(0.5)
    assert payload["data"]["Response Relevancy"] == [0.5, 0.5, 0.5]


def test_eval_result_still_accepts_bare_scores(
    dataset: Dataset,
) -> None:
    """The pre-existing EvalResult construction path keeps working."""
    result = EvalResult(
        llm=FakeLLM(),
        metrics=[response_relevancy],
        dataset=dataset,
        scores=[[1.0, 1.0, 1.0]],
        response_time=0.1,
    )
    assert result.summary()[0].value == pytest.approx(1.0)
