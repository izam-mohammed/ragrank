"""Assertion helpers for evaluations in a test suite."""

from __future__ import annotations

from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation.base import evaluate
from ragrank.evaluation.compare import compare
from ragrank.evaluation.outputs import EvalResult
from ragrank.evaluation.runner import RunConfig
from ragrank.exceptions import RagRankError
from ragrank.llm import BaseLLM
from ragrank.metric import BaseMetric


class MetricAssertionError(RagRankError, AssertionError):
    """A metric did not meet its threshold.

    Subclasses AssertionError so test runners treat it as a failed
    assertion rather than an error, and RagRankError so it can be
    caught alongside the library's other exceptions.
    """


def assert_metric(
    data: DataNode,
    metric: BaseMetric,
    *,
    threshold: float | None = None,
    llm: BaseLLM | None = None,
) -> None:
    """Assert that one datapoint meets a metric's threshold.

    Args:
        data (DataNode): The datapoint to score.
        metric (BaseMetric): The metric to apply.
        threshold (float | None): Overrides the metric's own threshold.
        llm (BaseLLM | None): The model to judge with.

    Raises:
        MetricAssertionError: If the score is below the threshold, or
            the metric could not produce a score at all.

    Examples::

        assert_metric(node, response_relevancy, threshold=0.7)
    """
    __tracebackhide__ = True
    if threshold is not None:
        metric = metric.model_copy(update={"threshold": threshold})
    if metric.threshold is None:
        raise ValueError(
            f"{metric.name!r} has no threshold, so there is nothing "
            "to assert. Pass threshold= to assert_metric()."
        )

    result = metric.with_llm(llm).score(data)

    if result.score is None:
        raise MetricAssertionError(
            f"{metric.name} could not score this datapoint: "
            f"{result.error}"
        )
    if not result.passed:
        raise MetricAssertionError(
            f"{metric.name} scored {result.score:.3f}, "
            f"below the threshold of {metric.threshold:.3f}."
            + _detail(result)
        )


def assert_evaluation(
    data: Dataset | DataNode | dict,
    metrics: BaseMetric | list[BaseMetric],
    *,
    llm: BaseLLM | None = None,
    run_config: RunConfig | None = None,
) -> EvalResult:
    """Assert that a whole dataset passes every gating metric.

    Args:
        data (Dataset | DataNode | dict): The data to evaluate.
        metrics (BaseMetric | list[BaseMetric]): Metrics to apply. At
            least one must carry a threshold.
        llm (BaseLLM | None): The model to judge with.
        run_config (RunConfig | None): How the run executes.

    Returns:
        EvalResult: The result, so the test can inspect it further.

    Raises:
        MetricAssertionError: If any gating metric fell below its
            threshold.

    Examples::

        result = assert_evaluation(dataset, [strict_relevancy])
    """
    __tracebackhide__ = True
    result = evaluate(
        data, llm=llm, metrics=metrics, run_config=run_config
    )

    if result.passed is None:
        raise ValueError(
            "No metric declared a threshold, so there is nothing to "
            "assert. Set threshold= on at least one metric."
        )

    if not result.passed:
        failures = [
            f"{item.name} scored {item.value:.3f}"
            for item in result.summary()
            if item.passed is False and item.value is not None
        ]
        raise MetricAssertionError(
            "Evaluation failed: " + "; ".join(failures)
        )
    return result


def assert_no_regression(
    baseline: EvalResult, candidate: EvalResult
) -> None:
    """Assert that no metric got significantly worse.

    Changes within the noise are allowed through -- the point is to
    catch real regressions, not to demand that a number never moves.

    Args:
        baseline (EvalResult): The run to compare against.
        candidate (EvalResult): The new run.

    Raises:
        MetricAssertionError: If any metric regressed significantly.

    Examples::

        assert_no_regression(last_release, this_branch)
    """
    __tracebackhide__ = True
    diff = compare(baseline, candidate)
    if diff.regressed:
        raise MetricAssertionError(
            "Regression detected:\n"
            + "\n".join(repr(item) for item in diff.regressed)
        )


def _detail(result: object) -> str:
    """Extra context for a failure message, when the metric has any."""
    claims = result.metadata.get("claims")
    if claims:
        unsupported = [
            item["claim"]
            for item in claims
            if item.get("supported") == 0.0
        ]
        if unsupported:
            return (
                "\n  Unsupported claims:\n    - "
                + "\n    - ".join(unsupported)
            )

    chunks = result.metadata.get("chunk_scores")
    if chunks:
        return f"\n  Per chunk scores: {chunks}"

    if result.reason:
        return f"\n  Reason: {result.reason}"
    return ""
