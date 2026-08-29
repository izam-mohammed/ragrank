"""Contract tests every metric must satisfy, plus CustomMetric."""

from __future__ import annotations

import pytest
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, FakeLLM, LLMResult
from ragrank.metric import (
    RAG_TRIAD,
    RETRIEVAL_METRICS,
    BaseMetric,
    CustomMetric,
    ExactMatch,
    MetricResult,
    context_relevancy,
    context_utilization,
    exact_match,
    hit_rate,
    json_valid,
    response_conciseness,
    response_relevancy,
)

JUDGED = [
    response_relevancy,
    response_conciseness,
    context_relevancy,
    context_utilization,
]

NODE = DataNode(
    question="q",
    context=["c"],
    response="r",
    reference="r",
    retrieved_ids=["d1"],
    reference_ids=["d1"],
)


@pytest.mark.parametrize("metric", JUDGED)
def test_judged_metrics_carry_no_llm_by_default(
    metric: BaseMetric,
) -> None:
    """Importing ragrank must not construct an OpenAI client."""
    assert metric.llm is None


@pytest.mark.parametrize("metric", JUDGED)
def test_judged_metrics_have_a_prompt_and_a_name(
    metric: BaseMetric,
) -> None:
    """The basics every judged metric needs."""
    assert metric.prompt is not None
    assert metric.name
    assert repr(metric) == metric.name


@pytest.mark.parametrize("metric", JUDGED + RETRIEVAL_METRICS)
def test_every_metric_declares_a_score_range(
    metric: BaseMetric,
) -> None:
    """Bounds are what make an out-of-range answer detectable."""
    low, high = metric.score_range
    assert low < high


@pytest.mark.parametrize("metric", JUDGED)
def test_judged_metrics_required_columns_are_real_fields(
    metric: BaseMetric,
) -> None:
    """A metric cannot require a field that does not exist."""
    assert metric.required_columns <= set(DataNode.model_fields)


def test_rag_triad_is_the_three_diagnostic_metrics() -> None:
    """The preset must stay meaningful."""
    names = [item.name for item in RAG_TRIAD]
    assert names == [
        "Context Relevancy",
        "Context Utilization",
        "Response Relevancy",
    ]


def test_retrieval_preset_is_all_deterministic() -> None:
    """Nothing in this preset may cost an LLM call."""
    assert all(item.llm is None for item in RETRIEVAL_METRICS)
    assert all(item.prompt is None for item in RETRIEVAL_METRICS)


# --------------------------- with_llm ---------------------------


def test_with_llm_binds_when_the_metric_has_none() -> None:
    """The run's LLM is lent to metrics that need one."""
    llm = FakeLLM()
    bound = response_relevancy.with_llm(llm)
    assert bound.llm is llm
    assert response_relevancy.llm is None


def test_with_llm_respects_an_explicit_choice() -> None:
    """A metric's own LLM is never overridden."""
    own = FakeLLM(responses=["0.1"])
    metric = response_relevancy.model_copy(update={"llm": own})
    assert metric.with_llm(FakeLLM(responses=["0.9"])).llm is own


def test_with_llm_of_none_is_a_no_op() -> None:
    """No LLM offered means nothing to bind."""
    assert response_relevancy.with_llm(None) is response_relevancy


def test_resolve_llm_precedence() -> None:
    """Own LLM, then the run's, then the default."""
    own, run = FakeLLM(responses=["1"]), FakeLLM(responses=["2"])
    metric = response_relevancy.model_copy(update={"llm": own})
    assert metric.resolve_llm(run) is own
    assert response_relevancy.resolve_llm(run) is run


# --------------------------- aggregate ---------------------------


def test_default_aggregate_is_the_mean() -> None:
    """Unless a metric says otherwise."""
    assert response_relevancy.aggregate([0.0, 1.0]) == 0.5


def test_aggregate_of_nothing_is_none() -> None:
    """No scores means no aggregate, not zero."""
    assert response_relevancy.aggregate([]) is None


def test_a_metric_may_override_aggregation() -> None:
    """Metrics that do not reduce as a mean can say so."""

    class WorstCase(ExactMatch):
        """An ExactMatch that reports its worst row, not its mean."""

        def aggregate(self, scores: list[float]) -> float | None:
            """Reduce by minimum instead of mean."""
            return min(scores) if scores else None

    assert WorstCase().aggregate([0.2, 0.9]) == 0.2


# --------------------------- MetricResult ---------------------------


def test_passed_is_none_without_a_threshold() -> None:
    """No threshold means no verdict to give."""
    assert json_valid.score(NODE).passed is None


def test_passed_is_none_without_a_score() -> None:
    """An abstention cannot pass or fail."""
    metric = exact_match.model_copy(update={"threshold": 0.5})
    result = metric.score(
        DataNode(question="q", context=["c"], response="r")
    )
    assert result.score is None
    assert result.passed is None


def test_metric_result_is_frozen() -> None:
    """Results are values, not mutable state."""
    result = hit_rate.score(NODE)
    with pytest.raises(Exception):  # noqa: B017, PT011
        result.score = 0.0


def test_metric_result_metadata_defaults_to_empty() -> None:
    """Metrics without extra detail still have a usable dict."""
    assert hit_rate.score(NODE).metadata == {}


# --------------------------- CustomMetric ---------------------------


class Doubler(CustomMetric):
    """A CustomMetric that returns half the response length, capped."""

    @property
    def metric_name(self) -> str:
        """The metric's name."""
        return "Doubler"

    def metric_score(self, data: DataNode) -> float:
        """A tenth of the response length, capped at 1."""
        return min(len(data.response) / 10, 1.0)


class Broken(CustomMetric):
    """A CustomMetric that returns something that is not a number."""

    @property
    def metric_name(self) -> str:
        """The metric's name."""
        return "Broken"

    def metric_score(self, data: DataNode) -> float:
        """Deliberately return something that is not a number."""
        return "not a number"


class OutOfRange(CustomMetric):
    """A CustomMetric that returns a score outside its own range."""

    @property
    def metric_name(self) -> str:
        """The metric's name."""
        return "Out Of Range"

    def metric_score(self, data: DataNode) -> float:
        """Deliberately return a score outside the range."""
        return 47.0


def test_custom_metric_scores() -> None:
    """The subclass hook still works."""
    result = Doubler().score(NODE)
    assert isinstance(result, MetricResult)
    assert result.score == pytest.approx(0.1)
    assert Doubler().name == "Custom Metric - Doubler"


def test_custom_metric_with_a_non_numeric_result_abstains() -> None:
    """A broken metric records an error instead of raising."""
    result = Broken().score(NODE)
    assert result.score is None
    assert "not a number" in result.error


def test_custom_metric_out_of_range_abstains() -> None:
    """Range checking applies to custom metrics too."""
    result = OutOfRange().score(NODE)
    assert result.score is None
    assert "outside the valid range" in result.error


def test_custom_metric_process_time_is_positive() -> None:
    """Timings must be durations."""
    assert Doubler().score(NODE).process_time >= 0


# --------------------------- BaseLLM ---------------------------


def test_base_llm_generate_batches_through_generate_text() -> None:
    """The default batch implementation calls the single-text one."""
    llm = FakeLLM(responses=["a", "b"])
    results = llm.generate(["one", "two"])
    assert [item.response for item in results] == ["a", "b"]
    assert llm.prompts == ["one", "two"]


def test_base_llm_repr_is_its_name() -> None:
    """Readable in logs and error messages."""
    assert repr(FakeLLM()) == "Fake LLM"


def test_fake_llm_requires_at_least_one_response() -> None:
    """An empty script is a programmer error."""
    with pytest.raises(ValueError, match="at least one"):
        FakeLLM(responses=[]).generate_text("x")


def test_custom_llm_subclass_works() -> None:
    """Anyone can implement BaseLLM."""

    class Echo(BaseLLM):
        @property
        def name(self) -> str:
            return "Echo"

        def generate_text(self, text: str) -> LLMResult:
            return LLMResult(response=text[:3])

    assert Echo().generate_text("hello").response == "hel"
