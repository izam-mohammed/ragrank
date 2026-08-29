"""Tests for the assertion helpers."""

from __future__ import annotations

import pytest
from ragrank import evaluate
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation import RunConfig
from ragrank.llm import FakeLLM
from ragrank.metric import (
    context_relevancy,
    faithfulness,
    response_relevancy,
)
from ragrank.testing import (
    MetricAssertionError,
    assert_evaluation,
    assert_metric,
    assert_no_regression,
)

SERIAL = RunConfig(show_progress=False, max_workers=1)
NODE = DataNode(
    question="Where is the Eiffel Tower?",
    context=["The Eiffel Tower is in Paris."],
    response="It is in Paris. It was built in 1750.",
)


def claim_judge(prompt: str) -> str:
    """Ground the Paris claim, reject the 1750 one."""
    if prompt.startswith("Claim Extraction"):
        return '["It is in Paris.", "It was built in 1750."]'
    claim = prompt.rsplit("claim:", 1)[1].split("\n")[0]
    return "A" if "Paris" in claim else "C"


# --------------------------- assert_metric ---------------------------


def test_passes_above_the_threshold() -> None:
    """A good score raises nothing."""
    assert_metric(
        NODE,
        response_relevancy,
        threshold=0.7,
        llm=FakeLLM(responses=["0.9"]),
    )


def test_fails_below_the_threshold() -> None:
    """A bad score fails the test with both numbers named."""
    with pytest.raises(MetricAssertionError) as caught:
        assert_metric(
            NODE,
            response_relevancy,
            threshold=0.7,
            llm=FakeLLM(responses=["0.4"]),
        )
    message = str(caught.value)
    assert "0.400" in message
    assert "0.700" in message


def test_is_an_assertion_error() -> None:
    """Test runners must treat it as a failure, not an error."""
    assert issubclass(MetricAssertionError, AssertionError)


def test_an_unscorable_row_fails_rather_than_passing() -> None:
    """Silence is not success."""
    with pytest.raises(
        MetricAssertionError, match="could not score"
    ):
        assert_metric(
            NODE,
            response_relevancy.model_copy(update={"max_retries": 0}),
            threshold=0.5,
            llm=FakeLLM(responses=["banana"]),
        )


def test_threshold_argument_overrides_the_metric() -> None:
    """The call site wins."""
    lenient = response_relevancy.model_copy(
        update={"threshold": 0.1}
    )
    with pytest.raises(MetricAssertionError):
        assert_metric(
            NODE,
            lenient,
            threshold=0.9,
            llm=FakeLLM(responses=["0.5"]),
        )


def test_metric_threshold_is_used_when_none_is_given() -> None:
    """A metric that already gates needs no argument."""
    strict = response_relevancy.model_copy(update={"threshold": 0.9})
    with pytest.raises(MetricAssertionError):
        assert_metric(NODE, strict, llm=FakeLLM(responses=["0.5"]))


def test_no_threshold_anywhere_is_a_usage_error() -> None:
    """Asserting nothing is a mistake, and says so clearly."""
    with pytest.raises(ValueError, match="no threshold"):
        assert_metric(
            NODE, response_relevancy, llm=FakeLLM(responses=["0.5"])
        )


def test_failure_names_the_unsupported_claim() -> None:
    """A faithfulness failure should point at the invented sentence."""
    with pytest.raises(MetricAssertionError) as caught:
        assert_metric(
            NODE,
            faithfulness,
            threshold=0.9,
            llm=FakeLLM(response_fn=claim_judge),
        )
    assert "It was built in 1750." in str(caught.value)


def test_failure_shows_per_chunk_scores() -> None:
    """A chunkwise failure should show which chunk let it down."""
    node = DataNode(
        question="q",
        context=["good chunk", "bad chunk"],
        response="r",
    )
    with pytest.raises(MetricAssertionError) as caught:
        assert_metric(
            node,
            context_relevancy,
            threshold=0.9,
            llm=FakeLLM(
                response_fn=lambda p: "A" if "good" in p else "C"
            ),
        )
    assert "Per chunk scores" in str(caught.value)


# ------------------------ assert_evaluation ------------------------


def test_evaluation_passes(dataset_fixture: Dataset) -> None:
    """A passing dataset returns its result for further inspection."""
    result = assert_evaluation(
        dataset_fixture,
        [response_relevancy.model_copy(update={"threshold": 0.5})],
        llm=FakeLLM(responses=["0.9"]),
        run_config=SERIAL,
    )
    assert result.passed is True


def test_evaluation_fails_and_names_the_metric(
    dataset_fixture: Dataset,
) -> None:
    """The failure message must say which metric and what score."""
    with pytest.raises(MetricAssertionError) as caught:
        assert_evaluation(
            dataset_fixture,
            [
                response_relevancy.model_copy(
                    update={"threshold": 0.9}
                )
            ],
            llm=FakeLLM(responses=["0.2"]),
            run_config=SERIAL,
        )
    assert "Response Relevancy" in str(caught.value)


def test_evaluation_without_a_threshold_is_a_usage_error(
    dataset_fixture: Dataset,
) -> None:
    """Nothing to gate on is a mistake."""
    with pytest.raises(ValueError, match="threshold"):
        assert_evaluation(
            dataset_fixture,
            [response_relevancy],
            llm=FakeLLM(responses=["0.5"]),
            run_config=SERIAL,
        )


# ----------------------- assert_no_regression -----------------------


@pytest.fixture
def dataset_fixture() -> Dataset:
    """Six rows."""
    return Dataset(
        question=[f"q{i}" for i in range(6)],
        context=[["c"]] * 6,
        response=["r"] * 6,
    )


def test_no_regression_passes_on_an_improvement(
    dataset_fixture: Dataset,
) -> None:
    """Getting better is not a regression."""
    before = evaluate(
        dataset_fixture,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    after = evaluate(
        dataset_fixture,
        llm=FakeLLM(responses=["0.9"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert_no_regression(before, after)


def test_no_regression_fails_on_a_real_drop(
    dataset_fixture: Dataset,
) -> None:
    """A significant drop must fail."""
    before = evaluate(
        dataset_fixture,
        llm=FakeLLM(
            responses=["0.5", "0.5", "0.6", "0.5", "0.5", "0.5"]
        ),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    after = evaluate(
        dataset_fixture,
        llm=FakeLLM(responses=["0.1"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    with pytest.raises(MetricAssertionError, match="Regression"):
        assert_no_regression(before, after)


def test_no_regression_tolerates_noise(
    dataset_fixture: Dataset,
) -> None:
    """A change within the noise must not fail a build."""
    before = evaluate(
        dataset_fixture,
        llm=FakeLLM(
            responses=["0.5", "0.5", "0.6", "0.5", "0.5", "0.5"]
        ),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    after = evaluate(
        dataset_fixture,
        llm=FakeLLM(
            responses=["0.5", "0.5", "0.6", "0.5", "0.5", "0.48"]
        ),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert_no_regression(before, after)
