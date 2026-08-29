"""Tests for token and cost accounting."""

from __future__ import annotations

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.evaluation import RunConfig, TokenUsage
from ragrank.llm import BaseLLM, FakeLLM, LLMResult
from ragrank.metric import (
    exact_match,
    faithfulness,
    response_relevancy,
)

SERIAL = RunConfig(show_progress=False, max_workers=1)


@pytest.fixture
def dataset() -> Dataset:
    """Two rows with a reference."""
    return Dataset(
        question=["q1", "q2"],
        context=[["c"], ["c"]],
        response=["r", "r"],
        reference=["r", "r"],
    )


# --------------------------- TokenUsage ---------------------------


def test_total_tokens_sums_both_directions() -> None:
    """Total is prompt plus response."""
    usage = TokenUsage(
        prompt_tokens=100, response_tokens=25, calls=1
    )
    assert usage.total_tokens == 125


def test_cost_at_given_rates() -> None:
    """Rates are per single token."""
    usage = TokenUsage(
        prompt_tokens=1_000_000, response_tokens=1_000_000, calls=1
    )
    cost = usage.cost(
        per_prompt_token=0.15 / 1e6, per_response_token=0.60 / 1e6
    )
    assert cost == pytest.approx(0.75)


def test_usage_adds() -> None:
    """Two records combine field by field."""
    combined = TokenUsage(
        prompt_tokens=10, response_tokens=2, calls=1
    ) + TokenUsage(
        prompt_tokens=5,
        response_tokens=1,
        calls=1,
        unreported_calls=1,
    )
    assert combined.prompt_tokens == 15
    assert combined.response_tokens == 3
    assert combined.calls == 2
    assert combined.unreported_calls == 1


def test_is_complete_flags_missing_data() -> None:
    """A total built from partial data is a lower bound, and says so."""
    assert TokenUsage(calls=2).is_complete is True
    assert (
        TokenUsage(calls=2, unreported_calls=1).is_complete is False
    )


def test_repr_is_readable_and_flags_gaps() -> None:
    """The default rendering should be legible in a terminal."""
    complete = repr(
        TokenUsage(prompt_tokens=1500, response_tokens=20, calls=3)
    )
    assert "3 calls" in complete
    assert "1,520 tokens" in complete

    partial = repr(TokenUsage(calls=3, unreported_calls=2))
    assert "2 without usage data" in partial


def test_str_matches_repr() -> None:
    """print() and repr() agree."""
    usage = TokenUsage(prompt_tokens=10, response_tokens=2, calls=1)
    assert str(usage) == repr(usage)


def test_usage_is_frozen() -> None:
    """A record of what happened is not mutable."""
    with pytest.raises(Exception):  # noqa: B017, PT011
        TokenUsage().calls = 5


# --------------------------- in a run ---------------------------


def test_run_reports_usage(dataset: Dataset) -> None:
    """Every judged call is counted."""
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.8"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert result.usage.calls == 2
    assert result.usage.prompt_tokens > 0
    assert result.usage.response_tokens > 0
    assert result.usage.is_complete


def test_result_cost_helper(dataset: Dataset) -> None:
    """EvalResult.cost() forwards to the usage record."""
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.8"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert result.cost(
        per_prompt_token=1.0, per_response_token=1.0
    ) == pytest.approx(result.usage.total_tokens)


def test_deterministic_run_makes_no_calls(dataset: Dataset) -> None:
    """A run with no judged metrics must not touch a model at all."""
    result = evaluate(
        dataset,
        llm=FakeLLM(),
        metrics=[exact_match],
        run_config=SERIAL,
    )
    assert result.usage.calls == 0
    assert result.usage.total_tokens == 0


def test_multi_call_metrics_are_counted_per_call(
    dataset: Dataset,
) -> None:
    """Faithfulness makes 1 extraction + N verification calls a row."""
    llm = FakeLLM(
        response_fn=lambda p: '["one", "two"]'
        if p.startswith("Claim Extraction")
        else "A"
    )
    result = evaluate(
        dataset, llm=llm, metrics=[faithfulness], run_config=SERIAL
    )
    # 2 rows x (1 extraction + 2 verifications)
    assert result.usage.calls == 6


def test_usage_counts_a_metric_with_its_own_llm(
    dataset: Dataset,
) -> None:
    """Tracking must not depend on where the model came from."""
    metric = response_relevancy.model_copy(
        update={"llm": FakeLLM(responses=["0.5"])}
    )
    result = evaluate(dataset, metrics=[metric], run_config=SERIAL)
    assert result.usage.calls == 2


def test_provider_without_usage_data_is_flagged(
    dataset: Dataset,
) -> None:
    """A model that reports nothing makes the total a lower bound."""

    class Silent(BaseLLM):
        @property
        def name(self) -> str:
            return "Silent"

        def generate_text(self, text: str) -> LLMResult:
            return LLMResult(response="0.5")

    result = evaluate(
        dataset,
        llm=Silent(),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert result.usage.calls == 2
    assert result.usage.unreported_calls == 2
    assert result.usage.is_complete is False


def test_usage_is_correct_under_concurrency(
    dataset: Dataset,
) -> None:
    """The tracker is shared across threads and must not lose counts."""
    big = Dataset(
        question=[f"q{i}" for i in range(50)],
        context=[["c"]] * 50,
        response=["r"] * 50,
    )
    result = evaluate(
        big,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_relevancy],
        run_config=RunConfig(show_progress=False, max_workers=8),
    )
    assert result.usage.calls == 50


def test_usage_appears_in_json(dataset: Dataset) -> None:
    """Cost data should survive serialisation."""
    import json

    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.8"]),
        metrics=[response_relevancy],
        run_config=SERIAL,
    )
    assert json.loads(result.to_json())["usage"]["calls"] == 2
