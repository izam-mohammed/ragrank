"""Tests for defining a metric in one expression."""

from __future__ import annotations

import pytest
from ragrank.dataset import DataNode
from ragrank.llm import FakeLLM
from ragrank.metric import (
    FunctionMetric,
    Guidelines,
    LLMJudge,
    MetricType,
    metric,
)

NODE = DataNode(
    question="q",
    context=["c"],
    response="See [1] for detail.",
    reference="See [1].",
)


# --------------------------- @metric ---------------------------


def test_decorator_with_arguments() -> None:
    """A decorated function becomes a usable metric."""

    @metric(name="Has citation", threshold=1.0)
    def has_citation(response: str) -> bool:
        return "[" in response

    assert isinstance(has_citation, FunctionMetric)
    assert has_citation.name == "Has citation"
    result = has_citation.score(NODE)
    assert result.score == 1.0
    assert result.passed is True


def test_decorator_bare() -> None:
    """@metric works without parentheses, deriving the name."""

    @metric
    def word_count_ratio(response: str) -> float:
        return min(len(response.split()) / 10, 1.0)

    assert word_count_ratio.name == "Word Count Ratio"
    assert 0.0 <= word_count_ratio.score(NODE).score <= 1.0


def test_decorator_injects_only_requested_fields() -> None:
    """Parameters are injected by name from the DataNode."""
    seen = {}

    @metric
    def spy(response: str, reference: str) -> float:
        seen["response"] = response
        seen["reference"] = reference
        return 1.0

    spy.score(NODE)
    assert seen == {
        "response": "See [1] for detail.",
        "reference": "See [1].",
    }


def test_decorator_derives_required_columns() -> None:
    """The runner gets validation for free from the signature."""

    @metric
    def two_fields(question: str, response: str) -> float:
        return 1.0

    assert two_fields.required_columns == {"question", "response"}


def test_decorator_rejects_unknown_parameters() -> None:
    """A typo in a parameter name fails loudly at definition time."""
    with pytest.raises(ValueError, match="nonsense"):

        @metric
        def bad(nonsense: str) -> float:
            return 1.0


def test_decorator_coerces_bool_to_float() -> None:
    """Returning a bool is idiomatic and must work."""

    @metric
    def truthy(response: str) -> bool:
        return True

    assert truthy.score(NODE).score == 1.0


def test_decorator_can_abstain() -> None:
    """Returning None means 'not applicable to this row'."""

    @metric
    def sometimes(response: str) -> float | None:
        return None

    result = sometimes.score(NODE)
    assert result.score is None
    assert result.error is not None


def test_decorator_rejects_out_of_range_scores() -> None:
    """A metric that returns 47 for a 0-1 scale is a bug, not a score."""

    @metric
    def wild(response: str) -> float:
        return 47.0

    result = wild.score(NODE)
    assert result.score is None
    assert "outside the valid range" in result.error


def test_decorator_honours_a_custom_score_range() -> None:
    """A metric may declare its own bounds."""

    @metric(score_range=(0.0, 100.0))
    def percentage(response: str) -> float:
        return 47.0

    assert percentage.score(NODE).score == 47.0


def test_decorator_metric_needs_no_llm() -> None:
    """Function metrics never touch a model."""

    @metric
    def anything(response: str) -> float:
        return 1.0

    assert anything.llm is None
    assert anything.prompt is None


# --------------------------- LLMJudge ---------------------------


def test_llm_judge_maps_choices_to_scores() -> None:
    """The model picks a label; the numbers stay in Python."""
    judge = LLMJudge(
        judge_name="Tone",
        instructions="Is the tone right?",
        rubric={"A": 1.0, "B": 0.5, "C": 0.0},
        llm=FakeLLM(responses=["B"]),
    )
    assert judge.score(NODE).score == 0.5
    assert judge.name == "Tone"


def test_llm_judge_builds_a_prompt_from_its_configuration() -> None:
    """Instructions and choices must reach the model."""
    llm = FakeLLM(responses=["A"])
    LLMJudge(
        judge_name="X",
        instructions="SENTINEL_INSTRUCTION",
        rubric={"A": 1.0, "B": 0.0},
        llm=llm,
    ).score(NODE)
    assert "SENTINEL_INSTRUCTION" in llm.prompts[0]
    assert "A, B" in llm.prompts[0]


def test_llm_judge_rejects_an_off_rubric_answer() -> None:
    """A label outside the rubric is not silently coerced."""
    judge = LLMJudge(
        judge_name="X",
        instructions="...",
        rubric={"A": 1.0, "B": 0.0},
        llm=FakeLLM(responses=["Z"]),
        max_retries=0,
    )
    result = judge.score(NODE)
    assert result.score is None
    assert "expected one of" in result.error


def test_llm_judge_input_fields_drive_required_columns() -> None:
    """Only the declared fields are shown to the judge."""
    judge = LLMJudge(
        judge_name="X",
        instructions="...",
        input_fields=["question", "response"],
    )
    assert judge.required_columns == {"question", "response"}


def test_llm_judge_accepts_few_shot_examples() -> None:
    """Examples calibrate the judge and must reach the prompt."""
    llm = FakeLLM(responses=["A"])
    LLMJudge(
        judge_name="X",
        instructions="...",
        input_fields=["response"],
        rubric={"A": 1.0, "B": 0.0},
        examples=[{"response": "EXAMPLE_TEXT", "verdict": "A"}],
        llm=llm,
    ).score(NODE)
    assert "EXAMPLE_TEXT" in llm.prompts[0]


# --------------------------- Guidelines ---------------------------


def test_guidelines_pass_and_fail() -> None:
    """A plain-English rule becomes a binary metric."""
    rule = "Never give medical advice."
    passing = Guidelines(
        judge_name="No advice",
        guidelines=rule,
        llm=FakeLLM(responses=["pass"]),
    )
    failing = Guidelines(
        judge_name="No advice",
        guidelines=rule,
        llm=FakeLLM(responses=["fail"]),
    )
    assert passing.score(NODE).score == 1.0
    assert passing.score(NODE).passed is True
    assert failing.score(NODE).score == 0.0
    assert failing.score(NODE).passed is False


def test_guidelines_is_binary_and_gates_by_default() -> None:
    """A policy check is pass/fail, and defaults to gating."""
    rule = Guidelines(judge_name="X", guidelines="Be nice.")
    assert rule.metric_type == MetricType.BINARY
    assert rule.threshold == 1.0


def test_guidelines_text_reaches_the_prompt() -> None:
    """The rule itself must be what the judge is shown."""
    llm = FakeLLM(responses=["pass"])
    Guidelines(
        judge_name="X", guidelines="SENTINEL_RULE", llm=llm
    ).score(NODE)
    assert "SENTINEL_RULE" in llm.prompts[0]
