"""Correctness against a reference answer."""

from __future__ import annotations

from ragrank.bridge.pydantic import Field
from ragrank.metric.base import LLMMetric, MetricType
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import (
    CORRECTNESS_PROMPT,
    CORRECTNESS_RUBRIC,
)


class Correctness(LLMMetric):
    """Is the response right, judged against a known answer.

    The reference-based counterpart to `exact_match` and `token_f1`:
    those compare strings, so "Paris" and "The capital is Paris" score
    poorly despite being the same answer. This judges meaning, and
    tolerates differences in wording, length and formatting.

    Attributes:
        metric_type (MetricType): The type of metric, which is non-binary.
        llm (BaseLLM | None): The language model used to judge.
        prompt (Prompt): The prompt used to elicit the verdict.
        rubric (dict[str, float]): Choice labels the judge picks from.
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of metric, which is non-binary.",
    )
    prompt: Prompt = Field(
        default=CORRECTNESS_PROMPT,
        description="The prompt provided for generating the response",
    )
    rubric: dict[str, float] | None = Field(
        default_factory=lambda: dict(CORRECTNESS_RUBRIC),
        description="Choice labels the judge picks from.",
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return "Correctness"


correctness = Correctness()
