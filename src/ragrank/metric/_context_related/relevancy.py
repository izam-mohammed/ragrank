"""Context relevancy metric"""

from __future__ import annotations

from ragrank.bridge.pydantic import Field
from ragrank.metric.base import ChunkwiseLLMMetric, MetricType
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import (
    CONTEXT_RELEVANCY_PROMPT,
    CONTEXT_RELEVANCY_RUBRIC,
)


class ContextRelevancy(ChunkwiseLLMMetric):
    """How relevant the retrieved context is to the question.

    Each retrieved chunk is judged on its own and the verdicts are
    averaged, so a single irrelevant chunk among several good ones is
    visible in `metadata["chunk_scores"]` rather than being smoothed
    away by one whole-list verdict.

    Attributes:
        metric_type (MetricType): The type of metric, which is non-binary.
        llm (BaseLLM | None): The language model used to judge.
        prompt (Prompt): The prompt used to elicit the score.
        rubric (dict[str, float]): Choice labels the judge picks from.
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of metric, which is non-binary.",
    )
    prompt: Prompt = Field(
        default=CONTEXT_RELEVANCY_PROMPT,
        description="The prompt provided for generating the response",
    )
    rubric: dict[str, float] | None = Field(
        default_factory=lambda: dict(CONTEXT_RELEVANCY_RUBRIC),
        description="Choice labels the judge picks from.",
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return "Context Relevancy"


context_relevancy = ContextRelevancy()
