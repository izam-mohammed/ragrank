"""Context utilization metric"""

from __future__ import annotations

from ragrank.bridge.pydantic import Field
from ragrank.metric.base import LLMMetric, MetricType
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import CONTEXT_UTILIZATION_PROMPT


class ContextUtilization(LLMMetric):
    """How well the response uses the retrieved context.

    Attributes:
        metric_type (MetricType): The type of metric, which is non-binary.
        llm (BaseLLM | None): The language model used to judge.
        prompt (Prompt): The prompt used to elicit the score.
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of metric, which is non-binary.",
    )
    prompt: Prompt = Field(
        default=CONTEXT_UTILIZATION_PROMPT,
        description="The prompt provided for generating the response",
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return "Context Utilization"


context_utilization = ContextUtilization()
