"""Context Utilization metric"""

from __future__ import annotations

import logging
from time import time

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric.base import BaseMetric, MetricResult, MetricType
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import CONTEXT_UTILIZATION_PROMPT

logger = logging.getLogger(__name__)


class ContextUtilization(BaseMetric):
    """
    A metric to measure the utilization of context
    in language model responses.

    Attributes:
        metric_type (MetricType): The type of metric, which is non-binary.
        llm (BaseLLM): The language model used to generate the response.
        prompt (Prompt): The prompt provided for generating the response.

    Methods:
        name() -> str:
            Get the name for the metric.

        score(data: DataNode) -> MetricResult:
            Calculate the context utilization score for the given data.

        _reason(data: DataNode, score: float) -> Optional[str]:
            Provide a reason for the given data and score.
    """

    metric_type: MetricType = Field(
        default_factory=lambda: MetricType.NON_BINARY,
        description="The type of metric, which is non-binary.",
    )
    llm: BaseLLM = Field(
        default_factory=lambda: default_llm(),
        description="The language model used to generate the response.",
    )
    prompt: Prompt = Field(
        default_factory=lambda: CONTEXT_UTILIZATION_PROMPT,
        description="The prompt provided for generating the response",
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """

        return "Context Utilization"

    def score(self, data: DataNode) -> MetricResult:
        """Calculate the context utilization score for the given data.

        Args:
            data (DataNode): The data node containing the model dump.

        Returns:
            MetricResult: The context utilization score.
        """

        tm = time()
        prompt_str = self.prompt.to_string()
        prompt_dt = prompt_str.format(**data.model_dump())
        response = self.llm.generate_text(
            prompt_dt,
        )
        try:
            score = float(response.response)
        except ValueError:
            logger.error(
                f"Got unexpected LLM response - '{response.response}'"
            )
            raise ValueError(
                "Got unexpected response from the LLM"
            ) from ValueError
        delta = tm - time()
        return MetricResult(
            datanode=data,
            metric=self,
            score=score,
            reason=None,
            process_time=delta,
        )

    def _reason(self, data: DataNode, score: float) -> str | None:
        """Provide a reason for the given data and score.

        Args:
            data (DataNode): The data node containing the model dump.
            score (float): The context utilization score.

        Returns:
            str | None: The reason for the score.
        """

        raise NotImplementedError


context_utilization = ContextUtilization()
