"""Custom metric"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import perf_counter

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM
from ragrank.metric.base import BaseMetric, MetricResult
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import NONE_PROMPT

logger = logging.getLogger(__name__)


class CustomMetric(BaseMetric, ABC):
    """
    A custom metric for evaluating responses.

    Attributes:
        llm (BaseLLM): The language model used to generate the response.
        prompt (Prompt): The prompt provided for generating the response.

    Methods:
        name(self) -> str:
            Get the name for the metric.

        score(self, data: DataNode) -> MetricResult:
            Calculate the score for the custom metric based on
                the provided data.

        metric_score(self, data: DataNode) -> float:
            Calculate the score for the custom metric based on
                the provided data.

        reason(self, data: DataNode, score: float) -> str | None:
            Determine the reason for the given score.
    """

    llm: BaseLLM | None = Field(
        default=None,
        description="The language model used to generate the response.",
    )
    prompt: Prompt = Field(
        default_factory=lambda: NONE_PROMPT,
        description="The prompt provisded for generating the response",
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """

        return f"Custom Metric - {self.metric_name}"

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the specific Metric

        Returns:
            str: The name of the actual metric.
        """

    def score(self, data: DataNode) -> MetricResult:
        """Calculate the score for the custom metric
        based on the provided data.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The result of the metric calculation.
        """

        started = perf_counter()
        raw = self.metric_score(data=data)
        try:
            score: float | None = float(raw)
        except (TypeError, ValueError):
            logger.error(
                "%s produced a non-numeric score: %r", self.name, raw
            )
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error=f"metric_score returned {raw!r}, which is not a number",
                process_time=perf_counter() - started,
            )

        low, high = self.score_range
        if not low <= score <= high:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error=(
                    f"score {score} is outside the valid range "
                    f"[{low}, {high}]"
                ),
                process_time=perf_counter() - started,
            )

        return MetricResult(
            datanode=data,
            metric=self,
            score=score,
            reason=self._reason(data=data, score=score),
            process_time=perf_counter() - started,
        )

    @abstractmethod
    def metric_score(self, data: DataNode) -> float:
        """Calculate the actual score for the custom metric
        based on the provided data.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            Float: The result score of the metric
        """

    def _reason(self, data: DataNode, score: float) -> str | None:
        """Provide a reason for the given score.

        Args:
            data (DataNode): The input data used for scoring.
            score (float): The score indicating the relevancy of the response.

        Returns:
            str | None: The reason for the given score.
        """
        return None
