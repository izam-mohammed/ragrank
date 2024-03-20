"""Response relevancy metric"""

import logging
from time import time
from typing import Union

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric.base import BaseMetric, MetricResult, MetricType
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import RESPONSE_RELEVANCY_PROMPT

logger = logging.getLogger(__name__)


class ResponseRelevancy(BaseMetric):
    """
    A class representing a metric for evaluating the relevancy of responses.

    Attributes:
        metric_type (MetricType): The type of metric, which is non-binary.
        llm (BaseLLM): The language model used to generate the response.
        prompt (Prompt): The prompt provided for generating the response.

    Methods:
        name(self) -> str:
            Get the name for the metric.

        score(self, data: DataNode) -> Union[float, int]:
            Calculate the score for the response relevancy metric
                based on the provided data.
        _reason(self, data: DataNode, score: float) -> str:
            Determine the reason for the given score.
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
        default_factory=lambda: RESPONSE_RELEVANCY_PROMPT,
        description="The prompt provided for generating the response",
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """

        return "Response Relevancy"

    def score(self, data: DataNode) -> Union[float, int]:
        """Calculate the score for the response relevancy metric.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            Union[float, int]: The score indicating the
                relevancy of the response.
        """

        tm = time()
        prompt_str = self.prompt.to_string()
        prompt_dt = prompt_str.format(**data.model_dump())
        response = self.llm.generate_text(
            prompt_dt,
        )
        try:
            response = float(response.response)
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
            score=response,
            reason=None,
            process_time=delta,
        )

    def _reason(self, data: DataNode, score: float) -> str:
        """Determine the reason for the given score.
        Not implemented yet.

        Args:
            data (DataNode): The input data used for scoring.
            score (float): The score indicating the
                relevancy of the response.

        Returns:
            str: The reason for the given score.
        """

        raise NotImplementedError


response_relevancy = ResponseRelevancy()
