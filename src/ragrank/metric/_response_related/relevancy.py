"""Response relevancy metric"""

from time import time
from typing import Union

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric.base import BaseMetric, MetricResult, MetricType
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import RESPONSE_RELEVANCY_PROMPT


class ResponseRelevancy(BaseMetric):
    """A metric to evaluate the relevancy of a response.

    Attributes:
        metric_type (MetricType): The type of metric, which is non-binary.
        llm (BaseLLM): The language model used to generate the response.
        prompt (Prompt): The prompt provided for generating the response.

    Methods:
        name(): Get the name of the metric.
        score(data: DataNode): Compute the relevancy score for a response.
        _reason(data: DataNode, score: float): Provide a reason
            for the computed score.
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
        """Method to provide a reason for the prediction.

        Args:
            data (DataNode): The data node for which the score was computed.
            score (Union[int, float]): The computed score.

        Returns:
            str: A reason for the prediction.
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
        """Method to provide a reason for the prediction.
        Not implemented ...

        Args:
            data (DataNode): The data node for which the score was computed.
            score (Union[int, float]): The computed score.

        Returns:
            str: A reason for the prediction.
        """
        raise NotImplementedError


response_relevancy = ResponseRelevancy()
