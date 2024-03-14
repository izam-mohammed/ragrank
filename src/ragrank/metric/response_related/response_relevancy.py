"""Response relevancy metric"""

from time import time
from typing import Union

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric, MetricResult
from ragrank.metric.base import MetricType
from ragrank.prompt import Prompt

PROMPT: Prompt = Prompt(
    name="Answer Relevancy",
    instructions="You are good in determining the relevancy of the answer. Your response should be a float value. give a score between 0 and 1. don't have to explain anything. Your response should be a one number",  # noqa: E501
    examples=[
        {
            "question": "How are you",
            "response": "I am fine",
            "relevancy": "0.99",
        },
        {
            "question": "How are you",
            "response": "The whether is very bad",
            "relevancy": "0.02",
        },
    ],
    input_keys=["question", "response"],
    output_key="relevancy",
)


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

    metric_type: MetricType = MetricType.NON_BINARY
    llm: BaseLLM = Field(default_factory=lambda: default_llm())
    prompt: Prompt = Field(default_factory=lambda: PROMPT)

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

        Args:
            data (DataNode): The data node for which the score was computed.
            score (Union[int, float]): The computed score.

        Returns:
            str: A reason for the prediction.
        """
        return "There is no reason for it"


response_relevancy = ResponseRelevancy()
