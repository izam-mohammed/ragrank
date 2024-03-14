"""Base module for metric"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM
from ragrank.prompt import Prompt


class MetricType(Enum):
    """Enumeration of metric types."""

    BINATY = "binary"
    NON_BINARY = "non_binary"


class BaseMetric(BaseModel, ABC):
    """Base class for defining metrics.

    Attributes:
        metric_type (MetricType): The type of the metric.
        llm (BaseLLM): The language model associated with the metric.
        prompt (Prompt): The prompt associated with the metric.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric_type: MetricType
    llm: BaseLLM
    prompt: Prompt

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """

    @abstractmethod
    def score(self, data: DataNode) -> Union[float, int]:
        """Method to compute the metric score.

        Args:
            data (DataNode): The data node for which the score is computed.

        Returns:
            Union[int, float]: The computed score.
        """

    @abstractmethod
    def _reason(self, data: DataNode, score: float) -> str:
        """Method to provide a reason for the prediction.

        Args:
            data (DataNode): The data node for which the score was computed.
            score (Union[int, float]): The computed score.

        Returns:
            str: A reason for the prediction.
        """

    def save(self) -> None:
        """Method to save the metric. Not implemented in base class."""
        raise NotImplementedError

    def load(self) -> None:
        """Method to load the metric. Not implemented in base class."""
        raise NotImplementedError


class MetricResult(BaseModel):
    """Class to hold the result of a metric computation.

    Attributes:
        datanode (DataNode): The data node associated with the metric result.
        metrics (List[BaseMetric]): List of metrics used in the computation.
        scores (List[Union[int, float]]): List of scores computed
            for each metric.
        reasons (List[str]): List of reasons corresponding to
            each metric score.
        process_time (Optional[float], optional): Processing time for
            the computation. Defaults to None.
    """

    datanode: DataNode
    metric: BaseMetric
    score: Union[float, int]
    reason: Optional[str]
    process_time: Optional[float] = Field(default=None, repr=False)
