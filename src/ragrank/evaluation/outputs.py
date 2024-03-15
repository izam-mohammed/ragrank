"""Contains the ouputs of evaluation"""

from typing import Annotated, List

from pandas import DataFrame

from ragrank.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from ragrank.dataset import Dataset
from ragrank.llm import BaseLLM
from ragrank.metric import BaseMetric


class EvalResult(BaseModel):
    """The result set for evaluation result"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: BaseLLM
    metrics: List[BaseMetric]
    dataset: Dataset
    scores: List[Annotated[List[float], "each metric"]]
    response_time: float = Field(gt=0)

    @model_validator(mode="after")
    def validator(self) -> "EvalResult":
        """Validation for the eval result"""

        if len(self.metrics) != len(self.scores):
            raise ValueError(
                "The number of metrics and number of scores is not equal"
            ) from ValueError

        dataset_size = len(self.dataset)
        for score in self.scores:
            if len(score) != dataset_size:
                raise ValueError(
                    "The number of datapoints and scores are not balanced"
                ) from ValueError

        return self

    def to_dataframe(self) -> DataFrame:
        """convert the result to a pandas dataframe"""

        df = self.dataset.to_pandas()
        for i in range(len(self.metrics)):
            df[self.metrics[i].name] = self.scores[i]

        return df

    def __repr__(self) -> str:
        """representation of the eval result"""
        data = [
            {
                self.metrics[k].name: self.scores[k][i]
                for k in range(len(self.metrics))
            }
            for i in range(len(self.dataset))
        ]
        return str(data)

    def __str__(self) -> str:
        """str representation of the model"""
        return self.__repr__()
