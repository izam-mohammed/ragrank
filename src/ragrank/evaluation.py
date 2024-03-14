"""evaluation: the main module"""

from time import time
from typing import Annotated, List, Optional, Union

from pandas import DataFrame

from ragrank.bridge.pydantic import BaseModel, model_validator
from ragrank.dataset import DataNode, Dataset, from_dict
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric, response_relevancy


def evaluate(
    dataset: Union[Dataset, DataNode, dict],
    llm: Optional[BaseLLM] = None,
    metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
) -> str:
    """The main evaluation function"""
    if isinstance(dataset, dict):
        dataset = from_dict(dataset)
    if isinstance(dataset, DataNode):
        dataset = dataset.to_dataset()
    if llm is None:
        llm = default_llm()
    if metrics is None:
        metrics = [response_relevancy]
    if isinstance(metrics, BaseMetric):
        metrics = [metrics]

    dt = time()
    scores = [
        [metric.score(datanode).score for datanode in dataset]
        for metric in metrics
    ]
    delta = dt - time()

    return EvalResult(
        llm=llm,
        metrics=metrics,
        dataset=dataset,
        response_time=delta,
        scores=scores,
    )


class EvalResult(BaseModel):
    """The result set for evaluation result"""

    class Config:
        arbitrary_types_allowed = True

    llm: BaseLLM
    metrics: List[BaseMetric]
    dataset: Dataset
    scores: List[Annotated[List[float], "each metric"]]
    response_time: float

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

    def to_pandas(self) -> DataFrame:
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
