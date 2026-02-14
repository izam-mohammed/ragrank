"""Contains the ouputs of evaluation"""


from pandas import DataFrame

from ragrank.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from ragrank.dataset import Dataset
from ragrank.dataset.reader import RAGRANK_DICT_TYPE
from ragrank.llm import BaseLLM
from ragrank.metric import BaseMetric


class EvalResult(BaseModel):
    """
    Represents the result of an evaluation.

    Attributes:
        llm (BaseLLM): The language model used for evaluation.
        metrics (List[BaseMetric]): List of metrics used for evaluation.
        dataset (Dataset): The dataset used for evaluation.
        scores (List[List[float]]):
            List of scores for each metric.
        response_time (float): Response time for the evaluation process.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    llm: BaseLLM = Field(
        description="The language model used for evaluation"
    )
    metrics: list[BaseMetric] = Field(
        description="List of metrics used for evaluation."
    )
    dataset: Dataset = Field(
        description="The dataset used for evaluation"
    )
    scores: list[list[float]] = Field(
        description="List of scores for each metric"
    )
    response_time: float = Field(
        gt=0, description="Response time for the evaluation process."
    )

    @model_validator(mode="after")
    def validator(self) -> "EvalResult":
        """
        Validate the evaluation result after instantiation.

        Raises:
            ValueError: If the number of metrics and scores are not equal,
                or if the number of datapoints and scores are not balanced.
        """
        if len(self.metrics) != len(self.scores):
            raise ValueError(
                "The number of metrics and number of scores is not equal. \n"
                "Ensure that each metric has a corresponding score."
            ) from ValueError

        dataset_size = len(self.dataset)
        for score in self.scores:
            if len(score) != dataset_size:
                raise ValueError(
                    "The number of datapoints and scores are not balanced. \n"
                    "Ensure that each score list has the same "
                    "length as dataset."
                ) from ValueError

        return self

    def to_dict(self) -> RAGRANK_DICT_TYPE:
        """
        Convert the evaluation result to a dict.

        Returns:
            dict: A dict containing the evaluation results.
        """
        dict_data = self.dataset.to_dict()
        for i in range(len(self.metrics)):
            dict_data[self.metrics[i].name] = self.scores[i]
        return dict_data

    def to_dataframe(self) -> DataFrame:
        """
        Convert the evaluation result to a pandas DataFrame.

        Returns:
            DataFrame: A DataFrame containing the evaluation results.
        """
        df = self.dataset.to_dataframe()
        for i in range(len(self.metrics)):
            df[self.metrics[i].name] = self.scores[i]
        return df

    def __repr__(self) -> str:
        """
        Return a string representation of the evaluation result.

        Returns:
            str: A string representation of the evaluation result.
        """
        data = [
            {
                self.metrics[k].name: self.scores[k][i]
                for k in range(len(self.metrics))
            }
            for i in range(len(self.dataset))
        ]
        return str(data)

    def __str__(self) -> str:
        """
        Return a string representation of the evaluation result.

        Returns:
            str: A string representation of the evaluation result.
        """
        return self.__repr__()
