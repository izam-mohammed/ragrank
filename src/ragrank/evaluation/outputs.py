"""Contains the ouputs of evaluation"""

from __future__ import annotations

import json
from statistics import fmean, stdev
from typing import TYPE_CHECKING, Any

from ragrank.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from ragrank.dataset import Dataset
from ragrank.dataset.reader import RAGRANK_DICT_TYPE
from ragrank.llm import BaseLLM
from ragrank.metric import BaseMetric, MetricResult
from ragrank.utils.optional import require

if TYPE_CHECKING:
    from pandas import DataFrame


class MetricSummary(BaseModel):
    """Aggregate statistics for one metric across a run.

    Attributes:
        name (str): The metric's name.
        count (int): Rows attempted.
        scored (int): Rows that produced a score.
        failed (int): Rows that did not.
        value (float | None): The metric's own aggregation of its
            scores -- the mean unless the metric says otherwise.
        stderr (float | None): Standard error of the mean. A score with
            no error bar over a handful of rows says very little.
        minimum (float | None): Lowest score.
        maximum (float | None): Highest score.
        pass_rate (float | None): Fraction of scored rows meeting the
            metric's threshold, if it has one.
        passed (bool | None): Whether the aggregate meets the threshold.
    """

    model_config: ConfigDict = ConfigDict(frozen=True)

    name: str
    count: int
    scored: int
    failed: int
    value: float | None = None
    stderr: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    pass_rate: float | None = None
    passed: bool | None = None


class EvalResult(BaseModel):
    """
    Represents the result of an evaluation.

    Attributes:
        llm (BaseLLM): The language model used for evaluation.
        metrics (List[BaseMetric]): List of metrics used for evaluation.
        dataset (Dataset): The dataset used for evaluation.
        scores (List[List[float | None]]):
            List of scores for each metric. None marks a row the metric
            could not score.
        results (List[List[MetricResult]] | None): Full per-row results,
            including reasons, errors and timings.
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
    scores: list[list[float | None]] = Field(
        description="List of scores for each metric"
    )
    results: list[list[MetricResult]] | None = Field(
        default=None,
        repr=False,
        description="Full per-row results, including reasons and errors.",
    )
    response_time: float = Field(
        gt=0, description="Response time for the evaluation process."
    )

    @model_validator(mode="after")
    def validator(self) -> EvalResult:
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
            ) from None

        dataset_size = len(self.dataset)
        for score in self.scores:
            if len(score) != dataset_size:
                raise ValueError(
                    "The number of datapoints and scores are not balanced. \n"
                    "Ensure that each score list has the same "
                    "length as dataset."
                ) from None

        return self

    def summary(self) -> list[MetricSummary]:
        """Aggregate the run, one row per metric.

        Returns:
            list[MetricSummary]: Aggregates including standard error and,
                where the metric has a threshold, a pass rate.
        """
        summaries = []
        for metric, row in zip(self.metrics, self.scores, strict=False):
            valid = [score for score in row if score is not None]
            value = metric.aggregate(valid)

            passing = (
                [score >= metric.threshold for score in valid]
                if metric.threshold is not None
                else []
            )

            summaries.append(
                MetricSummary(
                    name=metric.name,
                    count=len(row),
                    scored=len(valid),
                    failed=len(row) - len(valid),
                    value=value,
                    stderr=(
                        stdev(valid) / len(valid) ** 0.5
                        if len(valid) > 1
                        else None
                    ),
                    minimum=min(valid) if valid else None,
                    maximum=max(valid) if valid else None,
                    pass_rate=fmean(passing) if passing else None,
                    passed=(
                        value >= metric.threshold
                        if metric.threshold is not None
                        and value is not None
                        else None
                    ),
                )
            )
        return summaries

    @property
    def passed(self) -> bool | None:
        """Whether every metric with a threshold met it.

        Returns:
            bool | None: None when no metric declares a threshold.
        """
        verdicts = [
            item.passed
            for item in self.summary()
            if item.passed is not None
        ]
        return all(verdicts) if verdicts else None

    @property
    def failed_count(self) -> int:
        """How many (row, metric) pairs produced no score.

        Returns:
            int: The number of unscored pairs.
        """
        return sum(
            1 for row in self.scores for score in row if score is None
        )

    def to_dict(self) -> RAGRANK_DICT_TYPE:
        """
        Convert the evaluation result to a dict.

        Returns:
            dict: A dict containing the evaluation results.
        """
        dict_data = self.dataset.to_dict()
        for metric, row in zip(self.metrics, self.scores, strict=False):
            dict_data[metric.name] = row
        return dict_data

    def to_dataframe(self) -> DataFrame:
        """
        Convert the evaluation result to a pandas DataFrame.

        Returns:
            DataFrame: A DataFrame containing the evaluation results.
        """
        pandas = require("pandas", "pandas")
        return pandas.DataFrame(self.to_dict())

    def to_json(self, **kwargs: Any) -> str:  # noqa: ANN401
        """Serialise the run, without needing pandas.

        Args:
            **kwargs: Passed through to `json.dumps`.

        Returns:
            str: The run as JSON.
        """
        payload = {
            "llm": self.llm.name,
            "response_time": self.response_time,
            "data": self.to_dict(),
            "summary": [item.model_dump() for item in self.summary()],
        }
        return json.dumps(payload, **kwargs)

    def __repr__(self) -> str:
        """
        Return a string representation of the evaluation result.

        Returns:
            str: A string representation of the evaluation result.
        """
        parts = []
        for item in self.summary():
            value = (
                "n/a" if item.value is None else f"{item.value:.3f}"
            )
            text = f"{item.name}: {value}"
            if item.stderr is not None:
                text += f" +/- {item.stderr:.3f}"
            if item.failed:
                text += f" ({item.failed} unscored)"
            parts.append(text)
        return "\n".join(parts) if parts else "EvalResult(empty)"

    def __str__(self) -> str:
        """
        Return a string representation of the evaluation result.

        Returns:
            str: A string representation of the evaluation result.
        """
        return self.__repr__()
