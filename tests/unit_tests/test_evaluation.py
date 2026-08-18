"""Test cases for evaluation module"""

from __future__ import annotations

import pytest
from pandas import DataFrame
from ragrank import evaluate
from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation import EvalResult
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric
from ragrank.metric.base import MetricType
from ragrank.prompt import Prompt


@pytest.fixture
def sample_dataset() -> Dataset:
    """Fixture for generating a sample dataset."""
    return Dataset(
        question=["sample question"],
        context=[["sample context"]],
        response=["sample response"],
    )


@pytest.fixture
def sample_datanode() -> DataNode:
    """Fixture for generating a sample data node."""
    return DataNode(
        question="sample question",
        context=["sample context"],
        response="sample response",
    )


@pytest.fixture
def sample_data_dict() -> dict[str, str | list[str]]:
    """Fixture for generating a sample data dictionary."""
    return {
        "question": "sample question",
        "context": ["sample context"],
        "response": "sample response",
    }


@pytest.fixture
def mock_metric() -> BaseMetric:
    """Fixture to create a mock metric."""
    BaseMetric.__abstractmethods__ = set()
    prompt = Prompt(
        name="Mock",
        instructions="",
        examples=[{"input": "", "output": ""}],
        input_keys=["input"],
        output_key="output",
    )
    return BaseMetric(
        metric_type=MetricType.NON_BINARY,
        llm=default_llm(),
        prompt=prompt,
    )


def test_evalresult_methods(
    sample_dataset: Dataset,
    mock_metric: BaseMetric,
) -> None:
    """Test methods of the EvalResult class."""
    eval_result = EvalResult(
        llm=default_llm(),
        metrics=[mock_metric],
        dataset=sample_dataset,
        scores=[[1.0]],
        response_time=0.1,
    )

    df = eval_result.to_dataframe()
    assert isinstance(
        df, DataFrame
    ), "Result should be a pandas DataFrame."


def test_evaluate_invalid_data_number() -> None:
    """Test evaluate function with invalid data number."""
    with pytest.raises(ValidationError):
        evaluate(123)


def test_evaluate_invalid_data() -> None:
    """Test evaluate function with invalid data."""
    with pytest.raises(ValueError):
        evaluate(
            {"invalid_key": "invalid_value"},
            metrics=[],
        )


def test_evaluate_invalid_llm(
    sample_dataset: Dataset,
) -> None:
    """Test evaluate function with invalid language model."""
    with pytest.raises(ValidationError):
        evaluate(
            sample_dataset,
            llm="invalid_llm",
            metrics=[],
        )


def test_evaluate_invalid_metrics(
    sample_dataset: Dataset,
) -> None:
    """Test evaluate function with invalid metrics."""
    with pytest.raises(ValidationError):
        evaluate(
            sample_dataset, metrics="invalid_metrics"
        )
    with pytest.raises(ValidationError):
        evaluate(
            sample_dataset,
            metrics=["invalid_metric1", "invalid_metric2"],
        )
