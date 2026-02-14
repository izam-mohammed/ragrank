"""Test cases for evaluation module"""

from __future__ import annotations

from typing import Dict, List

import pytest
from pandas import DataFrame
from ragrank import evaluate
from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation import EvalResult
from ragrank.llm import default_llm
from ragrank.metric import response_relevancy


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
def sample_data_dict() -> Dict[str, str | List[str]]:
    """Fixture for generating a sample data dictionary."""
    return {
        "question": "sample question",
        "context": ["sample context"],
        "response": "sample response",
    }


@pytest.mark.openai
def test_evaluate_with_dataset(sample_dataset: Dataset) -> None:
    """Test evaluate function with a sample dataset."""
    result = evaluate(sample_dataset)
    assert isinstance(
        result, EvalResult
    ), "Result should be an instance of EvalResult."


@pytest.mark.openai
def test_evaluate_with_datanode(sample_datanode: DataNode) -> None:
    """Test evaluate function with a sample data node."""
    result = evaluate(sample_datanode)
    assert isinstance(
        result, EvalResult
    ), "Result should be an instance of EvalResult."


@pytest.mark.openai
def test_evaluate_with_datadict(
    sample_data_dict: Dict[str, str | List[str]],
) -> None:
    """Test evaluate function with a sample data dictionary."""
    result = evaluate(sample_data_dict)
    assert isinstance(
        result, EvalResult
    ), "Result should be an instance of EvalResult."


@pytest.mark.openai
def test_evaluate_default_behavior(sample_dataset: Dataset) -> None:
    """Test evaluate function default behavior."""
    result = evaluate(sample_dataset)
    assert (
        result.llm == default_llm()
    ), "LLM should be the default language model."
    assert result.metrics == [
        response_relevancy
    ], "Metrics should include response_relevancy."


def test_evalresult_methods(sample_dataset: Dataset) -> None:
    """Test methods of the EvalResult class."""
    llm = default_llm()
    metrics = [response_relevancy]
    dataset = sample_dataset
    scores = [[1.0]]
    response_time = 0.1

    eval_result = EvalResult(
        llm=llm,
        metrics=metrics,
        dataset=dataset,
        scores=scores,
        response_time=response_time,
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
        evaluate({"invalid_key": "invalid_value"})


def test_evaluate_invalid_llm() -> None:
    """Test evaluate function with invalid language model."""
    with pytest.raises(ValidationError):
        evaluate(sample_dataset, llm="invalid_llm")


def test_evaluate_invalid_metrics() -> None:
    """Test evaluate function with invalid metrics."""
    with pytest.raises(ValidationError):
        evaluate(sample_dataset, metrics="invalid_metrics")
    with pytest.raises(ValidationError):
        evaluate(
            sample_dataset,
            metrics=["invalid_metric1", "invalid_metric2"],
        )
