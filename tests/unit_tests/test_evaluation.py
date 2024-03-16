import pytest
from pandas import DataFrame
from ragrank import evaluate
from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation import EvalResult
from ragrank.llm import default_llm
from ragrank.metric import response_relevancy


@pytest.fixture
def sample_dataset():
    """Fixture for generating a sample dataset."""
    return Dataset(
        question=["sample question"],
        context=[["sample context"]],
        response=["sample response"],
    )


@pytest.fixture
def sample_datanode():
    """Fixture for generating a sample data node."""
    return DataNode(
        question="sample question",
        context=["sample context"],
        response="sample response",
    )


@pytest.fixture
def sample_data_dict():
    """Fixture for generating a sample data dictionary."""
    return {
        "question": "sample question",
        "context": ["sample context"],
        "response": "sample response",
    }


def test_evaluate_with_dataset(sample_dataset):
    """Test evaluate function with a sample dataset."""
    result = evaluate(sample_dataset)
    assert isinstance(
        result, EvalResult
    ), "Result should be an instance of EvalResult."


def test_evaluate_with_datanode(sample_datanode):
    """Test evaluate function with a sample data node."""
    result = evaluate(sample_datanode)
    assert isinstance(
        result, EvalResult
    ), "Result should be an instance of EvalResult."


def test_evaluate_with_datadict(sample_data_dict):
    """Test evaluate function with a sample data dictionary."""
    result = evaluate(sample_data_dict)
    assert isinstance(
        result, EvalResult
    ), "Result should be an instance of EvalResult."


def test_evaluate_default_behavior(sample_dataset):
    """Test evaluate function default behavior."""
    result = evaluate(sample_dataset)
    assert (
        result.llm == default_llm()
    ), "LLM should be the default language model."
    assert result.metrics == [
        response_relevancy
    ], "Metrics should include response_relevancy."


def test_EvalResult_methods(sample_dataset):
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
    assert isinstance(df, DataFrame), "Result should be a pandas DataFrame."


def test_evaluate_invalid_data_number():
    """Test evaluate function with invalid data number."""
    with pytest.raises(ValidationError):
        evaluate(123)


def test_evaluate_invalid_data():
    """Test evaluate function with invalid data."""
    with pytest.raises(ValidationError):
        evaluate({"invalid_key": "invalid_value"})


def test_evaluate_invalid_llm():
    """Test evaluate function with invalid language model."""
    with pytest.raises(ValidationError):
        evaluate(sample_dataset, llm="invalid_llm")


def test_evaluate_invalid_metrics():
    """Test evaluate function with invalid metrics."""
    with pytest.raises(ValidationError):
        evaluate(sample_dataset, metrics="invalid_metrics")
    with pytest.raises(ValidationError):
        evaluate(
            sample_dataset, metrics=["invalid_metric1", "invalid_metric2"]
        )
