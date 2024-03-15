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
    return Dataset(
        question=["sample question"],
        context=[["sample context"]],
        response=["sample response"],
    )


@pytest.fixture
def sample_datanode():
    return DataNode(
        question="sample question",
        context=["sample context"],
        response="sample response",
    )


@pytest.fixture
def sample_data_dict():
    return {
        "question": "sample question",
        "context": ["sample context"],
        "response": "sample response",
    }


def test_evaluate_with_dataset(sample_dataset):
    result = evaluate(sample_dataset)
    assert isinstance(result, EvalResult)


def test_evaluate_with_datanode(sample_datanode):
    result = evaluate(sample_datanode)
    assert isinstance(result, EvalResult)


def test_evaluate_with_datadict(sample_data_dict):
    result = evaluate(sample_data_dict)
    assert isinstance(result, EvalResult)


def test_evaluate_default_behavior(sample_dataset):
    result = evaluate(sample_dataset)
    assert result.llm == default_llm()
    assert result.metrics == [response_relevancy]


def test_EvalResult_methods(sample_dataset):
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
    assert isinstance(df, DataFrame)


def test_evaluate_invalid_data_number():
    with pytest.raises(ValidationError):
        evaluate(123)


def test_evaluate_invalid_data():
    with pytest.raises(ValidationError):
        evaluate({"invalid_key": "invalid_value"})


def test_evaluate_invalid_llm():
    with pytest.raises(ValidationError):
        evaluate(sample_dataset, llm="invalid_llm")


def test_evaluate_invalid_metrics():
    with pytest.raises(ValidationError):
        evaluate(sample_dataset, metrics="invalid_metrics")
    with pytest.raises(ValidationError):
        evaluate(
            sample_dataset, metrics=["invalid_metric1", "invalid_metric2"]
        )
