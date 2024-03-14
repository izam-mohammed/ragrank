import pytest
from ragrank.dataset import DataNode
from ragrank.llm import default_llm
from ragrank.metric import BaseMetric, MetricResult
from ragrank.metric.base import MetricType
from ragrank.prompt import Prompt


@pytest.fixture
def base_metric_dict_mock():
    BaseMetric.__abstractmethods__ = set()
    return BaseMetric


@pytest.fixture
def llm_mock():
    return default_llm()


@pytest.fixture
def prompt_mock():
    return Prompt(
        name="Test Prompt",
        instructions="Testing instructions",
        examples=[
            {"input": "example input 1", "output": "example output 1"},
            {"input": "example input 2", "output": "example output 2"},
        ],
        input_keys=["input"],
        output_key="output",
    )


@pytest.fixture
def data_node_mock():
    return DataNode(
        question="sample question",
        context=["sample context"],
        response="sample response",
    )


@pytest.fixture
def base_metric_dict(llm_mock, prompt_mock):
    return {
        "metric_type": MetricType.BINATY,
        "llm": llm_mock,
        "prompt": prompt_mock,
    }


def test_base_metric_initialization(
    base_metric_dict, llm_mock, prompt_mock, base_metric_dict_mock
):
    base_metric = base_metric_dict_mock(**base_metric_dict)
    assert base_metric.metric_type == MetricType.BINATY
    assert base_metric.llm == llm_mock
    assert base_metric.prompt == prompt_mock


@pytest.fixture
def metric_result_mock(
    base_metric_dict, data_node_mock, base_metric_dict_mock
):
    return {
        "datanode": data_node_mock,
        "metric": base_metric_dict_mock(**base_metric_dict),
        "score": 0.75,
        "reason": "Some reason",
    }


def test_metric_result_initialization(
    metric_result_mock, base_metric_dict, data_node_mock, base_metric_dict_mock
):
    metric_result = MetricResult(**metric_result_mock)
    assert metric_result.datanode == data_node_mock
    assert metric_result.metric == base_metric_dict_mock(**base_metric_dict)
    assert metric_result.score == 0.75
    assert metric_result.reason == "Some reason"


def test_metric_result_default_values(metric_result_mock):
    metric_result = MetricResult(**metric_result_mock)
    assert metric_result.process_time is None
