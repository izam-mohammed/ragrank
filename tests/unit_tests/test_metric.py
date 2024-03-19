"""Test cases for Metric Module"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric, MetricResult
from ragrank.metric.base import MetricType
from ragrank.prompt import Prompt


@pytest.fixture
def base_metric_dict_mock() -> BaseMetric:
    """Fixture to mock a dictionary for a base metric."""

    BaseMetric.__abstractmethods__ = set()
    return BaseMetric


@pytest.fixture
def llm_mock() -> BaseLLM:
    """Fixture to mock a default language model."""

    return default_llm()


@pytest.fixture
def prompt_mock() -> Prompt:
    """Fixture to mock a prompt."""

    return Prompt(
        name="Test Prompt",
        instructions="Testing instructions",
        examples=[
            {
                "input": "example input 1",
                "output": "example output 1",
            },
            {
                "input": "example input 2",
                "output": "example output 2",
            },
        ],
        input_keys=["input"],
        output_key="output",
    )


@pytest.fixture
def data_node_mock() -> DataNode:
    """Fixture to mock a data node."""

    return DataNode(
        question="sample question",
        context=["sample context"],
        response="sample response",
    )


@pytest.fixture
def base_metric_dict(
    llm_mock: BaseLLM, prompt_mock: Prompt
) -> Dict[str, BaseLLM | Prompt | MetricType]:
    """Fixture to create a dictionary for a base metric."""

    return {
        "metric_type": MetricType.BINARY,
        "llm": llm_mock,
        "prompt": prompt_mock,
    }


def test_base_metric_initialization(
    base_metric_dict: Dict[str, BaseLLM | Prompt | MetricType],
    llm_mock: BaseLLM,
    prompt_mock: Prompt,
    base_metric_dict_mock: BaseMetric,
) -> None:
    """Test for base metric initialization."""

    base_metric = base_metric_dict_mock(**base_metric_dict)
    assert (
        base_metric.metric_type == MetricType.BINARY
    ), "Base metric type mismatch"
    assert base_metric.llm == llm_mock, "Language model mismatch"
    assert base_metric.prompt == prompt_mock, "Prompt mismatch"


@pytest.fixture
def metric_result_mock(
    base_metric_dict: Dict[str, BaseLLM | Prompt | MetricType],
    data_node_mock: DataNode,
    base_metric_dict_mock: BaseMetric,
) -> None:
    """Fixture to mock a metric result."""

    return {
        "datanode": data_node_mock,
        "metric": base_metric_dict_mock(**base_metric_dict),
        "score": 0.75,
        "reason": "Some reason",
    }


def test_metric_result_initialization(
    metric_result_mock: Dict[str, Any],
    base_metric_dict: Dict[str, BaseLLM | Prompt | MetricType],
    data_node_mock: DataNode,
    base_metric_dict_mock: BaseMetric,
) -> None:
    """Test for metric result initialization."""

    metric_result = MetricResult(**metric_result_mock)
    assert (
        metric_result.datanode == data_node_mock
    ), "Data node mismatch"
    assert metric_result.metric == base_metric_dict_mock(
        **base_metric_dict
    ), "Metric dictionary mismatch"
    assert metric_result.score == 0.75, "Score mismatch"
    assert metric_result.reason == "Some reason", "Reason mismatch"


def test_metric_result_default_values(
    metric_result_mock: Dict[str, Any],
) -> None:
    """Test for default values of metric result."""

    metric_result = MetricResult(**metric_result_mock)
    assert (
        metric_result.process_time is None
    ), "Process time not initialized properly"
