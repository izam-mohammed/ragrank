from typing import List

import pytest
from pandas import DataFrame
from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import DataNode, Dataset, from_dict


@pytest.fixture
def valid_datanode() -> dict:
    return {
        "question": "What is the capital of France?",
        "context": ["France is a country located in Europe."],
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def invalid_datanode() -> dict:
    return {
        "question": 123,  # Invalid question (not a string)
        "context": ["France is a country located in Europe."],
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def valid_dataset() -> dict:
    return {
        "question": ["Q1", "Q2"],
        "context": [["C1"], ["C2"]],
        "response": ["R1", "R2"],
    }


@pytest.fixture
def invalid_dataset() -> dict:
    return {
        "question": ["Q1", "Q2"],
        "context": [["C1"], ["C2"]],
        "response": ["R1"],  # Invalid, missing response for one question
    }


def test_valid_datanode(valid_datanode):
    data_node = DataNode(**valid_datanode)
    assert isinstance(data_node, DataNode)


def test_invalid_datanode(invalid_datanode):
    with pytest.raises(Exception):
        DataNode(**invalid_datanode)


def test_valid_dataset(valid_dataset):
    dataset = Dataset(**valid_dataset)
    assert isinstance(dataset, Dataset)


def test_invalid_dataset(invalid_dataset):
    with pytest.raises(ValueError):
        Dataset(**invalid_dataset)


def test_datanode_to_dataset(valid_datanode):
    data_node = DataNode(**valid_datanode)
    dataset = data_node.to_dataset()
    assert isinstance(dataset, Dataset)


def test_dataset_len(valid_dataset):
    dataset = Dataset(**valid_dataset)
    assert len(dataset) == 2


def test_dataset_subscription(valid_dataset):
    dataset = Dataset(**valid_dataset)
    assert dataset[0].question == "Q1"
    assert dataset[0].context == ["C1"]
    assert dataset[0].response == "R1"


def test_dataset_append(valid_datanode):
    dataset = Dataset(question=["Q1"], context=[["C1"]], response=["R1"])
    data_node = DataNode(**valid_datanode)
    dataset.append(data_node)
    assert len(dataset) == 2


def test_dataset_addition(valid_dataset):
    dataset1 = Dataset(**valid_dataset)
    dataset2 = Dataset(question=["Q3"], context=[["C3"]], response=["R3"])
    combined_dataset = dataset1 + dataset2
    assert len(combined_dataset) == 3


def test_dataset_to_pandas(valid_dataset):
    dataset = Dataset(**valid_dataset)
    df = dataset.to_pandas()
    assert isinstance(df, DataFrame)
    assert len(df) == 2
    assert df.shape == (2, 3)


def test_from_dict_datanode(valid_datanode):
    data_node = from_dict(valid_datanode)
    assert isinstance(data_node, DataNode)


def test_from_dict_dataset(valid_dataset):
    dataset = from_dict(valid_dataset)
    assert isinstance(dataset, Dataset)


def test_empty_input_data_node():
    with pytest.raises(ValidationError):
        DataNode()


def test_empty_input_dataset():
    with pytest.raises(ValidationError):
        Dataset()
