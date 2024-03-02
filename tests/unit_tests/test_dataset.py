import pytest
from ragrank.dataset import Dataset, from_dict
from ragrank.logger import logger


@pytest.fixture
def mock_dataset_dict():
    data = {
        "question": "Who is the precident of America ?",
        "context": ["The president of America is Trump"],
        "response": "Trump",
    }
    return data


def test_dataset(mock_dataset_dict):
    dataset = Dataset(**mock_dataset_dict)
    logger.info(f"{dataset}")
    assert True


def test_from_dataset(mock_dataset_dict):
    dataset = from_dict(mock_dataset_dict)
    logger.info(f"{dataset}")
    assert True
