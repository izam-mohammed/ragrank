import pytest
from ragrank.dataset import Dataset
from ragrank.logger import logger


@pytest.fixture
def mock_dataset_dict():
    data = {
        "question": "Who is the precident of America ?",
        "context": ["The president of America is Trump"],
        "respons": "Trump",
    }
    return data


def test_dataset(mock_dataset_dict):
    dataset = Dataset(**mock_dataset_dict)
    logger.info(f"{dataset}")
    assert True
