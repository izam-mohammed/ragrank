"""A module to evaluate all of the metrics"""

import logging
from time import time
from typing import List

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.metric import (
    BaseMetric,
    context_relevancy,
    context_utilization,
    response_conciseness,
    response_relevancy,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_dataset() -> Dataset:
    """Fixture for generating a sample datanode."""
    return Dataset(
        question=["What causes earthquakes?"],
        context=[
            [
                (
                    "Earthquakes are natural geological phenomena that occur"
                    " due to the movement of tectonic plates beneath the "
                    "Earth's surface. These movements can cause stress to "
                    "build up along fault lines, leading to sudden "
                    "releases of energy in the form of seismic waves."
                ),
            ]
        ],
        response=[
            (
                "Earthquakes are primarily caused by the movement "
                "of tectonic plates beneath the Earth's surface, "
                "which leads to the accumulation of stress along "
                "fault lines and subsequent release of energy."
            )
        ],
    )


@pytest.fixture
def metrics() -> List[BaseMetric]:
    """List of metrics"""

    return [
        response_conciseness,
        response_relevancy,
        context_relevancy,
        context_utilization,
    ]


@pytest.mark.openai
def test_metrics(
    metrics: List[BaseMetric], sample_dataset: Dataset
) -> None:
    """Test the dataset with each metric."""

    logger.info("Starting the metric wise evaluation")
    for metric in metrics:
        start = time()
        try:
            result = evaluate(
                dataset=sample_dataset, metrics=[metric]
            )
            logger.info(f"result - {result}")
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Error evaluating metric {metric.name}: {e}"
            )
            continue

        time_cost = time() - start
        logger.info(
            f"==== Completed the metric '{metric.name}'"
            f" in {time_cost:.3f} seconds ===="
        )
