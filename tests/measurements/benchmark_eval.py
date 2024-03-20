"""Benchmarking the evaluation module"""

from __future__ import annotations

import logging
import os
from time import time
from typing import List, Optional

from datasets.arrow_dataset import Dataset
from ragrank import evaluate
from ragrank.constants import DEBUG_MODE
from ragrank.dataset import ColumnMap, from_hfdataset
from ragrank.metric import (
    BaseMetric,
    context_relevancy,
    response_relevancy,
)

os.environ[DEBUG_MODE] = "true"

# logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[
        logging.FileHandler(os.path.join("logs", "ragrank_logs.log")),
    ],
)


def benchmark(
    metrics: Optional[List[BaseMetric] | BaseMetric] = None,
    data: Optional[Dataset] = None,
) -> None:
    """Check the benchmarks of the metrics"""

    if metrics is None:
        metrics = [response_relevancy]

    if data is None:
        data = from_hfdataset(
            "izammohammed/engineering_qa",
            split="train",
            column_map=ColumnMap(response="answers"),
        )

    logging.info(f" loaded the data contain {len(data)} datapoints ")

    start_time = time()
    result = evaluate(
        dataset=data,
        metrics=metrics,
    )
    end_time = time() - start_time
    avg_sum = [
        sum(metric_vals) / len(metric_vals)
        for metric_vals in result.scores
    ]

    logging.info(
        f" took {end_time / 60} minute to complete evaluation "
    )
    logging.info(f"average results metric wise - {avg_sum}")


if __name__ == "__main__":
    benchmark(metrics=[context_relevancy])
