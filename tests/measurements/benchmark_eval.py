"""Benchmarking the evaluation module"""

from __future__ import annotations

from time import time
from typing import List, Optional

from ragrank import evaluate
from ragrank.dataset import ColumnMap, from_hfdataset
from ragrank.logger import logger
from ragrank.metric import BaseMetric, response_relevancy


def benchmark(
    metrics: Optional[List[BaseMetric] | BaseMetric] = None,
) -> None:
    """Check the benchmarks of the metrics"""
    if metrics is None:
        metrics = [response_relevancy]

    data = from_hfdataset(
        "izammohammed/engineering_qa",
        split="train",
        column_map=ColumnMap(response="answers"),
    )
    logger.info(f"loaded the data contain {len(data)} datapoints")

    start_time = time()
    _ = evaluate(
        dataset=data,
        metrics=metrics,
    )
    end_time = time() - start_time

    logger.info("\n\n" + f"took {end_time} seconds")


if __name__ == "__main__":
    benchmark()
