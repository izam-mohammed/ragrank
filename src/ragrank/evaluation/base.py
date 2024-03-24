"""evaluation: the main module"""

from __future__ import annotations

import logging
from time import time
from typing import List, Optional

from numpy import mean

from ragrank._trace import EvaluationEvent, trace
from ragrank.bridge.pydantic import validate_call
from ragrank.dataset import DataNode, Dataset, from_dict
from ragrank.evaluation.outputs import EvalResult
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric, response_relevancy

logger = logging.getLogger(__name__)


@validate_call(validate_return=False)
def evaluate(
    dataset: Dataset | DataNode | dict,
    *,
    llm: Optional[BaseLLM] = None,
    metrics: Optional[BaseMetric | List[BaseMetric]] = None,
) -> EvalResult:
    """
    Evaluate the performance of a given dataset using specified metrics.

    Parameters:
        dataset (Union[Dataset, DataNode, dict]): The dataset to be evaluated.
            It can be provided either as a `Dataset` object, `DataNode` object,
            or a `dict` representing the dataset.
        llm (Optional[BaseLLM]): The LLM (Language Model) used for evaluation.
            If None, a default LLM will be used.
        metrics (Optional[Union[BaseMetric, List[BaseMetric]]]): The metric or
            list of metrics used for evaluation. If None,
            response relevancy metric will be used.

    Returns:
        EvalResult: An object containing the evaluation results.

    Examples::

        from ragrank import evaluate
        from ragrank.dataset import from_dict

        data = from_dict({
            "question": "Who is the 46th Prime Minister of US ?",
            "context": [
                "Joseph Robinette Biden is an American politician, "
                "he is the 46th and current president of the United States.",
            ],
            "response": "Joseph Robinette Biden",
        })
        result = evaluate(data)

        print(result)
    """
    if isinstance(dataset, dict):
        dataset = from_dict(dataset)
    if isinstance(dataset, DataNode):
        dataset = dataset.to_dataset()
    if llm is None:
        llm = default_llm()
    if metrics is None:
        metrics = [response_relevancy]
    if isinstance(metrics, BaseMetric):
        metrics = [metrics]

    dt = time()
    scores = [
        [
            metric.score(datanode).score
            for datanode in dataset.with_progress("Evaluating")
        ]
        for metric in metrics
    ]
    logger.info(f"Evaluation completed with {len(metrics)} metrics")
    delta = time() - dt

    result = EvalResult(
        llm=llm,
        metrics=metrics,
        dataset=dataset,
        response_time=delta,
        scores=scores,
    )
    evaluation_event = EvaluationEvent(
        llm=llm.name,
        time_cost=delta,
        metrics={
            metrics[i].name: mean(scores[i])
            for i in range(len(metrics))
        },
        data_size=len(dataset),
    )
    trace(evaluation_event)
    return result
