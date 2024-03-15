"""evaluation: the main module"""

from time import time
from typing import List, Optional, Union

from numpy import mean

from ragrank._trace import EvaluationEvent, trace
from ragrank.bridge.pydantic import validate_call
from ragrank.dataset import DataNode, Dataset, from_dict
from ragrank.evaluation.outputs import EvalResult
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric, response_relevancy


@validate_call(validate_return=False)
def evaluate(
    dataset: Union[Dataset, DataNode, dict],
    *,
    llm: Optional[BaseLLM] = None,
    metrics: Optional[Union[BaseMetric, List[BaseMetric]]] = None,
) -> EvalResult:
    """The main evaluation function"""
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
        [metric.score(datanode).score for datanode in dataset]
        for metric in metrics
    ]
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
            metrics[i].name: mean(scores[i]) for i in range(len(metrics))
        },
        data_size=len(dataset),
    )
    trace(evaluation_event)
    return result
