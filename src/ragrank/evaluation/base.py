"""evaluation: the main module"""

from __future__ import annotations

import logging
from time import time

from ragrank.bridge.pydantic import validate_call
from ragrank.dataset import DataNode, Dataset, from_dict
from ragrank.evaluation.outputs import EvalResult
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric, response_relevancy

logger = logging.getLogger(__name__)


@validate_call(validate_return=False)
def evaluate(
    data: Dataset | DataNode | dict,
    *,
    llm: BaseLLM | None = None,
    metrics: BaseMetric | list[BaseMetric] | None = None,
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
    if isinstance(data, dict):
        data = from_dict(data)
    if isinstance(data, DataNode):
        data = data.to_dataset()
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
            for datanode in data.with_progress("Evaluating")
        ]
        for metric in metrics
    ]
    logger.info(f"Evaluation completed with {len(metrics)} metrics")
    delta = time() - dt

    result = EvalResult(
        llm=llm,
        metrics=metrics,
        dataset=data,
        response_time=delta,
        scores=scores,
    )

    return result
