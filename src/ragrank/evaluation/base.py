"""evaluation: the main module"""

from __future__ import annotations

import logging
from time import perf_counter

from ragrank.bridge.pydantic import validate_call
from ragrank.dataset import DataNode, Dataset, from_dict
from ragrank.evaluation.outputs import EvalResult
from ragrank.evaluation.runner import RunConfig, run_metrics
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric import BaseMetric, response_relevancy

logger = logging.getLogger(__name__)


@validate_call(validate_return=False)
def evaluate(
    data: Dataset | DataNode | dict,
    *,
    llm: BaseLLM | None = None,
    metrics: BaseMetric | list[BaseMetric] | None = None,
    run_config: RunConfig | None = None,
) -> EvalResult:
    """
    Evaluate the performance of a given dataset using specified metrics.

    Parameters:
        data (Union[Dataset, DataNode, dict]): The dataset to be evaluated.
            It can be provided either as a `Dataset` object, `DataNode` object,
            or a `dict` representing the dataset.
        llm (Optional[BaseLLM]): The LLM (Language Model) used for evaluation.
            Metrics that do not carry their own LLM use this one. If None,
            a default LLM will be used.
        metrics (Optional[Union[BaseMetric, List[BaseMetric]]]): The metric or
            list of metrics used for evaluation. If None,
            response relevancy metric will be used.
        run_config (Optional[RunConfig]): How the run executes --
            concurrency, retries, progress. Defaults to `RunConfig()`.

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
    if metrics is None:
        metrics = [response_relevancy]
    if isinstance(metrics, BaseMetric):
        metrics = [metrics]

    started = perf_counter()
    results = run_metrics(
        data, metrics, llm=llm, config=run_config
    )
    elapsed = perf_counter() - started

    logger.info(
        "Evaluated %d datapoints with %d metrics in %.2fs",
        len(data),
        len(metrics),
        elapsed,
    )

    return EvalResult(
        llm=llm or default_llm(),
        metrics=metrics,
        dataset=data,
        response_time=elapsed,
        scores=[[item.score for item in row] for row in results],
        results=results,
    )
