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
from ragrank.target import Target, run_target

logger = logging.getLogger(__name__)


@validate_call(validate_return=False)
def evaluate(
    data: Dataset | DataNode | dict | list[str],
    *,
    llm: BaseLLM | None = None,
    metrics: BaseMetric | list[BaseMetric] | None = None,
    run_config: RunConfig | None = None,
    target: Target | None = None,
) -> EvalResult:
    """
    Evaluate the performance of a given dataset using specified metrics.

    Parameters:
        data (Union[Dataset, DataNode, dict, List[str]]): The dataset to
            be evaluated, as a `Dataset`, a `DataNode`, or a `dict`. With
            `target` set it may instead be a list of questions, which the
            target answers.
        llm (Optional[BaseLLM]): The LLM (Language Model) used for evaluation.
            Metrics that do not carry their own LLM use this one. If None,
            a default LLM will be used.
        metrics (Optional[Union[BaseMetric, List[BaseMetric]]]): The metric or
            list of metrics used for evaluation. If None,
            response relevancy metric will be used.
        run_config (Optional[RunConfig]): How the run executes --
            concurrency, retries, progress. Defaults to `RunConfig()`.
        target (Optional[Target]): The system under test. When given,
            `data` supplies the questions and the target is run to
            produce the responses and context, so the same question set
            can be pointed at a changed pipeline.

    Returns:
        EvalResult: An object containing the evaluation results.

    Raises:
        ValueError: If `data` is a list of questions but no `target`
            was given to answer them.

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

    Point a question set at your own pipeline instead::

        result = evaluate(
            ["who wrote it?"],
            target=my_rag_pipeline,
            metrics=[faithfulness],
        )
    """
    if isinstance(data, list) and target is None:
        raise ValueError(
            "A list of questions has no responses to score. Pass "
            "target= so ragrank can run your pipeline, or build a "
            "dataset that already has responses."
        )
    if isinstance(data, dict):
        data = from_dict(data)
    if isinstance(data, DataNode):
        data = data.to_dataset()

    if target is not None:
        config = run_config or RunConfig()
        data = run_target(
            data,
            target,
            max_workers=config.max_workers,
            max_retries=config.max_retries,
            backoff=config.backoff,
        )
    if metrics is None:
        metrics = [response_relevancy]
    if isinstance(metrics, BaseMetric):
        metrics = [metrics]

    started = perf_counter()
    results, usage = run_metrics(
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
        usage=usage,
    )
