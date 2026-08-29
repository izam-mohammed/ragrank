"""Running metrics over a dataset.

The runner owns everything that is about *executing* an evaluation --
concurrency, retries, backoff, progress, fault tolerance -- so that
metrics only have to know how to score one row. Keeping metrics pure is
what makes running them in parallel safe.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean, median, mode, stdev
from time import perf_counter, sleep
from typing import Literal

from tqdm import tqdm

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation.usage import (
    TokenUsage,
    TrackedLLM,
    UsageTracker,
)
from ragrank.exceptions import ValidationError
from ragrank.llm import BaseLLM, default_llm
from ragrank.llm.cache import (
    CacheBackend,
    CachedLLM,
    DiskCache,
)
from ragrank.metric import BaseMetric, LLMMetric, MetricResult

logger = logging.getLogger(__name__)

Job = tuple[int, int, MetricResult]


class RunConfig(BaseModel):
    """Policy for how an evaluation run executes.

    Attributes:
        max_workers (int): Metric calls to run concurrently. 1 runs
            serially, which is easier to debug.
        max_retries (int): Retries after a failing LLM call, on top of
            any retries the metric performs for unparseable answers.
        backoff (float): Seconds to wait before the first retry,
            doubling each attempt.
        show_progress (bool): Display a progress bar, if tqdm is
            installed.
        raise_on_error (bool): Abort the run on the first failure
            rather than recording it and carrying on. Off by default:
            one bad row should not destroy a long, paid run.
        repetitions (int): Score each row this many times and reduce.
            Judges are not deterministic, and repeating makes that
            visible instead of presenting one sample as the truth.
        reducer (str): How to reduce repetitions: mean, median or mode.
        cache (CacheBackend | bool | None): Reuse judge responses for
            identical prompts. True uses an on-disk cache, or pass a
            CacheBackend of your own. Judges run at temperature 0 over
            a dataset that barely changes, so adding one metric would
            otherwise re-ask the others the same questions and bill you
            for it.
    """

    model_config: ConfigDict = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )

    max_workers: int = Field(
        default=4, ge=1, description="Concurrent metric calls."
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries after a failing LLM call.",
    )
    backoff: float = Field(
        default=0.5,
        ge=0.0,
        description="Initial retry delay, seconds.",
    )
    show_progress: bool = Field(
        default=True, description="Display a progress bar."
    )
    raise_on_error: bool = Field(
        default=False,
        description="Abort the run on the first failure.",
    )
    repetitions: int = Field(
        default=1,
        ge=1,
        description="Score each row this many times and reduce.",
    )
    reducer: Literal["mean", "median", "mode"] = Field(
        default="mean",
        description="How to reduce repetitions.",
    )
    cache: CacheBackend | bool | None = Field(
        default=None,
        description="Reuse judge responses for identical prompts.",
    )


def validate_dataset(
    dataset: Dataset, metrics: list[BaseMetric]
) -> None:
    """Check the dataset can satisfy every metric before spending tokens.

    Args:
        dataset (Dataset): The dataset about to be evaluated.
        metrics (list[BaseMetric]): The metrics about to run.

    Raises:
        ValidationError: If a metric needs a field the data lacks.
    """
    if not len(dataset):
        raise ValidationError("The dataset is empty.")

    known = set(DataNode.model_fields)
    # An optional column exists on the model but may hold no data.
    populated = {
        name
        for name in known
        if getattr(dataset, name, None) is not None
    }

    problems = []
    for metric in metrics:
        unknown = metric.required_columns - known
        if unknown:
            problems.append(
                f"{metric.name!r} needs unknown field(s) {sorted(unknown)}"
            )
            continue
        empty = metric.required_columns - populated
        if empty:
            problems.append(
                f"{metric.name!r} needs {sorted(empty)}, "
                "which the dataset does not provide"
            )

    if problems:
        raise ValidationError(
            "The dataset cannot satisfy every metric: "
            + "; ".join(problems)
            + f". Available fields are {sorted(populated)}."
        )


def _resolve_cache(
    cache: CacheBackend | bool | None,
) -> CacheBackend | None:
    """Turn the RunConfig setting into a backend, or None.

    Args:
        cache (CacheBackend | bool | None): The configured value.

    Returns:
        CacheBackend | None: The backend to use, if any.
    """
    if cache is None or cache is False:
        return None
    if cache is True:
        return DiskCache()
    return cache


def _with_tracking(
    metric: BaseMetric,
    llm: BaseLLM | None,
    tracker: UsageTracker,
    cache: CacheBackend | None,
) -> BaseMetric:
    """Bind a metric to a token-counting view of its language model.

    Deterministic metrics make no model calls, so they are left alone
    and a run using only those still needs no credentials.

    Args:
        metric (BaseMetric): The metric to bind.
        llm (BaseLLM | None): The LLM offered by the run.
        tracker (UsageTracker): Where to record usage.
        cache (CacheBackend | None): Cache to serve repeats from.

    Returns:
        BaseMetric: The metric, bound to a tracked model if it uses one.
    """
    if not isinstance(metric, LLMMetric):
        return metric

    inner = metric.llm or llm or default_llm()
    if cache is not None:
        inner = CachedLLM(inner=inner, backend=cache)
    return metric.model_copy(
        update={"llm": TrackedLLM(inner=inner, tracker=tracker)}
    )


def _score_repeatedly(
    metric: BaseMetric, node: DataNode, config: RunConfig
) -> MetricResult:
    """Score a row `repetitions` times and reduce.

    The spread across repetitions lands in `metadata`, so the variance
    of the judge itself is visible rather than hidden behind a single
    sample.

    Args:
        metric (BaseMetric): The metric to apply.
        node (DataNode): The row to score.
        config (RunConfig): The run policy.

    Returns:
        MetricResult: The reduced result.
    """
    if config.repetitions == 1:
        return metric.score(node)

    runs = [metric.score(node) for _ in range(config.repetitions)]
    scores = [item.score for item in runs if item.score is not None]

    if not scores:
        return runs[0]

    reducers = {"mean": fmean, "median": median, "mode": mode}
    return runs[0].model_copy(
        update={
            "score": reducers[config.reducer](scores),
            "error": None,
            "metadata": {
                **runs[0].metadata,
                "repetitions": scores,
                "repetition_spread": (
                    stdev(scores) if len(scores) > 1 else 0.0
                ),
            },
        }
    )


def _score_one(
    metric: BaseMetric, node: DataNode, config: RunConfig
) -> MetricResult:
    """Score one row with one metric, retrying transport failures.

    A metric that keeps failing yields a result carrying the error
    rather than propagating it, unless `raise_on_error` is set.

    Args:
        metric (BaseMetric): The metric to apply.
        node (DataNode): The row to score.
        config (RunConfig): The run policy.

    Returns:
        MetricResult: The result, possibly carrying an error.
    """
    started = perf_counter()
    delay = config.backoff
    last: Exception = RuntimeError("the metric never ran")

    for attempt in range(config.max_retries + 1):
        try:
            return _score_repeatedly(metric, node, config)
        except Exception as error:  # noqa: BLE001
            if config.raise_on_error:
                raise
            last = error
            logger.warning(
                "%s failed on attempt %d/%d: %s",
                metric.name,
                attempt + 1,
                config.max_retries + 1,
                error,
            )
            if attempt < config.max_retries and delay:
                sleep(delay)
                delay *= 2

    return MetricResult(
        datanode=node,
        metric=metric,
        score=None,
        error=f"{type(last).__name__}: {last}",
        process_time=perf_counter() - started,
    )


def run_metrics(
    dataset: Dataset,
    metrics: list[BaseMetric],
    *,
    llm: BaseLLM | None = None,
    config: RunConfig | None = None,
) -> tuple[list[list[MetricResult]], TokenUsage]:
    """Score every row of a dataset with every metric.

    Args:
        dataset (Dataset): The data to evaluate.
        metrics (list[BaseMetric]): The metrics to apply.
        llm (BaseLLM | None): The language model to lend to any metric
            that does not carry its own.
        config (RunConfig | None): The run policy.

    Returns:
        tuple[list[list[MetricResult]], TokenUsage]: Results indexed by
            metric then row, and the tokens the run consumed.
    """
    config = config or RunConfig()
    validate_dataset(dataset, metrics)

    tracker = UsageTracker()
    backend = _resolve_cache(config.cache)
    bound = [
        _with_tracking(metric, llm, tracker, backend)
        for metric in metrics
    ]
    nodes = list(dataset)
    jobs = [
        (metric_index, row_index)
        for metric_index in range(len(bound))
        for row_index in range(len(nodes))
    ]

    results: list[list[MetricResult | None]] = [
        [None] * len(nodes) for _ in bound
    ]

    def work(job: tuple[int, int]) -> Job:
        metric_index, row_index = job
        result = _score_one(
            bound[metric_index], nodes[row_index], config
        )
        return metric_index, row_index, result

    def collect(completed: Iterable[Job]) -> None:
        for metric_index, row_index, result in _with_progress(
            completed, total=len(jobs), enabled=config.show_progress
        ):
            results[metric_index][row_index] = result

    if config.max_workers == 1:
        collect(map(work, jobs))
    else:
        with ThreadPoolExecutor(
            max_workers=config.max_workers
        ) as pool:
            collect(pool.map(work, jobs))

    return results, tracker.usage()  # type: ignore[return-value]


def _with_progress(
    iterable: Iterable[Job], *, total: int, enabled: bool
) -> Iterator[Job]:
    """Wrap an iterable in a progress bar unless it is switched off."""
    if not enabled:
        return iter(iterable)

    return tqdm(
        iterable,
        total=total,
        ncols=100,
        desc="Evaluating ",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt}   "
            "remain: {remaining}s, {rate_fmt}"
        ),
        colour="green",
        leave=True,
    )
