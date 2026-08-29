"""Base module for metric"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from statistics import fmean
from time import perf_counter
from typing import Any

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric.parse import ParsedScore, parse_score
from ragrank.prompt import Prompt

logger = logging.getLogger(__name__)

RETRY_INSTRUCTION = (
    "\n\nYour previous answer could not be read as a score ({error}). "
    "Reply with the score and nothing else."
)


class MetricType(Enum):
    """Enumeration of metric types."""

    BINARY = "binary"
    NON_BINARY = "non_binary"


class BaseMetric(BaseModel, ABC):
    """Base class for defining metrics.

    Attributes:
        metric_type (MetricType): The type of the metric.
        llm (BaseLLM | None): The language model associated with the
            metric. When None the metric uses whichever LLM the
            evaluation run supplies, falling back to `default_llm()`.
        prompt (Prompt): The prompt associated with the metric.
        score_range (tuple[float, float]): Inclusive bounds a score must
            fall within to be considered valid.
        threshold (float | None): Score at or above which a result counts
            as passing. None means the metric never fails.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    metric_type: MetricType = Field(
        description="The type of the metric."
    )
    llm: BaseLLM | None = Field(
        default=None,
        description="The language model associated with the metric.",
    )
    prompt: Prompt | None = Field(
        default=None,
        description=(
            "The prompt associated with the metric. None for metrics "
            "that need no language model."
        ),
    )
    score_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        repr=False,
        description="Inclusive bounds for a valid score.",
    )
    threshold: float | None = Field(
        default=None,
        repr=False,
        description="Score at or above which the result passes.",
    )

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """

    @abstractmethod
    def score(self, data: DataNode) -> MetricResult:
        """Method to compute the metric score.

        Args:
            data (DataNode): The data node for which the score is computed.

        Returns:
            MetricResult: The computed score.
        """

    @property
    def required_columns(self) -> set[str]:
        """Fields of a DataNode this metric needs to produce a score.

        The evaluation runner checks these before spending a single
        token, so a missing column fails in milliseconds rather than
        part way through a paid run.

        Returns:
            set[str]: The required DataNode field names.
        """
        return set(self.prompt.input_keys) if self.prompt else set()

    def resolve_llm(self, llm: BaseLLM | None = None) -> BaseLLM:
        """Decide which language model this metric should use.

        Precedence is the metric's own LLM, then the one supplied by the
        evaluation run, then the library default.

        Args:
            llm (BaseLLM | None): The LLM offered by the caller.

        Returns:
            BaseLLM: The language model to use.
        """
        return self.llm or llm or default_llm()

    def with_llm(self, llm: BaseLLM | None) -> BaseMetric:
        """Return a copy of this metric bound to `llm`.

        A metric that already has an explicit LLM keeps it. Copying
        rather than mutating keeps metrics safe to share across threads.

        Args:
            llm (BaseLLM | None): The LLM to bind.

        Returns:
            BaseMetric: This metric, or a copy bound to `llm`.
        """
        if llm is None or self.llm is not None:
            return self
        return self.model_copy(update={"llm": llm})

    def aggregate(self, scores: list[float]) -> float | None:
        """Reduce per-row scores to a single number.

        Metrics that do not decompose as a mean -- an F1, a pass rate --
        override this.

        Args:
            scores (list[float]): The non-null scores from a run.

        Returns:
            float | None: The aggregate, or None if there is nothing to
                aggregate.
        """
        return fmean(scores) if scores else None

    def __repr__(self) -> str:
        """Representation of the metric

        Returns:
            str: The name of the metric.
        """
        return self.name

    def save(self) -> None:
        """Method to save the metric. Not implemented in base class."""
        raise NotImplementedError

    def load(self) -> None:
        """Method to load the metric. Not implemented in base class."""
        raise NotImplementedError


class LLMMetric(BaseMetric):
    """A metric that scores by prompting a language model.

    Subclasses supply a name and a prompt; the prompting, parsing,
    retrying and timing are handled here once rather than copied into
    every metric.

    Attributes:
        rubric (dict[str, float] | None): Choice label to score mapping.
            When set, the model is asked to pick a label and the numeric
            scale stays in Python, which is markedly more reliable than
            asking for a float.
        max_retries (int): How many times to re-ask after an unparseable
            answer.
    """

    rubric: dict[str, float] | None = Field(
        default=None,
        repr=False,
        description="Choice label to score mapping for the judge.",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        repr=False,
        description="Retries after an unparseable answer.",
    )

    def prompt_values(self, data: DataNode) -> dict[str, Any]:
        """Collect the values this metric's prompt needs.

        Args:
            data (DataNode): The data node being scored.

        Returns:
            dict[str, Any]: Values keyed by prompt input key.
        """
        return data.model_dump()

    def _judge(self, values: dict[str, Any]) -> ParsedScore:
        """Prompt the judge and parse its answer, with retries.

        Args:
            values (dict[str, Any]): Values for the prompt's input keys.

        Returns:
            ParsedScore: The score, or None with an explanation.
        """
        llm = self.resolve_llm()
        base_prompt = self.prompt.render(values)
        prompt_text = base_prompt
        parsed = ParsedScore(None, "the metric never ran")

        for attempt in range(self.max_retries + 1):
            response = llm.generate_text(prompt_text)
            parsed = parse_score(
                response.response,
                score_range=self.score_range,
                rubric=self.rubric,
            )
            if parsed.score is not None:
                return parsed

            logger.warning(
                "%s: unusable answer on attempt %d/%d - %s",
                self.name,
                attempt + 1,
                self.max_retries + 1,
                parsed.error,
            )
            prompt_text = base_prompt + RETRY_INSTRUCTION.format(
                error=parsed.error
            )

        return parsed

    def score(self, data: DataNode) -> MetricResult:
        """Score a single data node.

        An answer the judge gives that cannot be read as a score yields a
        result with `score=None` and a populated `error`, never an
        exception -- one uncooperative row must not destroy a run.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The result of the metric calculation.
        """
        started = perf_counter()
        parsed = self._judge(self.prompt_values(data))
        return MetricResult(
            datanode=data,
            metric=self,
            score=parsed.score,
            error=parsed.error if parsed.score is None else None,
            process_time=perf_counter() - started,
        )

    def reason(
        self, data: DataNode, score: float, raw_response: str
    ) -> str | None:
        """Explain a score. Subclasses may override.

        Args:
            data (DataNode): The data node that was scored.
            score (float): The parsed score.
            raw_response (str): The judge's raw answer.

        Returns:
            str | None: The explanation, if any.
        """
        return None


class DeterministicMetric(BaseMetric):
    """A metric computed in Python, with no language model involved.

    These cost nothing, take microseconds and give the same answer every
    time. Where one will do, it is a strictly better signal than a
    judge: if retrieval is broken, `hit_rate=0.31` tells you more than
    any LLM's opinion of your context, for free.
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of the metric.",
    )
    prompt: Prompt | None = Field(
        default=None,
        description="Unused; kept for the base contract.",
    )

    @abstractmethod
    def compute(self, data: DataNode) -> float | None:
        """Compute the score for one row.

        Returning None means "not applicable to this row" -- an honest
        abstention rather than a fabricated zero.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: The score, or None if not applicable.
        """

    def score(self, data: DataNode) -> MetricResult:
        """Compute the score, timing it and bounds-checking the result.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The result of the metric calculation.
        """
        started = perf_counter()
        value = self.compute(data)

        if value is None:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error="the metric does not apply to this row",
                process_time=perf_counter() - started,
            )

        low, high = self.score_range
        if not low <= value <= high:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error=(
                    f"score {value} is outside the valid range "
                    f"[{low}, {high}]"
                ),
                process_time=perf_counter() - started,
            )

        return MetricResult(
            datanode=data,
            metric=self,
            score=value,
            process_time=perf_counter() - started,
        )


class ChunkwiseLLMMetric(LLMMetric):
    """An LLM metric that judges each retrieved chunk separately.

    Scoring a whole context list in one prompt hides the thing you most
    want to know: one irrelevant chunk among nine good ones is invisible
    in a single averaged verdict. This judges each chunk on its own and
    reduces, keeping the per chunk scores in `metadata`.
    """

    def chunk_values(
        self, data: DataNode, chunk: str
    ) -> dict[str, Any]:
        """Build the prompt values for one chunk.

        Args:
            data (DataNode): The row being scored.
            chunk (str): The single context chunk to judge.

        Returns:
            dict[str, Any]: Values keyed by prompt input key.
        """
        return {**data.model_dump(), "context": chunk}

    def reduce(self, scores: list[float]) -> float:
        """Combine per chunk scores into the row's score.

        Args:
            scores (list[float]): The chunk scores that parsed.

        Returns:
            float: The reduced score.
        """
        return fmean(scores)

    def score(self, data: DataNode) -> MetricResult:
        """Judge every chunk, then reduce.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The reduced score, with per chunk detail in
                `metadata["chunk_scores"]`.
        """
        started = perf_counter()

        if not data.context:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error="no context to judge",
                process_time=perf_counter() - started,
            )

        chunk_scores: list[float | None] = []
        errors: list[str] = []
        for chunk in data.context:
            parsed = self._judge(self.chunk_values(data, chunk))
            chunk_scores.append(parsed.score)
            if parsed.error:
                errors.append(parsed.error)

        valid = [item for item in chunk_scores if item is not None]
        metadata = {"chunk_scores": chunk_scores}

        if not valid:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error=errors[0]
                if errors
                else "no chunk could be scored",
                metadata=metadata,
                process_time=perf_counter() - started,
            )

        return MetricResult(
            datanode=data,
            metric=self,
            score=self.reduce(valid),
            metadata=metadata,
            process_time=perf_counter() - started,
        )


class MetricResult(BaseModel):
    """Class to hold the result of a metric computation.

    Attributes:
        datanode (DataNode): The data node associated with the metric result.
        metric (BaseMetric): Metric used in the computation.
        score (float | None): Score computed for the metric, or None when
            the metric could not produce one.
        reason (str | None): Reason corresponding to the metric score.
        error (str | None): Why no score was produced, when that happened.
        process_time (float | None): Processing time for the computation.
    """

    model_config: ConfigDict = ConfigDict(frozen=True)

    datanode: DataNode = Field(
        description="The data node associated with the metric result.",
    )
    metric: BaseMetric = Field(
        description="List of metrics used in the computation."
    )
    score: float | None = Field(
        default=None,
        description="Score computed for the metric, None if unavailable.",
    )
    reason: str | None = Field(
        default=None,
        description="List of reasons corresponding to each metric score.",
    )
    error: str | None = Field(
        default=None,
        description="Why no score was produced, when that happened.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        repr=False,
        description=(
            "Metric specific detail -- per chunk scores, the judge's "
            "raw answer, intermediate values."
        ),
    )
    process_time: float | None = Field(
        default=None,
        repr=False,
        description="Processing time for the computation.",
    )

    @property
    def passed(self) -> bool | None:
        """Whether this result meets the metric's threshold.

        Returns:
            bool | None: True or False against the threshold, or None if
                the metric has no threshold or produced no score.
        """
        if self.metric.threshold is None or self.score is None:
            return None
        return self.score >= self.metric.threshold
