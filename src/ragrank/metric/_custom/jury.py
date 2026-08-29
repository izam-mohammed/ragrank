"""Judging by committee, and by comparison.

Two answers to the same problem -- a single language model's verdict is
noisy and poorly calibrated.

`Jury` asks several judges and combines them, which trades cost for
reliability. `Pairwise` sidesteps absolute scoring entirely: models are
much better at "is A better than B" than at "rate this 0 to 1", and
comparing two systems is usually the real question anyway.
"""

# ruff: noqa: E501
from __future__ import annotations

from statistics import fmean, median
from time import perf_counter
from typing import Literal

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.metric.base import (
    BaseMetric,
    LLMMetric,
    MetricResult,
    MetricType,
)
from ragrank.prompt import Prompt

PAIRWISE_PROMPT = Prompt(
    name="Pairwise Comparison",
    instructions="Two assistants answered the same question. Decide which answer is better, judging on accuracy first and helpfulness second. Ignore which one is longer, and ignore the order they are presented in. Reply with exactly one letter and nothing else.\n(A) Answer A is better.\n(B) Answer B is better.\n(T) They are equally good.",
    input_keys=["question", "answer_a", "answer_b"],
    output_key="verdict",
)

PAIRWISE_RUBRIC = {"A": 1.0, "T": 0.5, "B": 0.0}


class Jury(BaseMetric):
    """Runs several judges over the same row and combines them.

    A committee is more stable than any one member, and the spread
    across members tells you how contested the verdict was. Where the
    judges disagree loudly is usually where your rubric is ambiguous.

    Attributes:
        judges (list[BaseMetric]): The metrics to poll.
        aggregation (str): How to combine them: mean or median.
        jury_name (str): Display name for the combined metric.
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of the metric.",
    )
    judges: list[BaseMetric] = Field(
        description="The metrics to poll."
    )
    aggregation: Literal["mean", "median"] = Field(
        default="median",
        description="How to combine the judges' scores.",
    )
    jury_name: str = Field(
        default="Jury", description="Display name."
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return self.jury_name

    @property
    def required_columns(self) -> set[str]:
        """The union of what every judge needs.

        Returns:
            set[str]: The required DataNode field names.
        """
        needed: set[str] = set()
        for judge in self.judges:
            needed |= judge.required_columns
        return needed

    def with_llm(self, llm: object) -> BaseMetric:
        """Lend the run's LLM to any judge without one of its own.

        Args:
            llm (BaseLLM | None): The LLM offered by the run.

        Returns:
            BaseMetric: A copy with the judges bound.
        """
        if llm is None:
            return self
        return self.model_copy(
            update={
                "judges": [
                    judge.with_llm(llm) for judge in self.judges
                ]
            }
        )

    def score(self, data: DataNode) -> MetricResult:
        """Poll every judge and combine their verdicts.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The combined score, with each judge's verdict
                in `metadata["verdicts"]`.
        """
        started = perf_counter()

        verdicts = [
            {"judge": judge.name, "score": judge.score(data).score}
            for judge in self.judges
        ]
        scored = [
            item["score"]
            for item in verdicts
            if item["score"] is not None
        ]
        metadata = {
            "verdicts": verdicts,
            "disagreement": (
                max(scored) - min(scored) if len(scored) > 1 else 0.0
            ),
        }

        if not scored:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error="no judge produced a usable verdict",
                metadata=metadata,
                process_time=perf_counter() - started,
            )

        combine = fmean if self.aggregation == "mean" else median
        return MetricResult(
            datanode=data,
            metric=self,
            score=combine(scored),
            metadata=metadata,
            process_time=perf_counter() - started,
        )


class Pairwise(LLMMetric):
    """Judges the response against a baseline answer, not a scale.

    Models are markedly better at "which of these is better" than at
    "rate this from 0 to 1", and absolute judge scores drift between
    models and prompt versions in a way relative ones do not.

    Position bias is real -- judges favour whichever answer came first
    -- so each pair is judged twice with the order swapped. A verdict
    that flips is reported as a tie rather than as a win for whichever
    ordering happened to be asked first.

    Attributes:
        baseline_field (str): The DataNode field holding the answer to
            compare against. Defaults to `reference`.
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of the metric.",
    )
    prompt: Prompt = Field(
        default=PAIRWISE_PROMPT,
        description="The prompt used to compare two answers.",
    )
    rubric: dict[str, float] | None = Field(
        default_factory=lambda: dict(PAIRWISE_RUBRIC),
        description="A wins / tie / B wins.",
    )
    baseline_field: str = Field(
        default="reference",
        description="Field holding the answer to compare against.",
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return "Pairwise"

    @property
    def required_columns(self) -> set[str]:
        """Fields this metric needs.

        Returns:
            set[str]: question, response and the baseline field.
        """
        return {"question", "response", self.baseline_field}

    def score(self, data: DataNode) -> MetricResult:
        """Compare the response to the baseline, both ways round.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: 1.0 if the response wins both orderings, 0.0
                if it loses both, 0.5 for a tie or a disagreement.
        """
        started = perf_counter()

        baseline = getattr(data, self.baseline_field, None)
        if not baseline:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error=f"no {self.baseline_field} to compare against",
                process_time=perf_counter() - started,
            )

        forward = self._judge({
            "question": data.question,
            "answer_a": data.response,
            "answer_b": baseline,
        }).score
        reverse = self._judge({
            "question": data.question,
            "answer_a": baseline,
            "answer_b": data.response,
        }).score

        if forward is None or reverse is None:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error="the judge gave no usable verdict",
                process_time=perf_counter() - started,
            )

        # reverse is scored from the baseline's point of view
        flipped = 1.0 - reverse
        consistent = forward == flipped
        metadata = {
            "forward": forward,
            "reverse": flipped,
            "position_bias": not consistent,
        }

        return MetricResult(
            datanode=data,
            metric=self,
            score=forward if consistent else 0.5,
            metadata=metadata,
            process_time=perf_counter() - started,
        )
