"""Comparing two evaluation runs.

Nobody's workflow is a single run. The question is always "is this
better than what I had?", and answering it by squinting at two numbers
misses two things: whether the difference is larger than the noise, and
which rows actually moved.
"""

from __future__ import annotations

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field
from ragrank.evaluation.outputs import EvalResult


class MetricDelta(BaseModel):
    """How one metric moved between two runs.

    Attributes:
        name (str): The metric's name.
        before (float | None): Aggregate in the baseline run.
        after (float | None): Aggregate in the candidate run.
        delta (float | None): after - before.
        stderr (float | None): Combined standard error of the two runs.
        significant (bool | None): Whether the change is larger than
            roughly two combined standard errors. None when either run
            had too few scores to estimate spread.
        improved_rows (list[int]): Row indexes that scored higher.
        regressed_rows (list[int]): Row indexes that scored lower.
    """

    model_config: ConfigDict = ConfigDict(frozen=True)

    name: str
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    stderr: float | None = None
    significant: bool | None = None
    improved_rows: list[int] = Field(default_factory=list)
    regressed_rows: list[int] = Field(default_factory=list)

    def __repr__(self) -> str:
        """Readable one-line summary.

        Returns:
            str: The summary.
        """
        if self.delta is None:
            return f"{self.name}: n/a"

        arrow = "+" if self.delta >= 0 else ""
        text = (
            f"{self.name}: {self.before:.3f} -> {self.after:.3f} "
            f"({arrow}{self.delta:.3f})"
        )
        if self.significant is False:
            text += " [within noise]"
        elif self.significant is True:
            text += " [significant]"
        if self.regressed_rows:
            text += f", {len(self.regressed_rows)} rows regressed"
        return text

    def __str__(self) -> str:
        """Readable one-line summary.

        Returns:
            str: The summary.
        """
        return self.__repr__()


class Comparison(BaseModel):
    """The difference between a baseline and a candidate run.

    Attributes:
        deltas (list[MetricDelta]): One entry per shared metric.
    """

    model_config: ConfigDict = ConfigDict(frozen=True)

    deltas: list[MetricDelta] = Field(default_factory=list)

    @property
    def regressed(self) -> list[MetricDelta]:
        """Metrics that got significantly worse.

        Returns:
            list[MetricDelta]: The significant regressions.
        """
        return [
            item
            for item in self.deltas
            if item.delta is not None
            and item.delta < 0
            and item.significant
        ]

    @property
    def improved(self) -> list[MetricDelta]:
        """Metrics that got significantly better.

        Returns:
            list[MetricDelta]: The significant improvements.
        """
        return [
            item
            for item in self.deltas
            if item.delta is not None
            and item.delta > 0
            and item.significant
        ]

    def __bool__(self) -> bool:
        """Whether anything changed significantly.

        Returns:
            bool: True if any metric moved beyond the noise.
        """
        return bool(self.regressed or self.improved)

    def __repr__(self) -> str:
        """Readable summary, one line per metric.

        Returns:
            str: The summary.
        """
        if not self.deltas:
            return "Comparison(no shared metrics)"
        return "\n".join(repr(item) for item in self.deltas)

    def __str__(self) -> str:
        """Readable summary, one line per metric.

        Returns:
            str: The summary.
        """
        return self.__repr__()


def compare(before: EvalResult, after: EvalResult) -> Comparison:
    """Diff two runs, metric by metric and row by row.

    Only metrics present in both runs are compared -- adding a metric is
    not a regression in the ones you already had.

    Significance uses the standard errors the two runs already report:
    a change smaller than roughly two combined standard errors is
    flagged as noise rather than presented as an improvement. That is a
    rough test, not a formal one, and it is deliberately conservative.

    Args:
        before (EvalResult): The baseline run.
        after (EvalResult): The candidate run.

    Returns:
        Comparison: The per metric differences.

    Examples::

        baseline = evaluate(data, metrics=metrics)
        candidate = evaluate(data, metrics=metrics, llm=other_model)

        diff = compare(baseline, candidate)
        print(diff)
        assert not diff.regressed
    """
    old = {item.name: item for item in before.summary()}
    new = {item.name: item for item in after.summary()}

    old_scores = dict(
        zip(
            [metric.name for metric in before.metrics],
            before.scores,
            strict=False,
        )
    )
    new_scores = dict(
        zip(
            [metric.name for metric in after.metrics],
            after.scores,
            strict=False,
        )
    )

    deltas = []
    for name in old:
        if name not in new:
            continue

        left, right = old[name], new[name]
        delta = (
            right.value - left.value
            if left.value is not None and right.value is not None
            else None
        )

        combined = None
        if left.stderr is not None and right.stderr is not None:
            combined = (left.stderr**2 + right.stderr**2) ** 0.5

        significant = None
        if delta is not None and combined is not None:
            significant = abs(delta) > 2 * combined

        improved, regressed = _moved_rows(
            old_scores.get(name, []), new_scores.get(name, [])
        )

        deltas.append(
            MetricDelta(
                name=name,
                before=left.value,
                after=right.value,
                delta=delta,
                stderr=combined,
                significant=significant,
                improved_rows=improved,
                regressed_rows=regressed,
            )
        )

    return Comparison(deltas=deltas)


def _moved_rows(
    before: list[float | None], after: list[float | None]
) -> tuple[list[int], list[int]]:
    """Find which row indexes went up and which went down.

    Args:
        before (list[float | None]): Baseline per row scores.
        after (list[float | None]): Candidate per row scores.

    Returns:
        tuple[list[int], list[int]]: Improved and regressed indexes.
    """
    improved, regressed = [], []
    for index, (old, new) in enumerate(
        zip(before, after, strict=False)
    ):
        if old is None or new is None:
            continue
        if new > old:
            improved.append(index)
        elif new < old:
            regressed.append(index)
    return improved, regressed
