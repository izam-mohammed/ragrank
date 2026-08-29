"""Checking the judge against human labels.

Every number this library produces rests on one unexamined assumption:
that the judge agrees with a person. Almost nothing in the space
measures that, which is why "why should I trust these scores" is the
first question anyone sensible asks and the last one most eval tools
answer.

Label a few dozen rows by hand, run the judge over the same rows, and
this tells you how closely the two track -- and, more usefully, which
way the judge leans when they disagree. A judge that is generous by a
consistent 0.2 is fixable. One that correlates at 0.1 is not measuring
what you think it is.

Everything here is pure Python. Correlations on thirty rows do not need
a numerical stack, and adding one to the dependency tree to compute a
Pearson coefficient would be its own kind of joke.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field
from ragrank.evaluation.outputs import EvalResult
from ragrank.metric import BaseMetric

#: Below this many pairs, correlation is noise wearing a number.
ADVISABLE_PAIRS = 20


class Alignment(BaseModel):
    """How closely a judge tracks human labels.

    Attributes:
        metric (str): The metric that was checked.
        pairs (int): Rows where both a judge score and a human label
            were available.
        dropped (int): Rows skipped because one side was missing.
        pearson (float | None): Linear correlation. None when there are
            too few pairs, or when either side never varies.
        spearman (float | None): Rank correlation, which survives a
            judge whose scale is compressed but whose ordering is right.
        mean_absolute_error (float | None): Average size of a
            disagreement.
        bias (float | None): Mean signed error. Positive means the
            judge scores higher than the humans do -- a generous judge.
        agreement (float | None): Share of rows where judge and human
            land on the same side of `threshold`. None if no threshold
            was given.
        kappa (float | None): Cohen's kappa for that same split, which
            discounts the agreement you would get by chance. Roughly:
            below 0.4 is poor, above 0.6 is usable.
    """

    model_config: ConfigDict = ConfigDict(frozen=True)

    metric: str = Field(description="The metric that was checked.")
    pairs: int = Field(description="Usable judge/human pairs.")
    dropped: int = Field(
        default=0, description="Pairs skipped for missing values."
    )
    pearson: float | None = Field(
        default=None, description="Linear correlation."
    )
    spearman: float | None = Field(
        default=None, description="Rank correlation."
    )
    mean_absolute_error: float | None = Field(
        default=None, description="Average disagreement size."
    )
    bias: float | None = Field(
        default=None,
        description="Mean signed error; positive is a generous judge.",
    )
    agreement: float | None = Field(
        default=None,
        description="Share of rows agreeing about the threshold.",
    )
    kappa: float | None = Field(
        default=None,
        description="Cohen's kappa for the thresholded split.",
    )

    @property
    def trustworthy(self) -> bool | None:
        """A blunt read on whether the judge is usable.

        True when rank correlation is at least 0.6 over enough pairs to
        mean something. Deliberately crude -- it is a smoke alarm, not
        a certificate.

        Returns:
            bool | None: None when there is not enough evidence.
        """
        if self.spearman is None or self.pairs < ADVISABLE_PAIRS:
            return None
        return self.spearman >= 0.6

    def __repr__(self) -> str:
        """Readable summary of the alignment.

        Returns:
            str: One line per statistic that could be computed.
        """
        parts = [f"{self.metric} vs human labels (n={self.pairs})"]
        if self.dropped:
            parts.append(f"  {self.dropped} pair(s) incomplete")
        for label, value in (
            ("pearson", self.pearson),
            ("spearman", self.spearman),
            ("mean abs error", self.mean_absolute_error),
            ("bias", self.bias),
            ("agreement", self.agreement),
            ("kappa", self.kappa),
        ):
            if value is not None:
                parts.append(f"  {label}: {value:+.3f}")
        if self.pairs < ADVISABLE_PAIRS:
            parts.append(
                f"  warning: {self.pairs} pairs is too few to "
                "conclude much; label more rows."
            )
        return "\n".join(parts)

    def __str__(self) -> str:
        """Readable summary of the alignment.

        Returns:
            str: The same text as `__repr__`.
        """
        return self.__repr__()


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation of two equal-length sequences.

    Args:
        left (Sequence[float]): The first series.
        right (Sequence[float]): The second series.

    Returns:
        float | None: The coefficient, or None when either series is
            constant and the coefficient is therefore undefined.
    """
    if len(left) < 2:
        return None

    mean_left, mean_right = fmean(left), fmean(right)
    delta_left = [item - mean_left for item in left]
    delta_right = [item - mean_right for item in right]

    covariance = sum(
        a * b for a, b in zip(delta_left, delta_right, strict=True)
    )
    spread = (
        sum(item * item for item in delta_left)
        * sum(item * item for item in delta_right)
    ) ** 0.5

    return covariance / spread if spread else None


def rank(values: Sequence[float]) -> list[float]:
    """Rank a sequence, averaging over ties.

    Args:
        values (Sequence[float]): The values to rank.

    Returns:
        list[float]: Ranks, in the order the values were given.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        shared = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1

    return ranks


def spearman(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Spearman rank correlation.

    Args:
        left (Sequence[float]): The first series.
        right (Sequence[float]): The second series.

    Returns:
        float | None: The coefficient, or None if undefined.
    """
    return pearson(rank(left), rank(right))


def cohens_kappa(
    left: Sequence[bool], right: Sequence[bool]
) -> float | None:
    """Cohen's kappa for two binary raters.

    Args:
        left (Sequence[bool]): The first rater's verdicts.
        right (Sequence[bool]): The second rater's verdicts.

    Returns:
        float | None: Kappa, or None when both raters were unanimous
            and chance agreement is already total.
    """
    total = len(left)
    if not total:
        return None

    observed = fmean(
        [
            float(a == b)
            for a, b in zip(left, right, strict=True)
        ]
    )
    positive_left = fmean([float(item) for item in left])
    positive_right = fmean([float(item) for item in right])
    expected = positive_left * positive_right + (
        1 - positive_left
    ) * (1 - positive_right)

    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def _judge_scores(
    judge: EvalResult | Sequence[float | None],
    metric: BaseMetric | str | None,
) -> tuple[list[float | None], str]:
    """Pull the scores to check out of whatever the caller passed.

    Args:
        judge (EvalResult | Sequence[float | None]): A finished run, or
            the scores directly.
        metric (BaseMetric | str | None): Which metric to take from a
            run holding more than one.

    Returns:
        tuple[list[float | None], str]: The scores and a display name.

    Raises:
        ValueError: If the metric is ambiguous or not in the run.
    """
    if not isinstance(judge, EvalResult):
        name = (
            metric
            if isinstance(metric, str)
            else getattr(metric, "name", "judge")
        )
        return list(judge), name

    names = [item.name for item in judge.metrics]
    if metric is None:
        if len(names) != 1:
            raise ValueError(
                "This run has several metrics "
                f"({', '.join(names)}); pass metric= to say which "
                "one the human labels are for."
            )
        return list(judge.scores[0]), names[0]

    wanted = metric if isinstance(metric, str) else metric.name
    if wanted not in names:
        raise ValueError(
            f"{wanted!r} is not in this run. It scored: "
            + ", ".join(names)
        )
    return list(judge.scores[names.index(wanted)]), wanted


def align(
    judge: EvalResult | Sequence[float | None],
    human: Sequence[float | None],
    *,
    metric: BaseMetric | str | None = None,
    threshold: float | None = None,
) -> Alignment:
    """Measure how closely a judge tracks human labels.

    Rows where either side is missing are dropped rather than treated
    as zero, and counted in `dropped` so the omission is visible.

    Args:
        judge (EvalResult | Sequence[float | None]): A finished run, or
            the judge's scores directly.
        human (Sequence[float | None]): The human labels, in the same
            row order.
        metric (BaseMetric | str | None): Which metric to check, for a
            run that scored more than one.
        threshold (float | None): Score at or above which a row counts
            as a pass. Supplying one adds `agreement` and `kappa`,
            which are what you want when the decision the metric feeds
            is itself a pass or a fail.

    Returns:
        Alignment: The comparison.

    Raises:
        ValueError: If the two series are different lengths, or the
            metric is ambiguous.

    Examples::

        from ragrank.evaluation import align

        report = align(result, human_labels, threshold=0.5)
        print(report)
    """
    scores, name = _judge_scores(judge, metric)

    if len(scores) != len(human):
        raise ValueError(
            f"Got {len(scores)} judge scores and {len(human)} human "
            "labels; they must line up row for row."
        )

    usable = [
        (float(a), float(b))
        for a, b in zip(scores, human, strict=True)
        if a is not None and b is not None
    ]
    dropped = len(scores) - len(usable)

    if not usable:
        return Alignment(metric=name, pairs=0, dropped=dropped)

    judged = [pair[0] for pair in usable]
    labelled = [pair[1] for pair in usable]

    agreement = kappa = None
    if threshold is not None:
        judge_pass = [item >= threshold for item in judged]
        human_pass = [item >= threshold for item in labelled]
        agreement = fmean([
            float(a == b)
            for a, b in zip(judge_pass, human_pass, strict=True)
        ])
        kappa = cohens_kappa(judge_pass, human_pass)

    return Alignment(
        metric=name,
        pairs=len(usable),
        dropped=dropped,
        pearson=pearson(judged, labelled),
        spearman=spearman(judged, labelled),
        mean_absolute_error=fmean([
            abs(a - b)
            for a, b in zip(judged, labelled, strict=True)
        ]),
        bias=fmean([
            a - b for a, b in zip(judged, labelled, strict=True)
        ]),
        agreement=agreement,
        kappa=kappa,
    )
