"""Defining a metric without writing a class.

Three ways in, in rough order of how much you need::

    @metric(name="Has citation")  # any Python you like
    def has_citation(response: str) -> bool:
        return "[" in response


    LLMJudge(name="Tone", instructions=..., rubric={"A": 1.0, "B": 0.0})

    Guidelines(name="Policy", guidelines="Never give medical advice.")
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any

from ragrank.bridge.pydantic import ConfigDict, Field
from ragrank.dataset import DataNode
from ragrank.metric.base import (
    DeterministicMetric,
    LLMMetric,
    MetricType,
)
from ragrank.prompt import Prompt

GUIDELINE_RUBRIC = {"pass": 1.0, "fail": 0.0}

GUIDELINE_INSTRUCTIONS = (
    "You are checking whether a response follows a rule. "
    "The rule is:\n\n{guidelines}\n\n"
    "Reply with exactly one word and nothing else: "
    "'pass' if the response follows the rule, 'fail' if it does not."
)


class FunctionMetric(DeterministicMetric):
    """A metric backed by a plain function.

    The function's parameter names decide what it is given: name a
    parameter after a `DataNode` field and that field is passed in.
    Nothing else is required -- no base class, no registration.

    Attributes:
        func (Callable): The function that computes the score.
        metric_name (str): The metric's display name.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    func: Callable[..., Any] = Field(
        repr=False,
        description="The function that computes the score.",
    )
    metric_name: str = Field(
        description="The metric's display name."
    )
    params: tuple[str, ...] = Field(
        default=(),
        repr=False,
        description="DataNode fields the function asks for.",
    )

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return self.metric_name

    @property
    def required_columns(self) -> set[str]:
        """Fields the wrapped function asks for.

        Returns:
            set[str]: The parameter names.
        """
        return set(self.params)

    def compute(self, data: DataNode) -> float | None:
        """Call the function with the fields it asked for.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: The score, coerced from bool if needed.
        """
        result = self.func(**{
            name: getattr(data, name) for name in self.params
        })
        if result is None:
            return None
        if isinstance(result, bool):
            return float(result)
        return float(result)


def metric(
    _func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    threshold: float | None = None,
    score_range: tuple[float, float] = (0.0, 1.0),
    metric_type: MetricType = MetricType.NON_BINARY,
) -> Any:  # noqa: ANN401
    """Turn a function into a metric.

    Parameters are injected by name from the `DataNode`, so a function
    declares exactly what it needs and gets nothing else. That also
    gives the runner the metric's `required_columns` for free.

    Args:
        _func (Callable | None): The function, when used bare.
        name (str | None): Display name. Defaults to the function name.
        threshold (float | None): Score at or above which a row passes.
        score_range (tuple[float, float]): Valid bounds for the score.
        metric_type (MetricType): Binary or non-binary.

    Returns:
        Any: The metric, or a decorator producing one.

    Examples::

        @metric(name="Has citation", threshold=1.0)
        def has_citation(response: str) -> bool:
            return "[" in response
    """

    def build(func: Callable[..., Any]) -> FunctionMetric:
        known = set(DataNode.model_fields)
        params = tuple(signature(func).parameters)

        unknown = [item for item in params if item not in known]
        if unknown:
            raise ValueError(
                f"{func.__name__}() asks for {unknown}, which are not "
                f"DataNode fields. Available: {sorted(known)}."
            )

        return FunctionMetric(
            func=func,
            params=params,
            metric_name=name
            or func.__name__.replace("_", " ").title(),
            threshold=threshold,
            score_range=score_range,
            metric_type=metric_type,
        )

    return build(_func) if _func is not None else build


class LLMJudge(LLMMetric):
    """A judge defined in one expression.

    The model picks a label from `rubric`; the numbers stay in Python.
    Asking for a letter is markedly more reliable than asking for a
    float, so this is the recommended way to add a judged metric.

    Attributes:
        judge_name (str): The metric's display name.
        instructions (str): What the judge is being asked to decide.
        input_fields (list[str]): DataNode fields to show the judge.

    Examples::

        tone = LLMJudge(
            judge_name="Tone",
            instructions="Is the response's tone right for support?",
            rubric={"A": 1.0, "B": 0.5, "C": 0.0},
        )
    """

    metric_type: MetricType = Field(
        default=MetricType.NON_BINARY,
        description="The type of the metric.",
    )
    judge_name: str = Field(description="The metric's display name.")
    instructions: str = Field(
        repr=False, description="What the judge must decide."
    )
    input_fields: list[str] = Field(
        default_factory=lambda: ["question", "context", "response"],
        repr=False,
        description="DataNode fields to show the judge.",
    )
    rubric: dict[str, float] | None = Field(
        default_factory=lambda: {"A": 1.0, "B": 0.0},
        description="Choice label to score mapping.",
    )
    examples: list[dict[str, Any]] = Field(
        default_factory=list,
        repr=False,
        description="Few shot examples, to calibrate the judge.",
    )

    def model_post_init(self, _context: Any) -> None:  # noqa: ANN401
        """Build the prompt from the configuration."""
        if self.prompt is None:
            choices = ", ".join(self.rubric or {})
            self.prompt = Prompt(
                name=self.judge_name,
                instructions=(
                    f"{self.instructions}\n\nReply with exactly one "
                    f"of [{choices}] and nothing else."
                ),
                examples=self.examples,
                input_keys=self.input_fields,
                output_key="verdict",
            )

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return self.judge_name


class Guidelines(LLMJudge):
    """Pass/fail against a rule written in plain English.

    Covers a surprising amount of what people actually want to measure,
    with no scaffolding at all.

    Examples::

        policy = Guidelines(
            judge_name="No medical advice",
            guidelines="The response must never give medical advice.",
        )
    """

    guidelines: str = Field(
        description="The rule the response must follow."
    )
    instructions: str = Field(
        default="",
        repr=False,
        description="Derived from guidelines.",
    )
    rubric: dict[str, float] | None = Field(
        default_factory=lambda: dict(GUIDELINE_RUBRIC),
        description="pass / fail.",
    )
    metric_type: MetricType = Field(default=MetricType.BINARY)
    threshold: float | None = Field(default=1.0)

    def model_post_init(self, _context: Any) -> None:  # noqa: ANN401
        """Build the guideline prompt."""
        if not self.instructions:
            self.instructions = GUIDELINE_INSTRUCTIONS.format(
                guidelines=self.guidelines
            )
        super().model_post_init(_context)
