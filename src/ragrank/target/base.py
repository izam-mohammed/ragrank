"""Running the system under test, rather than only scoring its output.

Every other part of ragrank takes a dataset that already contains
responses, which quietly assumes somebody else ran the pipeline and
saved the results. That is fine for a one-off, and wrong for the thing
people actually want: hold a set of questions fixed, change the
pipeline, and see what moved.

A `Target` closes that gap. It is any callable that takes a question and
returns what your RAG system produced for it -- so the question set
becomes the fixture and the pipeline becomes the variable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import Any, Union

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field
from ragrank.dataset import Dataset
from ragrank.exceptions import RagRankError

logger = logging.getLogger(__name__)


class TargetError(RagRankError):
    """The system under test failed to answer a question."""


class TargetOutput(BaseModel):
    """What a target produced for one question.

    Attributes:
        response (str): The answer the system generated.
        context (list[str]): The chunks it retrieved, in rank order.
            Empty is allowed -- some systems do not expose retrieval --
            but the context-based metrics need it.
        retrieved_ids (list[str] | None): Ids of the retrieved chunks,
            which unlock the ranking metrics at no extra cost.
    """

    model_config: ConfigDict = ConfigDict(frozen=True)

    response: str = Field(
        description="The answer the system generated."
    )
    context: list[str] = Field(
        default_factory=list,
        description="The chunks the system retrieved, in rank order.",
    )
    retrieved_ids: list[str] | None = Field(
        default=None,
        description="Ids of the retrieved chunks, in rank order.",
    )


#: Anything a target may hand back for one question.
TargetReturn = Union[  # noqa: UP007
    str,
    TargetOutput,
    dict,
    tuple,
]

#: A system under test: question in, what it produced out.
Target = Callable[[str], TargetReturn]


def normalise_output(value: TargetReturn) -> TargetOutput:
    """Accept the shapes a RAG function naturally returns.

    A bare string is an answer with no visible retrieval. A two-tuple is
    read as `(response, context)`, which is what a small RAG helper
    usually returns. A mapping is read by key.

    Args:
        value (TargetReturn): Whatever the target returned.

    Returns:
        TargetOutput: The normalised output.

    Raises:
        TargetError: If the value is not a shape we can read.
    """
    if isinstance(value, TargetOutput):
        return value
    if isinstance(value, str):
        return TargetOutput(response=value)
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TargetError(
                "A target returning a tuple should return "
                f"(response, context), but this one had {len(value)} "
                "items."
            )
        response, context = value
        return TargetOutput(
            response=response, context=_as_chunks(context)
        )
    if isinstance(value, dict):
        if "response" not in value:
            raise TargetError(
                "A target returning a mapping needs a 'response' key; "
                f"this one had {sorted(value)}."
            )
        return TargetOutput(
            response=value["response"],
            context=_as_chunks(value.get("context", [])),
            retrieved_ids=value.get("retrieved_ids"),
        )
    raise TargetError(
        f"A target should return a string, a tuple, a mapping or a "
        f"TargetOutput, not {type(value).__name__}."
    )


def _as_chunks(value: Any) -> list[str]:  # noqa: ANN401
    """Coerce a context value into a list of chunks.

    Args:
        value (Any): A string, or an iterable of strings.

    Returns:
        list[str]: The chunks.
    """
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def run_target(
    questions: Sequence[str] | Dataset,
    target: Target,
    *,
    references: Sequence[str] | None = None,
    max_workers: int = 4,
    max_retries: int = 0,
    backoff: float = 0.5,
    skip_failures: bool = False,
) -> Dataset:
    """Ask the system under test every question, and collect a dataset.

    Passing a `Dataset` reuses its questions and its references and
    discards its responses, which is the regression shape: same
    questions, new pipeline, comparable numbers.

    Unlike scoring, a failure here is not partial data worth keeping --
    a question with no answer cannot be scored at all -- so this raises
    by default rather than quietly shrinking your evaluation set. Pass
    `skip_failures=True` when a long generation run should survive a
    few bad rows.

    Args:
        questions (Sequence[str] | Dataset): The questions to ask, or a
            dataset to take them from.
        target (Target): The system under test.
        references (Sequence[str] | None): Ground truth answers, kept
            alongside the generated responses.
        max_workers (int): Questions to ask concurrently.
        max_retries (int): Retries after the target raises.
        backoff (float): Seconds before the first retry, doubling.
        skip_failures (bool): Drop questions the target could not
            answer instead of raising.

    Returns:
        Dataset: The generated dataset, ready to evaluate.

    Raises:
        ValueError: If there are no questions, or references do not
            line up with them.
        TargetError: If the target failed and `skip_failures` is off.

    Examples::

        from ragrank import evaluate
        from ragrank.target import run_target


        def my_rag(question: str) -> tuple[str, list[str]]:
            chunks = retriever.search(question)
            return generator(question, chunks), chunks


        dataset = run_target(["who wrote it?"], my_rag)
        result = evaluate(dataset)
    """
    if isinstance(questions, Dataset):
        references = references or questions.reference
        questions = list(questions.question)
    else:
        questions = list(questions)

    if not questions:
        raise ValueError("There are no questions to ask.")

    if references is not None and len(references) != len(questions):
        raise ValueError(
            f"Got {len(references)} references for "
            f"{len(questions)} questions; they must line up."
        )

    def ask(question: str) -> TargetOutput | None:
        delay = backoff
        last: Exception = RuntimeError("the target never ran")
        for attempt in range(max_retries + 1):
            try:
                return normalise_output(target(question))
            except Exception as error:  # noqa: BLE001
                last = error
                logger.warning(
                    "target failed on attempt %d/%d for %r: %s",
                    attempt + 1,
                    max_retries + 1,
                    question,
                    error,
                )
                if attempt < max_retries and delay:
                    sleep(delay)
                    delay *= 2
        if skip_failures:
            return None
        raise TargetError(
            f"The target failed on {question!r}: "
            f"{type(last).__name__}: {last}"
        ) from last

    if max_workers == 1:
        outputs = [ask(question) for question in questions]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outputs = list(pool.map(ask, questions))

    kept = [
        (index, item)
        for index, item in enumerate(outputs)
        if item is not None
    ]
    if not kept:
        raise TargetError(
            "The target failed on every question, so there is nothing "
            "to evaluate."
        )

    dropped = len(outputs) - len(kept)
    if dropped:
        logger.warning(
            "dropped %d question(s) the target could not answer",
            dropped,
        )

    fields: dict[str, Any] = {
        "question": [questions[index] for index, _ in kept],
        "context": [item.context for _, item in kept],
        "response": [item.response for _, item in kept],
    }
    if references is not None:
        fields["reference"] = [
            references[index] for index, _ in kept
        ]
    if all(item.retrieved_ids is not None for _, item in kept):
        fields["retrieved_ids"] = [
            item.retrieved_ids for _, item in kept
        ]

    return Dataset(**fields)
