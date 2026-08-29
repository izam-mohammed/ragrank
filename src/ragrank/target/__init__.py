"""Running the system under test.

Examples::

    from ragrank import evaluate
    from ragrank.target import run_target

    dataset = run_target(questions, my_rag_pipeline)
    result = evaluate(dataset, metrics=[faithfulness])
"""

from ragrank.target.base import (
    Target,
    TargetError,
    TargetOutput,
    normalise_output,
    run_target,
)

__all__ = [
    "Target",
    "TargetOutput",
    "TargetError",
    "run_target",
    "normalise_output",
]
