"""The main module for ragrank"""

from ragrank.evaluation.base import evaluate
from ragrank.evaluation.outputs import EvalResult

__all__ = [
    "evaluate",
    "EvalResult",
]
