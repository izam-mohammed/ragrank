"""Base class for metrics module"""

from ragrank.metric._custom.instruct import (
    CustomInstruct,
    InstructConfig,
)
from ragrank.metric._custom.metric import CustomMetric
from ragrank.metric.base import BaseMetric, MetricResult

__all__ = [
    "BaseMetric",
    "MetricResult",
    "CustomMetric",
    "CustomInstruct",
    "InstructConfig",
]
