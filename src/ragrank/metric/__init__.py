"""Base class for metrics module"""

from ragrank.metric._context_related.relevancy import (
    ContextRelevancy,
    context_relevancy,
)
from ragrank.metric._context_related.utilization import (
    ContextUtilization,
    context_utilization,
)
from ragrank.metric._custom.instruct import (
    CustomInstruct,
    InstructConfig,
)
from ragrank.metric._custom.metric import CustomMetric
from ragrank.metric._response_related.conciseness import (
    ResponseConciseness,
    response_conciseness,
)
from ragrank.metric._response_related.relevancy import (
    ResponseRelevancy,
    response_relevancy,
)
from ragrank.metric.base import (
    BaseMetric,
    LLMMetric,
    MetricResult,
    MetricType,
)

RAG_TRIAD = [
    context_relevancy,
    context_utilization,
    response_relevancy,
]

__all__ = [
    "BaseMetric",
    "LLMMetric",
    "MetricResult",
    "MetricType",
    "ResponseRelevancy",
    "ResponseConciseness",
    "ContextRelevancy",
    "ContextUtilization",
    "response_relevancy",
    "response_conciseness",
    "context_relevancy",
    "context_utilization",
    "CustomMetric",
    "CustomInstruct",
    "InstructConfig",
    "RAG_TRIAD",
]
