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
from ragrank.metric._heuristic.text import (
    ExactMatch,
    JsonValid,
    LevenshteinRatio,
    RougeL,
    StringPresence,
    TokenF1,
    exact_match,
    json_valid,
    levenshtein_ratio,
    rouge_l,
    string_presence,
    token_f1,
)
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
    ChunkwiseLLMMetric,
    DeterministicMetric,
    LLMMetric,
    MetricResult,
    MetricType,
)

#: The three metrics that between them localise most RAG failures:
#: bad retrieval, unused context, and an answer that misses the point.
RAG_TRIAD = [
    context_relevancy,
    context_utilization,
    response_relevancy,
]

__all__ = [
    # base classes
    "BaseMetric",
    "LLMMetric",
    "ChunkwiseLLMMetric",
    "DeterministicMetric",
    "MetricResult",
    "MetricType",
    # llm judged
    "ResponseRelevancy",
    "ResponseConciseness",
    "ContextRelevancy",
    "ContextUtilization",
    "response_relevancy",
    "response_conciseness",
    "context_relevancy",
    "context_utilization",
    # heuristic, no llm
    "ExactMatch",
    "StringPresence",
    "LevenshteinRatio",
    "TokenF1",
    "RougeL",
    "JsonValid",
    "exact_match",
    "string_presence",
    "levenshtein_ratio",
    "token_f1",
    "rouge_l",
    "json_valid",
    # retrieval, no llm
    # custom
    "CustomMetric",
    "CustomInstruct",
    "InstructConfig",
    # presets
    "RAG_TRIAD",
]
