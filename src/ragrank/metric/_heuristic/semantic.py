"""Similarity metrics backed by embeddings."""

from __future__ import annotations

from time import perf_counter

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.embedding import BaseEmbedding
from ragrank.metric.base import DeterministicMetric, MetricResult


class SemanticSimilarity(DeterministicMetric):
    """How close the response is to the reference in meaning.

    Sits between `token_f1`, which cares about the words used, and
    `correctness`, which costs a judge call. Cheaper and faster than a
    judge, far more forgiving than string comparison.

    Attributes:
        embedding (BaseEmbedding | None): The model to embed with. The
            run must supply one; there is no default, because an
            embedding model is a deployment choice.
    """

    embedding: BaseEmbedding | None = Field(
        default=None, description="The embedding model to use."
    )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """
        return "Semantic Similarity"

    @property
    def required_columns(self) -> set[str]:
        """Fields this metric needs.

        Returns:
            set[str]: `response` and `reference`.
        """
        return {"response", "reference"}

    def compute(self, data: DataNode) -> float | None:
        """Cosine similarity between response and reference.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: The similarity, or None without a reference.
        """
        if data.reference is None:
            return None
        return self.embedding.similarity(
            data.response, data.reference
        )

    def score(self, data: DataNode) -> MetricResult:
        """Score, failing clearly when no embedding model was given.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The result of the metric calculation.
        """
        if self.embedding is None:
            return MetricResult(
                datanode=data,
                metric=self,
                score=None,
                error=(
                    "no embedding model was set. Pass "
                    "SemanticSimilarity(embedding=...)."
                ),
                process_time=perf_counter() - perf_counter(),
            )
        return super().score(data)


semantic_similarity = SemanticSimilarity()
