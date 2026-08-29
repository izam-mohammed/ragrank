"""Ranking metrics over retrieved document ids.

These are the standard information retrieval measures, computed from
`retrieved_ids` against `reference_ids`. They cost nothing, are exactly
reproducible, and answer the first question worth asking about a RAG
system: did the retriever find the right documents at all?

If it did not, no amount of judging the generated answer will tell you
why -- and `hit_rate=0.31` is a far sharper signal than a language
model's opinion of your context.
"""

from __future__ import annotations

from math import log2

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.metric.base import DeterministicMetric, MetricType


class RankingMetric(DeterministicMetric):
    """Base for metrics over retrieved and expected document ids.

    Attributes:
        k (int | None): Consider only the top k retrieved ids. None
            uses the whole list.
    """

    k: int | None = Field(
        default=None,
        ge=1,
        description="Consider only the top k retrieved ids.",
    )

    @property
    def required_columns(self) -> set[str]:
        """Fields this metric needs.

        Returns:
            set[str]: `retrieved_ids` and `reference_ids`.
        """
        return {"retrieved_ids", "reference_ids"}

    def _at_k(self, retrieved: list[str]) -> list[str]:
        """Truncate the retrieved list to k.

        Args:
            retrieved (list[str]): The retrieved ids, in rank order.

        Returns:
            list[str]: The first k ids.
        """
        return retrieved if self.k is None else retrieved[: self.k]

    def _suffix(self) -> str:
        """The '@k' suffix for this metric's name.

        Returns:
            str: '@k' or the empty string.
        """
        return f"@{self.k}" if self.k is not None else ""

    def compute(self, data: DataNode) -> float | None:
        """Compute the metric, after checking the row has ids.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: The score, or None if ids are missing or
                there is nothing relevant to find.
        """
        if data.retrieved_ids is None or data.reference_ids is None:
            return None
        relevant = set(data.reference_ids)
        if not relevant:
            return None
        return self.rank(self._at_k(data.retrieved_ids), relevant)

    def rank(
        self, retrieved: list[str], relevant: set[str]
    ) -> float | None:
        """Score one row's ranking.

        Args:
            retrieved (list[str]): Retrieved ids, truncated to k.
            relevant (set[str]): The ids that should be retrieved.

        Returns:
            float | None: The score.
        """
        raise NotImplementedError


class HitRate(RankingMetric):
    """1.0 when at least one relevant document was retrieved."""

    metric_type: MetricType = Field(default=MetricType.BINARY)

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return f"Hit Rate{self._suffix()}"

    def rank(
        self, retrieved: list[str], relevant: set[str]
    ) -> float | None:
        """Check for any relevant id in the retrieved list.

        Args:
            retrieved (list[str]): Retrieved ids.
            relevant (set[str]): Expected ids.

        Returns:
            float | None: 1.0 or 0.0.
        """
        return float(any(item in relevant for item in retrieved))


class MRR(RankingMetric):
    """Reciprocal rank of the first relevant document."""

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return f"MRR{self._suffix()}"

    def rank(
        self, retrieved: list[str], relevant: set[str]
    ) -> float | None:
        """Compute 1 / rank of the first hit.

        Args:
            retrieved (list[str]): Retrieved ids.
            relevant (set[str]): Expected ids.

        Returns:
            float | None: The reciprocal rank, or 0.0 for no hit.
        """
        for position, item in enumerate(retrieved, start=1):
            if item in relevant:
                return 1.0 / position
        return 0.0


class PrecisionAtK(RankingMetric):
    """Fraction of retrieved documents that are relevant."""

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return f"Precision{self._suffix()}"

    def rank(
        self, retrieved: list[str], relevant: set[str]
    ) -> float | None:
        """Compute precision.

        Args:
            retrieved (list[str]): Retrieved ids.
            relevant (set[str]): Expected ids.

        Returns:
            float | None: The precision, or None if nothing retrieved.
        """
        if not retrieved:
            return None
        hits = sum(1 for item in retrieved if item in relevant)
        return hits / len(retrieved)


class RecallAtK(RankingMetric):
    """Fraction of relevant documents that were retrieved."""

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return f"Recall{self._suffix()}"

    def rank(
        self, retrieved: list[str], relevant: set[str]
    ) -> float | None:
        """Compute recall.

        Args:
            retrieved (list[str]): Retrieved ids.
            relevant (set[str]): Expected ids.

        Returns:
            float | None: The recall.
        """
        found = {item for item in retrieved if item in relevant}
        return len(found) / len(relevant)


class MAP(RankingMetric):
    """Average precision -- precision at each hit, averaged."""

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return f"MAP{self._suffix()}"

    def rank(
        self, retrieved: list[str], relevant: set[str]
    ) -> float | None:
        """Compute average precision.

        Args:
            retrieved (list[str]): Retrieved ids.
            relevant (set[str]): Expected ids.

        Returns:
            float | None: The average precision.
        """
        hits = 0
        total = 0.0
        for position, item in enumerate(retrieved, start=1):
            if item in relevant:
                hits += 1
                total += hits / position
        return total / min(len(relevant), len(retrieved) or 1)


class NDCG(RankingMetric):
    """Normalised discounted cumulative gain, binary relevance."""

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return f"NDCG{self._suffix()}"

    def rank(
        self, retrieved: list[str], relevant: set[str]
    ) -> float | None:
        """Compute NDCG against the ideal ranking.

        Args:
            retrieved (list[str]): Retrieved ids.
            relevant (set[str]): Expected ids.

        Returns:
            float | None: The NDCG.
        """
        gain = sum(
            1.0 / log2(position + 1)
            for position, item in enumerate(retrieved, start=1)
            if item in relevant
        )
        ideal_hits = min(len(relevant), len(retrieved))
        ideal = sum(
            1.0 / log2(position + 1)
            for position in range(1, ideal_hits + 1)
        )
        return gain / ideal if ideal else None


hit_rate = HitRate()
mrr = MRR()
precision_at_k = PrecisionAtK()
recall_at_k = RecallAtK()
mean_average_precision = MAP()
ndcg = NDCG()
