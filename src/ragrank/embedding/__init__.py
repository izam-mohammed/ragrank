"""Text embeddings, for metrics that compare meaning rather than words."""

from ragrank.embedding.base import (
    BaseEmbedding,
    FakeEmbedding,
    cosine_similarity,
)

__all__ = ["BaseEmbedding", "FakeEmbedding", "cosine_similarity"]
