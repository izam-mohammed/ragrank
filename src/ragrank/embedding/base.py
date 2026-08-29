"""The embedding interface.

Token overlap says "Paris" and "The capital is Paris" barely match. An
LLM judge says they match but costs a call. Embeddings sit between the
two: cheaper and faster than a judge, far more forgiving than string
comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
from math import sqrt

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field


def cosine_similarity(
    left: list[float], right: list[float]
) -> float:
    """Cosine similarity of two vectors, clamped to [0, 1].

    Negative similarity is clamped rather than returned, so the result
    can be used directly as a score.

    Args:
        left (list[float]): The first vector.
        right (list[float]): The second vector.

    Returns:
        float: The similarity, between 0 and 1.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(left) != len(right):
        raise ValueError(
            f"Cannot compare vectors of length {len(left)} and "
            f"{len(right)}."
        )

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


class BaseEmbedding(BaseModel, ABC):
    """Abstract base for an embedding model.

    Implement `embed_text`; batching comes for free but can be
    overridden where a provider supports it natively.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    @property
    @abstractmethod
    def name(self) -> str:
        """The embedding model's name.

        Returns:
            str: The name.
        """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The vector.
        """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed several strings.

        Args:
            texts (list[str]): The texts to embed.

        Returns:
            list[list[float]]: One vector per text.
        """
        return [self.embed_text(text) for text in texts]

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity between two strings.

        Args:
            left (str): The first string.
            right (str): The second string.

        Returns:
            float: The similarity, between 0 and 1.
        """
        first, second = self.embed([left, right])
        return cosine_similarity(first, second)

    def __repr__(self) -> str:
        """Readable name.

        Returns:
            str: The model's name.
        """
        return self.name


class FakeEmbedding(BaseEmbedding):
    """A deterministic, credential-free embedding model.

    Vectors are derived from a hash of the text, so identical strings
    embed identically and different strings do not. That is enough to
    test the wiring end to end without a provider; it carries no real
    semantics, so do not read meaning into its scores.

    Attributes:
        dimensions (int): Length of the produced vectors.
    """

    dimensions: int = Field(
        default=8, ge=1, description="Length of the vectors."
    )

    @property
    def name(self) -> str:
        """The model's name.

        Returns:
            str: The name.
        """
        return "Fake Embedding"

    def embed_text(self, text: str) -> list[float]:
        """Derive a stable pseudo-vector from the text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The vector.
        """
        digest = sha256(text.strip().casefold().encode()).digest()
        return [
            digest[index % len(digest)] / 255.0
            for index in range(self.dimensions)
        ]
