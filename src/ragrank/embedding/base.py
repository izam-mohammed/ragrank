from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

Embedding = list[float]


class SimilarityMode(str, Enum):
    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def mean_agg(embeddings: list[Embedding]) -> Embedding:
    return list(np.array(embeddings).mean(axis=0))


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float: ...


class RagrankBaseEmbegging(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> Embedding: ...

    def embed_dataset(self, dataset): ...

    def similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
    ) -> float: ...

    async def aembed_text(self, text: str) -> Embedding: ...

    async def aembed_dataset(self, dataset): ...

    async def asimilarity(
        self,
        embedding1,
        embedding2,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
    ): ...


def embedding_factory(): ...
