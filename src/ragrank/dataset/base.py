"""Contain all of the base classes for dataset"""

from __future__ import annotations

from typing import List

from ragrank.bridge.pydantic import BaseModel


class Dataset(BaseModel):
    """A dataclass that hold one datapoint"""

    question: str
    context: List[str]
    response: str
