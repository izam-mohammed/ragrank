"""Contain all of the base classes for dataset"""

from abc import ABC, abstractmethod


class RagrankBaseDataset(ABC):
    """The base Dataset class for Ragrank"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def upload(self, data: dict) -> None:
        """Abstact function for upload the dataset"""
