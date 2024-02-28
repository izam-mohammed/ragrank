from abc import ABC, abstractmethod


class RagrankBaseDataset(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def upload(self, data): ...
