from abc import ABC, abstractmethod


class RagrankBaseMetric(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate(self): ...

    async def aevaluate(self): ...
