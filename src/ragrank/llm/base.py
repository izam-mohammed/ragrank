from abc import ABC, abstractmethod


class RagrankBaseLLM(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def generate_text(self, text: str): ...

    @abstractmethod
    async def agenerate_text(self, texts): ...

    @abstractmethod
    def batch(self, batch): ...

    @abstractmethod
    async def abatch(self, batch): ...

    @abstractmethod
    def stream(self, text): ...

    @abstractmethod
    async def stream(self, text): ...

    def get_token_ids(self): ...

    def get_num_tokens(self): ...
