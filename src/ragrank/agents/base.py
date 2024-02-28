from typing import Any


class BaseRagrankAgent:
    def __init__(self) -> None: ...

    def call(self): ...

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call()
