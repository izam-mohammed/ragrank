"""Lazy access to optional third party dependencies.

Extras cover provider and framework SDKs only -- `openai`, `datasets`,
LangChain, LlamaIndex. Anything the core API touches is a real
dependency, so that `to_dataframe()` and a progress bar just work.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import an optional dependency, or explain how to install it.

    Args:
        module (str): The importable module name, e.g. ``"datasets"``.
        extra (str): The ragrank extra that provides it, e.g. ``"hf"``.

    Returns:
        ModuleType: The imported module.

    Raises:
        ModuleNotFoundError: If the module is not installed.
    """
    try:
        return import_module(module)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"`{module}` is needed for this feature but is not installed. "
            f'Install it with `pip install "ragrank[{extra}]"`.'
        ) from None
