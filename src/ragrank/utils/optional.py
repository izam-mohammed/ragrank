"""Lazy access to optional third party dependencies.

The ragrank core only requires pydantic. Everything else -- pandas, tqdm,
datasets, provider SDKs -- is an optional extra, imported at the point of
use so that installing the core stays cheap.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import an optional dependency, or explain how to install it.

    Args:
        module (str): The importable module name, e.g. ``"pandas"``.
        extra (str): The ragrank extra that provides it, e.g. ``"pandas"``.

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


def is_available(module: str) -> bool:
    """Report whether an optional dependency can be imported.

    Args:
        module (str): The importable module name.

    Returns:
        bool: True if the module imports cleanly.
    """
    try:
        import_module(module)
    except ModuleNotFoundError:
        return False
    return True
