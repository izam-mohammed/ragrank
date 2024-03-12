"""utils for the llm module"""

import os


def get_env_var(var_name: str) -> str:
    """Get the environement variables"""

    if var_name not in os.environ:
        raise ValueError(f"{var_name} not found in the environmen.")
    return os.environ[var_name]
