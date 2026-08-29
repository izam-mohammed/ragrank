"""Include the common util functions"""

from __future__ import annotations

from ast import literal_eval


def eval_cell(cell_value: str | list[str]) -> str | list[str]:
    """
    Evaluate a cell value and return it as a string or a list of strings.

    Cells that hold a Python/JSON style list literal are parsed into a real
    list. Anything else is returned unchanged.

    Args:
        cell_value (str | list[str]): The value of the cell.

    Returns:
        Union[str, List[str]]: The evaluated cell value.
    """
    if not isinstance(cell_value, str):
        return cell_value

    stripped = cell_value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return cell_value

    try:
        parsed = literal_eval(stripped)
    except (ValueError, SyntaxError):
        return cell_value

    if not isinstance(parsed, list):
        return cell_value
    return [
        item if isinstance(item, str) else str(item)
        for item in parsed
    ]
