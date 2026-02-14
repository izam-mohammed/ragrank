"""Include the common util functions"""

from __future__ import annotations


def eval_cell(cell_value: str | list[str]) -> str | list[str]:
    """
    Evaluate a cell value and return it as a string or a list of strings.

    Args:
        cell_value (str): The value of the cell.

    Returns:
        Union[str, List[str]]: The evaluated cell value.
    """
    if isinstance(cell_value, list):
        return cell_value
    if cell_value.startswith("[") and cell_value.endswith("]"):
        return cell_value[2:-2].split("', '")
    return cell_value
