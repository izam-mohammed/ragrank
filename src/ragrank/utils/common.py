"""Include the common util functions"""

from __future__ import annotations

import requests

from ragrank.constants import REQUEST_TIME_OUT, SERVER_URL


def send_request(
    *,
    json: dict,
    url: str = SERVER_URL,
    timeout: float = REQUEST_TIME_OUT,
    url_sufix: str = "",
) -> bool:
    """
    Send a POST request to the specified URL with JSON payload.

    Args:
        url (str): The URL to send the request to.
        json (dict): The JSON payload to send.
        timeout (float, optional): Timeout for the request in seconds.
            Defaults to REQUEST_TIME_OUT.

    Returns:
        bool: True if the request was successful, False otherwise.
    """
    response = requests.post(
        url=url + url_sufix,
        json=json,
        timeout=timeout,
    )
    return response.ok


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
