"""Include the common util functions"""

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
        url=url+url_sufix,
        json=json,
        timeout=timeout,
    )
    return response.ok
