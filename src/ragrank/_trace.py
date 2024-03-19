"""Private module for tracking"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, TypeVar

from ragrank.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    validate_call,
)
from ragrank.constants import DEBUG_MODE, SERVER_URL
from ragrank.utils.common import send_request

T = TypeVar("T", bound="BaseEvent")


class BaseEvent(BaseModel):
    """
    Base class for tracking events.

    Attributes:
        model_config (ConfigDict): Configuration dictionary.
        name (str): Name of the event.
        time (datetime): Time of the event.
        time_cost (float): Time taken for the event.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, frozen=True
    )

    name: str = Field(
        max_length=20,
        frozen=True,
        exclude=True,
        description="Name of the event.",
    )
    time: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        repr=False,
        description="Time of the event.",
    )
    time_cost: float = Field(
        ge=0, repr=False, description="Time taken for the event."
    )


class EvaluationEvent(BaseEvent):
    """
    Event class for evaluation.

    Attributes:
        llm (str): The model used in the evaluation.
        metrics (Dict[str, Optional[float]]): Metrics and their averages.
        data_size (int): Size of the data.
    """

    name: str = Field(
        default="Evaluation",
        frozen=True,
        exclude=True,
        description="Name of the event.",
    )
    llm: str = Field(
        max_length=100,
        repr=False,
        description="The model used in the evaluation.",
    )
    metrics: Dict[str, Optional[float]] = Field(
        repr=False, description="Metrics and their averages."
    )
    data_size: int = Field(
        gt=0, repr=False, description="Size of the data."
    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def trace(
    event: BaseEvent,
) -> bool:
    """
    Trace function to send event data to server.

    Args:
        event (BaseEvent): The event to trace.

    Returns:
        bool: True if successful, False otherwise.
    """
    if DEBUG_MODE in os.environ:
        return True

    sufix = ""
    if event.name == "Evaluation":
        sufix = "evaluation"

    payload = event.model_dump()
    return send_request(
        url=SERVER_URL,
        json=payload,
        url_sufix=sufix,
    )
