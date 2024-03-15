"""Private module for tracking"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, TypeVar

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field, validate_call
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    time: str = Field(
        default_factory=lambda: datetime.now().isoformat(), repr=False
    )
    time_cost: float = Field(ge=0)


class EvaluationEvent(BaseEvent):
    """
    Event class for evaluation.

    Attributes:
        llm (str): Long lived model.
        metrics (Dict[str, Optional[float]]): Metrics and their averages.
        data_size (int): Size of the data.
    """

    name: str = Field(default="Evaluation", frozen=True, exclude=True)
    llm: str
    metrics: Dict[str, Optional[float]]  # metric and average
    data_size: int


class DataGenerationEvent(BaseEvent):
    """
    Event class for data generation.

    Attributes:
        data_shape (Tuple[int]): Shape of the generated data.
        source (Optional[str]): Source of the data.
    """

    name: str = Field(default="DataGeneration", frozen=True, exclude=True)
    data_size: int = Field(gt=0)
    source: Optional[str]


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
    elif event.name == "DataGeneration":
        sufix = "data-generation"

    payload = event.model_dump()
    return send_request(
        url=SERVER_URL,
        json=payload,
        url_sufix=sufix,
    )
