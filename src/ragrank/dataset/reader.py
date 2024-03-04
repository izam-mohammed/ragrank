"""Reader module for Ragrank"""

from __future__ import annotations

from typing import Dict, List

from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import Dataset
from ragrank.exceptions import ValidationError as RagrankValidationError


def from_dict(data: Dict[str, List[str] | str]) -> Dataset:
    """create a dataset object from dictionary"""

    try:
        return Dataset(**data)
    except ValidationError as e:
        raise RagrankValidationError(str(e)) from ValidationError
