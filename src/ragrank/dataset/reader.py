"""Reader module for Ragrank"""

from __future__ import annotations

from typing import Dict, List, Union

from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import DataNode, Dataset
from ragrank.exceptions import ValidationError as RagrankValidationError

DATANODE_TYPE = Dict[str, Union[List[str], str]]
DATASET_TYPE = Dict[str, Union[List[str], List[List[str]]]]


def from_dict(
    data: Union[DATANODE_TYPE, DATASET_TYPE],
) -> Union[Dataset, DataNode]:
    """
    Create a Dataset or DataNode object from a dictionary representation.

    Args:
        data (Union[DATANODE_TYPE, DATASET_TYPE]): The dictionary containing
            the data representation.

    Returns:
        Union[Dataset, DataNode]: Either a Dataset or DataNode object.

    Raises:
        RagrankValidationError: If there is an issue with the
            schema validation.
    """

    try:
        if any(isinstance(i, str) for i in data.values()):
            return DataNode(**data)
        return Dataset(**data)
    except ValidationError as e:
        raise RagrankValidationError(f"Schema validation failed: {e}") from e
