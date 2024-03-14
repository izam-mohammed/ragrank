"""Reader module for Ragrank"""

from __future__ import annotations

from typing import Dict, List, Union

from ragrank.dataset import DataNode, Dataset

DATANODE_TYPE = Dict[str, Union[List[str], str]]
DATASET_TYPE = Dict[str, Union[List[str], List[List[str]]]]


def from_dict(
    data: Union[DATANODE_TYPE, DATASET_TYPE],
    *,
    return_as_dataset: bool = False,
) -> Union[Dataset, DataNode]:
    """
    Create a Dataset or DataNode object from a dictionary representation.

    Args:
        data (Union[DATANODE_TYPE, DATASET_TYPE]): The dictionary containing
            the data representation.

    Returns:
        Union[Dataset, DataNode]: Either a Dataset or DataNode object.

    Raises:
        ValidationError: If there is an issue with the
            schema validation.
    """

    if any(isinstance(i, str) for i in data.values()):
        if return_as_dataset:
            return Dataset(**{i: [data[i]] for i in data})
        return DataNode(**data)
    return Dataset(**data)
