"""Reader module for Ragrank"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from ragrank.bridge.pydantic import BaseModel
from ragrank.dataset import DataNode, Dataset

DATANODE_TYPE = Dict[str, Union[List[str], str]]
DATASET_TYPE = Dict[str, Union[List[str], List[List[str]]]]
RAGRANK_DICT_TYPE = Union[DATANODE_TYPE, DATASET_TYPE]


class ColumnMap(BaseModel):
    question: str = "question"
    context: str = "context"
    response: str = "response"


def from_dict(
    data: RAGRANK_DICT_TYPE,
    *,
    return_as_dataset: bool = False,
    column_map: ColumnMap = ColumnMap(),
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
    for key, value in column_map.items():
        if value not in data:
            raise ValueError(
                f"The column {value} not in the data"
            ) from ValueError
    data = {key: data[value] for key, value in column_map}  # mapping col

    if any(isinstance(i, str) for i in data.values()):
        if return_as_dataset:
            return Dataset(**{i: [data[i]] for i in data})
        return DataNode(**data)
    return Dataset(**data)


def from_csv(
    path: Union[str, Path],
    *args,
    column_map: ColumnMap = ColumnMap(),
    **kwargs,
) -> Union[Dataset, DataNode]:
    df: pd.DataFrame = pd.read_csv(filepath_or_buffer=path, *args, **kwargs)
    dict_data: RAGRANK_DICT_TYPE = df.to_dict(column_map=column_map)
    return from_dict(dict_data)