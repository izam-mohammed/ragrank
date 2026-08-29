"""Reader module for Ragrank"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from ragrank.bridge.pydantic import BaseModel, Field
from ragrank.dataset import DataNode, Dataset
from ragrank.utils.common import eval_cell
from ragrank.utils.optional import require

DATANODE_TYPE = dict[str, list[str] | str]
DATASET_TYPE = dict[str, list[str] | list[list[str]]]
RAGRANK_DICT_TYPE = DATANODE_TYPE | DATASET_TYPE


def from_dict(
    data: RAGRANK_DICT_TYPE,
    *,
    return_as_dataset: bool = False,
    column_map: ColumnMap | None = None,
) -> Dataset | DataNode:
    """
    Create a Dataset or DataNode object from a dictionary representation.

    Args:
        data (Union[DATANODE_TYPE, DATASET_TYPE]): The dictionary containing
            the data representation.
        return_as_dataset (bool, optional): If True, return as Dataset object,
            otherwise return as DataNode. Defaults to False.
        column_map (ColumnMap | None, optional): Column mapping.
            Defaults to ColumnMap().

    Returns:
        Union[Dataset, DataNode]: Either a Dataset or DataNode object.

    Raises:
        ValueError: If the column specified in column_map
            is not present in the data.
    """
    if column_map is None:
        column_map = ColumnMap()

    mapping = column_map.model_dump()
    missing = [
        mapping[field]
        for field in ColumnMap.REQUIRED
        if mapping[field] not in data
    ]
    if missing:
        raise ValueError(
            f"The column {missing[0]} not in the data"
        ) from None

    # Optional columns are used only when the caller supplied them.
    data = {
        key: data[value]
        for key, value in mapping.items()
        if value in data
    }

    if any(isinstance(i, str) for i in data.values()):
        if return_as_dataset:
            return Dataset(**{i: [data[i]] for i in data})
        return DataNode(**data)
    return Dataset(**data)


def from_records(
    records: list[dict[str, Any]],
    *,
    column_map: ColumnMap | None = None,
) -> Dataset:
    """Create a Dataset from a list of flat records.

    This is the interchange entry point: one self-describing mapping per
    datapoint is how most other frameworks hand you their outputs, so
    scoring somebody else's results does not require reshaping them into
    columns first.

    A `context` given as a single string is wrapped into a one-chunk
    list, since a record written by hand usually does that.

    Args:
        records (list[dict[str, Any]]): One mapping per datapoint.
        column_map (ColumnMap | None, optional): Column mapping.
            Defaults to ColumnMap().

    Returns:
        Dataset: The loaded dataset.

    Raises:
        ValueError: If the records are empty, are not mappings, or do
            not all carry the same fields.
    """
    if not records:
        raise ValueError("No records to read.")

    bad = next(
        (
            index
            for index, item in enumerate(records)
            if not isinstance(item, dict)
        ),
        None,
    )
    if bad is not None:
        raise ValueError(
            f"Record {bad} is a {type(records[bad]).__name__}, "
            "not a mapping."
        )

    if column_map is None:
        column_map = ColumnMap()
    mapping = column_map.model_dump()
    context_column = mapping["context"]

    keys = set(records[0])
    for index, record in enumerate(records[1:], start=1):
        if set(record) != keys:
            raise ValueError(
                f"Record {index} has fields {sorted(record)}, but "
                f"record 0 has {sorted(keys)}. Every record must "
                "carry the same fields."
            )

    columns: DATASET_TYPE = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if key == context_column:
            values = [
                [value] if isinstance(value, str) else value
                for value in values
            ]
        columns[key] = values

    data = from_dict(
        columns, column_map=column_map, return_as_dataset=True
    )
    return data if isinstance(data, Dataset) else data.to_dataset()


def _read_source(source: str | Path) -> str:
    """Resolve a path or a literal document into text.

    A string that starts with `[` or `{` is taken as the document
    itself; anything else is treated as a path.

    Args:
        source (str | Path): A path, or the document text.

    Returns:
        str: The document text.
    """
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    if source.lstrip()[:1] in {"[", "{"}:
        return source
    return Path(source).read_text(encoding="utf-8")


def from_json(
    source: str | Path | list[dict[str, Any]],
    *,
    column_map: ColumnMap | None = None,
) -> Dataset:
    """Create a Dataset from JSON, in either common shape.

    Accepts a path, a JSON string, or already-parsed data, and reads
    both a list of records and a dict of columns, because different
    frameworks emit different ones and neither is worth arguing about.

    Args:
        source (str | Path | list[dict[str, Any]]): A path to a JSON
            file, a JSON document, or parsed records.
        column_map (ColumnMap | None, optional): Column mapping.
            Defaults to ColumnMap().

    Returns:
        Dataset: The loaded dataset.

    Raises:
        ValueError: If the document is neither a list nor a mapping.

    Examples::

        from ragrank.dataset import from_json

        dataset = from_json("outputs.json")
    """
    parsed = (
        source
        if isinstance(source, list)
        else json.loads(_read_source(source))
    )

    if isinstance(parsed, list):
        return from_records(parsed, column_map=column_map)
    if isinstance(parsed, dict):
        data = from_dict(
            parsed, column_map=column_map, return_as_dataset=True
        )
        return (
            data if isinstance(data, Dataset) else data.to_dataset()
        )
    raise ValueError(
        "JSON data should be a list of records or a mapping of "
        f"columns, not {type(parsed).__name__}."
    )


def from_jsonl(
    source: str | Path,
    *,
    column_map: ColumnMap | None = None,
) -> Dataset:
    """Create a Dataset from JSON Lines, one record per line.

    Blank lines are skipped, so a trailing newline is not an error.

    Args:
        source (str | Path): A path to a JSONL file, or the document.
        column_map (ColumnMap | None, optional): Column mapping.
            Defaults to ColumnMap().

    Returns:
        Dataset: The loaded dataset.

    Raises:
        ValueError: If a line is not valid JSON.
    """
    text = (
        source
        if isinstance(source, str) and "\n" in source.strip()
        else _read_source(source)
    )

    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Line {number} is not valid JSON: {error}"
            ) from None

    return from_records(records, column_map=column_map)


def from_dataframe(
    data: pd.DataFrame,
    *,
    return_as_dataset: bool = False,
    column_map: ColumnMap | None = None,
) -> Dataset | DataNode:
    """
    Create a Dataset or DataNode object from a Pandas DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        return_as_dataset (bool, optional): If True, return as Dataset object,
            otherwise return as DataNode. Defaults to False.
        column_map (ColumnMap | None, optional): Column mapping.
            Defaults to ColumnMap().

    Returns:
        Union[Dataset, DataNode]: Either a Dataset or DataNode object.
    """
    if column_map is None:
        column_map = ColumnMap()

    modified_data = data.map(eval_cell)
    dict_data = {
        key: list(value_dict.values())
        for key, value_dict in modified_data.to_dict().items()
    }
    return from_dict(
        data=dict_data,
        return_as_dataset=return_as_dataset,
        column_map=column_map,
    )


def from_csv(
    path: str | Path,
    *,
    column_map: ColumnMap | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Dataset | DataNode:
    """
    Create a Dataset or DataNode object from a CSV file.

    Args:
        path (Union[str, Path]): The path to the CSV file.
        column_map (ColumnMap | None, optional): Column mapping.
            Defaults to ColumnMap().
        **kwargs: Keyword arguments to pass to pandas read_csv function.

    Returns:
        Union[Dataset, DataNode]: Either a Dataset or DataNode object.
    """
    if column_map is None:
        column_map = ColumnMap()

    df = pd.read_csv(filepath_or_buffer=path, **kwargs)
    return from_dataframe(data=df, column_map=column_map)


def from_hfdataset(
    url: str | tuple[str],
    *,
    split: str,
    column_map: ColumnMap | None = None,
) -> Dataset:
    """
    Create a Dataset object from a Hugging Face dataset.

    Args:
        url (Union[str, Tuple[str]]): The URL or tuple of URLs
            pointing to the dataset.
        split (str): The name of the split to load from the dataset.
        column_map (ColumnMap | None, optional): Column mapping.
            Defaults to ColumnMap().

    Returns:
        Dataset: A Dataset object containing the loaded data.
    """
    if column_map is None:
        column_map = ColumnMap()
    hf_datasets = require("datasets", "hf")
    dataset = (
        hf_datasets.load_dataset(url)
        if isinstance(url, str)
        else hf_datasets.load_dataset(*url)
    )
    data = dataset[split]
    data_dict = {
        column: data[column] for column in data.column_names
    }
    data_dict[column_map.context] = [
        eval_cell(cell) for cell in data_dict[column_map.context]
    ]
    return from_dict(
        data_dict, column_map=column_map, return_as_dataset=True
    )


class ColumnMap(BaseModel):
    """
    Represents a mapping of column names to their
        corresponding names in a dataset.

    Attributes:
        question (str): The name of the column containing questions.
        context (str): The name of the column containing contexts.
        response (str): The name of the column containing responses.
    """

    question: str = Field(
        default="question",
        description="The name of the column containing questions",
    )
    context: str = Field(
        default="context",
        description="The name of the column containing contexts",
    )
    response: str = Field(
        default="response",
        description="The name of the column containing responses",
    )
    reference: str = Field(
        default="reference",
        description="The name of the column containing ground truths",
    )
    retrieved_ids: str = Field(
        default="retrieved_ids",
        description="The name of the column containing retrieved ids",
    )
    reference_ids: str = Field(
        default="reference_ids",
        description="The name of the column containing expected ids",
    )

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "question",
        "context",
        "response",
    )
