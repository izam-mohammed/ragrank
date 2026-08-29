"""Contain all of the base classes for dataset"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ragrank.bridge.pydantic import BaseModel, Field, model_validator
from ragrank.utils.optional import is_available, require

if TYPE_CHECKING:
    from pandas import DataFrame

DATANODE_DICT_TYPE = dict[str, list[str] | str]
DATASET_DICT_TYPE = dict[str, list[str] | list[list[str]]]

logger = logging.getLogger(__name__)


class DataNode(BaseModel):
    """
    Represents a single data point in a dataset.

    Attributes:
        question (str): The question associated with the data point.
        context (list[str]): The context or background
            nformation related to the question.
        response (str): The response or answer to the question.
        reference (str | None): The ground truth answer, when one is
            known. Metrics that measure correctness need it; metrics
            that are reference-free ignore it.
        retrieved_ids (list[str] | None): Identifiers of the documents
            the retriever returned, in rank order.
        reference_ids (list[str] | None): Identifiers of the documents
            that *should* have been retrieved. With `retrieved_ids`
            this enables the ranking metrics, which cost no LLM calls.
    """

    question: str = Field(
        description="The question associated with the data point"
    )
    context: list[str] = Field(
        description="The context information related to the question"
    )
    response: str = Field(
        description="The response or answer to the question"
    )
    reference: str | None = Field(
        default=None,
        description="The ground truth answer, when one is known",
    )
    retrieved_ids: list[str] | None = Field(
        default=None,
        description="Ids of the retrieved documents, in rank order",
    )
    reference_ids: list[str] | None = Field(
        default=None,
        description="Ids of the documents that should be retrieved",
    )

    def to_dataset(self) -> Dataset:
        """
        Convert the data node to a Dataset instance.

        Returns:
            Dataset: A Dataset instance containing the current data node.
        """
        dataset = Dataset(**{
            field: [value]
            for field, value in self.model_dump(
                exclude_none=True
            ).items()
        })
        logger.info("DataNode converted to Dataset succesfully !")
        return dataset

    def __add__(self, other: DataNode) -> Dataset:
        """
        Concatenate two Datanodes.

        Args:
            other (DataNode): The datanode to concatenate with.

        Returns:
            Dataset: The concatenated dataset.
        """
        return self.to_dataset() + other.to_dataset()


class Dataset(BaseModel):
    """
    Represents a dataset containing questions, contexts,
        and responses.

    Attributes:
        question (list[str]): A list of questions.
        context (list[list[str]]): A list of contexts,
            each represented as a list of strings.
        response (list[str]): A list of responses
            corresponding to the questions.
        reference (list[str] | None): Ground truth answers, if known.
        retrieved_ids (list[list[str]] | None): Retrieved document ids.
        reference_ids (list[list[str]] | None): Expected document ids.
    """

    question: list[str] = Field(
        description="A list of questions, each represented as a string"
    )
    context: list[list[str]] = Field(
        description="A list of contexts, each represented as a list of strings"
    )
    response: list[str] = Field(
        description="A list of responses corresponding to the questions"
    )
    reference: list[str] | None = Field(
        default=None, description="Ground truth answers, if known"
    )
    retrieved_ids: list[list[str]] | None = Field(
        default=None, description="Retrieved document ids"
    )
    reference_ids: list[list[str]] | None = Field(
        default=None, description="Expected document ids"
    )

    OPTIONAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "reference",
        "retrieved_ids",
        "reference_ids",
    )

    @model_validator(mode="after")
    def validator(self) -> Dataset:
        """
        Validate the dataset after instantiation.

        Raises:
            ValueError: If the number of data points is not consistent
                across question, context, and response.
        """
        lengths = {
            "question": len(self.question),
            "context": len(self.context),
            "response": len(self.response),
        }
        for name in self.OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                lengths[name] = len(value)

        if len(set(lengths.values())) != 1:
            detail = ", ".join(
                f"{name}={size}" for name, size in lengths.items()
            )
            raise ValueError(
                "Every column of a dataset must have the same number "
                "of datapoints. \n"
                f"Got {detail}. Ensure that all lists contain the "
                "same number of datapoints."
            )

        return self

    def __len__(self) -> int:
        """
        Return the number of questions in the dataset.

        Returns:
            int: The number of questions in the dataset.
        """
        return len(self.question)

    def __getitem__(self, index: int) -> DataNode:
        """
        Retrieve a single data point from the dataset by index.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            dataNode: The question, context, and response of the data point.
        """
        fields: dict[str, Any] = {
            "question": self.question[index],
            "context": self.context[index],
            "response": self.response[index],
        }
        for name in self.OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                fields[name] = value[index]
        return DataNode(**fields)

    def __iter__(self) -> Iterator[DataNode]:
        """
        Returns an iterator over the dataset, yielding each DataNode.

        Returns:
            Iterator[DataNode]: An iterator yielding DataNode instances.
        """
        for i in range(len(self)):
            yield self[i]

    def append(self, data_node: DataNode) -> None:
        """
        Append a DataNode to the dataset.

        Args:
            data_node (DataNode): The DataNode to append.
        """
        self.question.append(data_node.question)
        self.context.append(data_node.context)
        self.response.append(data_node.response)
        for name in self.OPTIONAL_FIELDS:
            column = getattr(self, name)
            if column is not None:
                column.append(getattr(data_node, name))

    def __add__(self, other: Dataset) -> Dataset:
        """
        Concatenate two datasets.

        Args:
            other (Dataset): The dataset to concatenate with.

        Returns:
            Dataset: The concatenated dataset.
        """
        fields: dict[str, Any] = {
            "question": self.question + other.question,
            "context": self.context + other.context,
            "response": self.response + other.response,
        }
        for name in self.OPTIONAL_FIELDS:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine is not None and theirs is not None:
                fields[name] = mine + theirs
        return Dataset(**fields)

    def with_progress(
        self, purpose: str = "Iterating"
    ) -> Iterator[DataNode]:
        """
        Return an iterator over the dataset, with a progress bar if
        `tqdm` is installed.

        Args:
            purpose (str): The purpose for iterating over the dataset.

        Returns:
            Iterator[DataNode]: An iterator over the data nodes.
        """
        if not is_available("tqdm"):
            return iter(self)

        from tqdm import tqdm

        return tqdm(
            self,
            ncols=100,
            desc=purpose + " ",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt}   "
                "remain: {remaining}s, {rate_fmt}"
            ),
            colour="green",
            leave=True,
        )

    def to_dict(self) -> DATASET_DICT_TYPE:
        """Return a dict of the data

        Args:
            None

        Returns:
            dict: data representation
        """
        return self.model_dump(exclude_none=True)

    def to_dataframe(self) -> DataFrame:
        """Return a pandas dataframe of the data

        Args:
            None

        Returns:
            DataFrame: data representation
        """
        pd = require("pandas", "pandas")
        return pd.DataFrame(self.to_dict())

    def to_csv(self, path: str | Path, **kwargs: Any) -> None:  # noqa: ANN401
        """Save the data as a csv file

        Args:
            path (str | Path): path to the csv file

        Returns:
            None
        """
        dataframe = self.to_dataframe()
        dataframe.to_csv(path_or_buf=path, index=False, **kwargs)
