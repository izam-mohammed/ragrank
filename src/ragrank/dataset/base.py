"""Contain all of the base classes for dataset"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pandas import DataFrame
from tqdm import tqdm

from ragrank.bridge.pydantic import BaseModel, Field, model_validator

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

    def to_dataset(self) -> Dataset:
        """
        Convert the data node to a Dataset instance.

        Returns:
            Dataset: A Dataset instance containing the current data node.
        """
        dataset = Dataset(
            question=[self.question],
            context=[self.context],
            response=[self.response],
        )
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
        combined_dataset = Dataset(
            question=[self.question, other.question],
            context=[self.context, other.context],
            response=[self.response, other.response],
        )
        return combined_dataset


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

    @model_validator(mode="after")
    def validator(self) -> Dataset:
        """
        Validate the dataset after instantiation.

        Raises:
            ValueError: If the number of data points is not consistent
                across question, context, and response.
        """
        if not len(self.question) == len(self.response):
            raise ValueError(
                "The number of datapoints in question "
                "and response should be equal. \n"
                "Ensure that all lists contain the same number of datapoints."
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
        return DataNode(
            question=self.question[index],
            context=self.context[index],
            response=self.response[index],
        )

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

    def __add__(self, other: Dataset) -> Dataset:
        """
        Concatenate two datasets.

        Args:
            other (Dataset): The dataset to concatenate with.

        Returns:
            Dataset: The concatenated dataset.
        """
        combined_dataset = Dataset(
            question=self.question + other.question,
            context=self.context + other.context,
            response=self.response + other.response,
        )
        return combined_dataset

    def with_progress(self, purpose: str = "Iterating") -> tqdm:
        """
        Return a tqdm progress bar for iterating over the dataset.

        Args:
            purpose (str): The purpose for iterating over the dataset.

        Returns:
            tqdm: A tqdm progress bar.
        """
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
        return self.model_dump()

    def to_dataframe(self) -> DataFrame:
        """Return a pandas dataframe of the data

        Args:
            None

        Returns:
            DataFrame: data representation
        """
        return DataFrame(self.to_dict())

    def to_csv(self, path: str | Path, **kwargs: Any) -> None:  # noqa: ANN401
        """Save the data as a csv file

        Args:
            path (str | Path): path to the csv file

        Returns:
            None
        """
        dataframe = self.to_dataframe()
        dataframe.to_csv(path_or_buf=path, index=False, **kwargs)
