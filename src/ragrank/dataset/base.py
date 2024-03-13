"""Contain all of the base classes for dataset"""

from __future__ import annotations

from typing import Iterator, List

from ragrank.bridge.pydantic import BaseModel, model_validator


class DataNode(BaseModel):
    """
    Represents a single data point in a dataset.

    Attributes:
        question (str): The question associated with the data point.
        context (List[str]): The context or background
            nformation related to the question.
        response (str): The response or answer to the question.
    """

    question: str
    context: List[str]
    response: str


class Dataset(BaseModel):
    """
    Represents a dataset containing questions, contexts,
        and responses.

    Attributes:
        question (List[str]): A list of questions.
        context (List[List[str]]): A list of contexts,
            each represented as a list of strings.
        response (List[str]): A list of responses
            corresponding to the questions.
    """

    question: List[str]
    context: List[List[str]]
    response: List[str]

    @model_validator(mode="after")
    def validator(self) -> None:
        """
        Validate the dataset after instantiation.

        Raises:
            ValueError: If the number of data points is not consistent
                across question, context, and response.
        """
        if not len(self.question) == len(self.context) == len(self.context):
            raise ValueError("Number of datapoints should be stable")

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
