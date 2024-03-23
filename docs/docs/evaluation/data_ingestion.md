(data-ingestion)=
# Data Ingestion

In Ragrank you can add data in multiple ways. 2 types of data is there.

- `DataNode`: Contain single data point.
    ```python
    from ragrank.dataset import DataNode

    example_datanode = DataNode(
        question="What is the tallest mountain in the world?",
        context=[
            "Mount Everest is the tallest mountain above sea level.", 
            "It is located in the Himalayas.",
            ],
        response="The tallest mountain in the world is Mount Everest."
    )
    ```
- `Dataset`: Contain multiple data points.
    ```python
    from ragrank.dataset import Dataset

    example_dataset = Dataset(
        question=[
            "What is the tallest mountain in the world?",
            "Who wrote the Harry Potter series?",
        ],
        context=[
            [
                "Mount Everest is the tallest mountain above sea level.",
                "It is located in the Himalayas.",
            ],
            [
                "J.K. Rowling wrote the Harry Potter series.",
                "The series became extremely popular worldwide.",
            ],
        ],
        response=[
            "The tallest mountain in the world is Mount Everest.",
            "The Harry Potter series was written by J.K. Rowling.",
        ]
    )
    ```

## Column Map

In a datapoint, there should If your data columns have different names you can use `Column Map`

```python
from ragrank.dataset import from_csv, ColumnMap

example_datanode = {
    "question": "What is the largest mammal on Earth?",
    "context": [
        "The blue whale holds the title of the largest mammal.",
        "It is a marine mammal found in oceans around the world.",
    ],
    "response": "The largest mammal on Earth is the blue whale.",
}

data = from_csv(
    example_datanode,
    column_map=ColumnMap(
        question="query", context="related_context", response="answer"
    ),
)
```
In all reader methods, you can use `ColumnMap` to map columns.

```{Caution}
Internally, the data object is saving the data in the `question`, `context`, and `response` fields. After reading the data, the previous field names are not preserved. You can't access the data with the previous field names either.
```

## Data Readers

There are multiple data readers availble in Ragrank.

- **from_dict**: ingest data from a dict. Will convert `DataNode` and `Dataset` according to the type of data.
    ```python
    from ragrank.dataset import from_dict

    data = from_dict(
        {
            "question": "What is the largest mammal on Earth?",
            "context": [
                "The blue whale holds the title of the largest mammal.",
                "It is a marine mammal found in oceans around the world.",
            ],
            "response": "The largest mammal on Earth is the blue whale.",
        },
        return_as_dataset=False, 
        column_map=None # specify if any
    )
    ```

- **from_csv**: ingesting data from csv file
    ```python
    from ragrank.dataset import from_csv

    data = from_csv(
        path="data.csv", 
        column_map=None, # specify if any
    )
    ```

- **from_dataframe**: Ingesting data from Pandas DataFrame.
    ```python
    from ragrank.dataset import from_dataframe
    from pandas import DataFrame

    dataframe = DataFrame(
        {
            "question": "What is the largest mammal on Earth?",
            "context": [
                "The blue whale holds the title of the largest mammal.",
                "It is a marine mammal found in oceans around the world.",
            ],
            "response": "The largest mammal on Earth is the blue whale.",
        }
    )

    data = from_dataframe(
        data=dataframe,
        column_map=None # specify if any
    )
    ```

- **from_hfdataset**: Ingesting data from Huggingface datasets
    ```python
    from ragrank.dataset import from_hfdataset

    data = from_hfdataset(
        url="izammohammed/engineering_qa", 
        split="train", 
        column_map=None # specify if any
    )