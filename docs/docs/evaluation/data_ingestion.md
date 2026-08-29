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

## Scoring output from somewhere else

The shape most frameworks hand you is a list of records - one
self-describing mapping per datapoint. Ragrank reads that directly, so
there is nothing to reshape first.

- **from_records**: Ingesting a list of flat mappings.
    ```python
    from ragrank.dataset import from_records

    data = from_records([
        {
            "question": "What is the largest mammal on Earth?",
            "context": ["The blue whale holds the title."],
            "response": "The blue whale.",
        },
    ])
    ```

    A `context` given as a single string is wrapped into a one-chunk
    list. Fields ragrank does not recognise - a trace id, a timestamp -
    are ignored rather than rejected. Every record must carry the same
    fields, and a ragged one is named in the error.

- **from_json**: Ingesting JSON, in either common shape.
    ```python
    from ragrank.dataset import from_json

    data = from_json("outputs.json")
    ```

    Accepts a path, a JSON document, or already-parsed data, and reads
    both a list of records and a mapping of columns. Different tools
    emit different ones and neither is worth arguing about.

- **from_jsonl**: Ingesting JSON Lines, one record per line.
    ```python
    from ragrank.dataset import from_jsonl

    data = from_jsonl("outputs.jsonl")
    ```

    Line-delimited JSON streams and appends, which a single JSON array
    does not, so it is the better shape for a dataset that grows. Blank
    lines are skipped; a line that will not parse is reported by number.

### Writing it back out

```python
data.to_records()              # list of dicts
data.to_json("data.json")      # returns the string too
data.to_jsonl("data.jsonl")
data.to_csv("data.csv")
```

Which makes ragrank a way to move a dataset between tools, not only a
way to score one.
