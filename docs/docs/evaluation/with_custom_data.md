# with custom data

We can create custom data from various sources

## from a datafram

```python
from ragrank.dataset import from_df
from pandas import DataFrame
from ragrank import evaluate

data = DataFrame()
results = evaluate(data)
results
```

## from langchain docs

## from llama index

## from a file