# Detailed Evaluation

Here is the detailed evaluation with Ragrank.

```python
from ragrank import evaluate
from ragrank.dataset import from_df
from pandas import DataFrame

df = DataFrame()

dataset = from_df(df)
result = evaluate(dataset)
print(result.to_frame())
```

this will continue ...
