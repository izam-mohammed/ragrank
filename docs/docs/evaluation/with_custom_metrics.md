# with custom metric

We can create custom metrics with Ragrank

```python
from ragrank import evaluate
from ragrank.metrics import custom_metric
from ragrank.dataset import Dataset

metric = custom_metric()
data = Dataset()
result = evaluate(data)
result
```