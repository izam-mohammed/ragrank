# Context f1

This measure the context f1 score for the context of the model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import context_f1

data = load_data()
result = evaluate(
    data,
    metrics=[
        context_f1,
    ],
)
result
```