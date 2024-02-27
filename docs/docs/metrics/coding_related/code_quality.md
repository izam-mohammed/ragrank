# Code Quality

This measure the code quality 

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import code_quality

data = load_data()
result = evaluate(
    data,
    metrics=[
        code_quality(language="python"),
    ],
)
result
```