# Readability

This measure the readability of the code

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import code_readability

data = load_data()
result = evaluate(
    data,
    metrics=[
        code_readability(language="python"),
    ],
)
result
```