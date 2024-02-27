# Modularity

This measure the modularity of the code

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import code_modularity

data = load_data()
result = evaluate(
    data,
    metrics=[
        code_modularity(language="python"),
    ],
)
result
```