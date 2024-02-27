# Domain Coverage

This measure the coverage of a particular domain

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import domain_coverage

data = load_data()
result = evaluate(
    data,
    metrics=[
        domain_coverage(domain="medical"),
    ],
)
result
```