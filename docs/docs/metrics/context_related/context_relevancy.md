(context-relevancy)=
# Context relevancy

This measure how relevant the context provided for the given question. It is a non-binary metric. So the score will be a value between 0 and 1.

```{Hint}
- **question**: 
- **High relevant context**:
- **Low relevant context**
```

```python 
from ragrank import evaluate
from ragrank.dataset import DataNode
from ragrank.metric import response_relevancy

data = DataNode(
    question="What is the tallest mountain in the world?",
    context=[
        "Mount Everest is the tallest mountain above sea level.",
        "It is located in the Himalayas.",
    ],
    response="The tallest mountain in the world is Mount Everest.",
)

result = evaluate(
    data ,
    metrics=[
        response_relevancy,
    ]
)

print(result)
```