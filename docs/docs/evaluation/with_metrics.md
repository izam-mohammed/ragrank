(with-custom-metrics)=
# With Metrics

Ragrank have multiple evaluation to assess the your RAG models. On top of that you can create your own metric for evaluation.

Metrics availble in Ragrank:
- Context Relevancy
    ```python
    from ragrank.metric import response_relevancy
    ```
- Context Utilization
    ```python
    from ragrank.metric import response_conciseness
    ```
- Response Conciseness
    ```python
    from ragrank.metric import context_relevancy
    ```
- Response Relevancy
    ```python
    from ragrank.metric import context_utilization 
    ```

Evaluate with all of the above metrics

```python
from ragrank.dataset import DataNode
from ragrank import evaluate
from ragrank.metric import (
    response_relevancy,
    response_conciseness,
    context_relevancy,
    context_utilization,
)

# defining the data node
data = DataNode(
    question="What is the tallest mountain in the world?",
    context=[
        "Mount Everest is the tallest mountain above sea level.",
        "It is located in the Himalayas.",
    ],
    response="The tallest mountain in the world is Mount Everest.",
)

# evaluating the metrics
result = evaluate(
    data ,
    metrics=[
        response_relevancy,
        response_conciseness,
        context_relevancy,
        context_utilization
    ]
)

print(result)
```

Also there custom metris are also available

```{Warning}
`CustomMetric` and `Custom Instruct` is not like other pre defined metrics. You have to setup the metric first before using it. Checkout the [Custom Metric Guide](../metrics/custom_metrics/index.md) to make custom metrics
```

- Cutom Metric
    ```python
    from ragrank.metric import CustomMetric
    ```
- Custom Instruct
    ```python
    from ragrank.metric import CustomInstruct, InstructConfig
    ```