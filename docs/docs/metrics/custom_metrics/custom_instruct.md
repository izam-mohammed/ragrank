(custom-instruct)=
# Custom Instruct

Custom Instruct is not like pre-defined metrics. It have to initialize with the given configuration before using the metric. We can give custom instructions for the evaluations. It can be both binary metric or Non-binary metric.

```{Hint}
- **question**:
- **context**:
- **response**:
- **configured instructions**:
- **score**:
```

Importing the neccessory packages and modules
```python
from ragrank.dataset import DataNode
from ragrank import evaluate
from ragrank.metric import CustomInstruct, InstructConfig
from ragrank.metric.base import MetricType
```

Setting up the instructions.
```python
grammar_checker = CustomInstruct(
    config=InstructConfig(
        metric_type=MetricType.NON_BINARY,
        name="grammar checker",
        instructions="Give a score for grammar in the response",
        examples=[
            {
                "question": "how are you ?",
                "response": "I am fine, Thank you for asking",
                "score": 0.99
            },
            {
                "question": "how are you ?",
                "response": "I fine, thank in asking",
                "score": 0.03,
            }
        ],
        input_fields=["question", "response"],
        output_field="score",
    )
)
```

Performing the evaluation
```python
# defining the data node
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
    metrics=[grammar_checker],
)
print(result)
```