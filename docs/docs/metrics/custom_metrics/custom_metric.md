(custom-metric)=
# Custom Metric

Ragrank allows to create custom metrics as you wish. You can inherit the class `CustomMetric` and you have to write these 3 methods
- `metric_name() -> str`: Have to return the name of your metric
- `metric_score(data: DataNode) -> float`: It will get a datanode and have to return the score.
- `_reason(data:DataNode, score:float) -> str`: This is optional. You will get the data and score. Have to return the reason for it as string

configure the metric
```python
from ragrank.dataset.base import DataNode
from ragrank.metric import CustomMetric

class MyMetric(CustomMetric):
    def metric_score(self, data: DataNode) -> float:
        prompt = f"give a random performance number for this datapoint \n\n{data.model_dump()}"
        prompt += "you only have to give a number between 0 to 1. don't need any explaination. the response shold be a number"
        llm_response = self.llm.generate_text(prompt)
        try:
            return float(llm_response.response)
        except:
            return 0.0
        
    def metric_name(self) -> str:
        return "my metric"
    
my_metric = MyMetric()
```

evaluation using the `my_metric`
```python
from ragrank import evaluate
from ragrank.dataset import DataNode

data = DataNode(
    question="What is the tallest mountain in the world?",
    context=[
        "Mount Everest is the tallest mountain above sea level.",
        "It is located in the Himalayas.",
    ],
    response="The tallest mountain in the world is Mount Everest.",
)

result = evaluate(
    dataset=data,
    metrics=[
        my_metric,
    ]
)

print(result)
```