# Callbacks

```{attention}
This feature is not implemented yet ! Will update this shortly 😊
```

Ragrank support the langchain and llamaindex callbacks

```python
from ragrank import evaluate
from dataset import Dataset
from ragrank.callbacks import Callbackmanager

data = Dataset()
callbacks = Callbackmanager()
results = evaluate(data, callbacks=[callbacks])
results
```
