# Cohere

Reachable today through [LiteLLM](./litellm.md), which is why there is
no separate wrapper for it.

```bash
pip install "ragrank[litellm]"
```

```python
from ragrank.integrations.litellm import LiteLLM

judge = LiteLLM(model="cohere/command-r")
```

