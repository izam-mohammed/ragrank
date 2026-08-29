# OpenLLM

Reachable today through [LiteLLM](./litellm.md), which is why there is
no separate wrapper for it.

```bash
pip install "ragrank[litellm]"
```

```python
from ragrank.integrations.litellm import LiteLLM

judge = LiteLLM(model="openai/my-model")
```

Self-hosted models need the endpoint as well:

```python
judge = LiteLLM(
    model="openai/my-model",
    api_base="http://localhost:3000/v1",
)
```

Anything speaking the OpenAI wire format works this way, OpenLLM
included.
