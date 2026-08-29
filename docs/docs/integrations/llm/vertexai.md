# Vertex AI

Reachable today through [LiteLLM](./litellm.md), which is why there is
no separate wrapper for it.

```bash
pip install "ragrank[litellm]"
```

```python
from ragrank.integrations.litellm import LiteLLM

judge = LiteLLM(model="vertex_ai/gemini-2.0-flash")
```

Vertex AI reads its credentials the usual Google way - either
`GOOGLE_APPLICATION_CREDENTIALS` pointing at a service account file, or
an already-authenticated `gcloud` session.

```python
judge = LiteLLM(
    model="vertex_ai/gemini-2.0-flash",
    extra_params={"vertex_project": "my-project", "vertex_location": "us-central1"},
)
```
