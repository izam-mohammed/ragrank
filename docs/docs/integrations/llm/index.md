(integration-llm)=
# LLM

Ragrank can judge with any of these.

| Integration | Reach |
| --- | --- |
| [LiteLLM](./litellm.md) | A hundred-odd providers through one wrapper |
| [OpenAI](../../evaluation/with_llm.md) | The default, if you are already there |
| LangChain | Any LangChain chat model |
| LlamaIndex | Any LlamaIndex LLM |

If the provider you want is not the default, start with
[LiteLLM](./litellm.md) - it is one dependency instead of one wrapper
per provider.

```{toctree}
:hidden:

litellm
cohere
vertexai
openllm
```
