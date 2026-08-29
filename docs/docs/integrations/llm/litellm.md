(integration-litellm)=
# LiteLLM

One judge, every provider. LiteLLM normalises a hundred-odd providers
onto a single call, so this is the shortest path to judging with
anything that is not OpenAI.

```bash
pip install "ragrank[litellm]"
```

```python
from ragrank import evaluate
from ragrank.integrations.litellm import LiteLLM

judge = LiteLLM(model="anthropic/claude-sonnet-4-5")
result = evaluate(dataset, llm=judge)
```

## Model strings

LiteLLM's own, provider-prefixed for anything other than OpenAI:

```python
LiteLLM(model="gpt-4o-mini")                    # OpenAI
LiteLLM(model="anthropic/claude-sonnet-4-5")    # Anthropic
LiteLLM(model="gemini/gemini-2.0-flash")        # Google
LiteLLM(model="groq/llama-3.3-70b-versatile")   # Groq
LiteLLM(model="bedrock/anthropic.claude-v2")    # AWS Bedrock
```

The credential lives wherever that provider expects it -
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and so on. Leaving `api_key`
unset lets LiteLLM read the usual environment variable, which is
normally what you want.

## Anything self-hosted

```python
LiteLLM(
    model="ollama/llama3",
    api_base="http://localhost:11434",
)
```

A local model costs nothing per call, which changes the arithmetic on
the expensive metrics considerably.

## Passing provider-specific options

```python
LiteLLM(
    model="azure/my-deployment",
    api_base="https://my-resource.openai.azure.com",
    extra_params={"api_version": "2024-02-01"},
)
```

Everything in `extra_params` is forwarded to `litellm.completion`.

## Judge settings

The usual `LLMConfig`, applied the same way as every other LLM in
ragrank:

```python
from ragrank.llm import LLMConfig

judge = LiteLLM(model="gemini/gemini-2.0-flash")
judge.set_config(LLMConfig(temperature=0.0, max_tokens=64))
```

Only options you actually set are forwarded. Providers differ in what
they accept, and several reject an explicit null `stop` - so ragrank
sends nothing rather than sending nothing-shaped.

## Which one to use

If you are already on OpenAI, [`OpenaiLLM`](../../evaluation/with_llm.md)
is one fewer dependency. For anything else, this is the wrapper to
reach for - and it is what to use rather than routing through the
LangChain adapter to get to a model LangChain also supports.
