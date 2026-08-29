(cost-and-tokens)=
# 💰 Cost and Tokens

Every judged metric call goes through a language model, and every one of
them has a price. Ragrank counts them.

```python
result = evaluate(data, metrics=[faithfulness, response_relevancy])

result.usage
# 240 calls, 61,204 tokens (59,880 in / 1,324 out)
```

## Estimating the spend

```python
result.cost(
    per_prompt_token=0.15 / 1e6,
    per_response_token=0.60 / 1e6,
)
# 0.0097752
```

Rates are per single token, so a price of $0.15 per million input
tokens is `0.15 / 1e6`.

No price table ships with Ragrank. Prices change, they differ between
providers, and they differ between accounts, so you pass the rates you
are actually paying rather than trusting a number the library guessed
months ago.

## What is counted

Every call, wherever the model came from: passed to `evaluate()`, set
on the metric, or the library default. Multi-call metrics are counted
per call rather than per row, so faithfulness over 2 rows with 2 claims
each reports 6 calls, because that is what happened.

Deterministic metrics make no calls at all, so a run using only free
metrics reports zero and needs no credentials.

## When the numbers are incomplete

Not every provider reports token usage. Most non-OpenAI LangChain
models return nothing, and Ragrank will not invent it:

```python
result.usage.is_complete       # False
result.usage.unreported_calls  # 12
```

When `is_complete` is `False`, the totals are a lower bound rather than
an answer. Better to say "at least this much" than to quietly
under-report.

## Reducing it

Three levers, roughly in order of how much they help:

1. Use [free metrics](../metrics/heuristic_metrics/index.md) for
   anything they can answer.
2. Turn on [caching](./caching.md), so a re-run costs nothing.
3. Cap `max_claims` on faithfulness for long responses.
