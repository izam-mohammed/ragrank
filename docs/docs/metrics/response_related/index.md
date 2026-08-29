# Response Related

Here are the metrics related to response of the RAG model


````{grid}
:gutter: 2

```{grid-item-card} 💡 Faithfulness
:link: faithfulness
:link-type: ref

Is the model making things up? Splits the response into claims and checks each against the retrieved context, so a bad score points at the sentence that caused it.
```

```{grid-item-card} 💡 Correctness
:link: correctness
:link-type: ref

Is the answer right, judged against a reference you already know? Tolerates differences in wording that string comparison would punish.
```

````

````{grid}
:gutter: 2

```{grid-item-card} 💡 Response Relevancy
:link: response-relevancy
:link-type: ref

This evaluates how much the answer matches what is needed, showing how appropriate it is.
```

```{grid-item-card} 💡 Response Conciseness
:link: response-conciseness
:link-type: ref

This evaluates the conciseness of a response, ensuring it's short yet contains as much relevant information as possible.
```

````

```{toctree}
:hidden:

faithfulness
correctness
response_relevancy
response_conciseness
```