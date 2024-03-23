(core-concepts)=
# 🪩 Core Concepts

**A Bit History**

- At the 2017 NeurIPS conference, Google researchers introduced the transformer architecture in their landmark paper ["Attention Is All You Need"](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need).
- The following year in 2018, [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) was introduced and quickly became famous in the tech world.
- Although decoder-only GPT-1 was introduced in 2018, it was GPT-2 in 2019 that caught widespread attention. After that they released GPT-3 and GPT-4.
- Since 2022, source-available models have been gaining popularity, especially at first with BLOOM and LLaMA, though both have restrictions on the field of use. 
- Througout the time, LLM have a lot of issues such as [Hallucination](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)), Low quality response and many more. That's when The RAG models, LLM observability and monitoring came into the picture.

Understand the importance of evaluation and monitoring of the [LLM](./llm.md) and [RAG](./rag_model.md).

```{admonition} Please Note
:class: note

The content discussed in this blog regarding LLM (Large language model) and RAG (Retrieval-augmented generation) covers only the fundamental concepts. For a comprehensive understanding and advanced insights, it is recommended to consult additional resources.
```

-------

````{grid}
:gutter: 2

```{grid-item-card} 🌟 LLM
:link: what-is-llm
:link-type: ref

What and why use Large Language Models (LLM) for various business usecases. Learn more.
```

```{grid-item-card} ♻️ RAG model
:link: rag-model
:link-type: ref

RAG (Retrieval Augmented Generation) is a framework that boosts the accuracy of GenAI. Learn more.
```
````

````{grid}
:gutter: 2

```{grid-item-card} 📏 Evaluation
:link: why-evaluation
:link-type: ref

There are multiple customizations available in the evaluation process. Learn more
```

```{grid-item-card} 🎯 Why ragrank
:link: why-ragrank
:link-type: ref

What and why we build ragrank, what is the motive and intentions. Learn more.
```
````


```{toctree}
:hidden:

llm
rag_model
evaluation
why_ragrank
```