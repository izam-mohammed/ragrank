(the-evaluation)=

# 📏 Evaluation 

There are multiple customizations available in the evaluation process. Feel free explore everything.

> See the [Detailed Evaluation](./detailed_evaluation.md) to get an idea about the detailed process.

````{grid}
:gutter: 2

```{grid-item-card} 🚀 With Custom Metrics
:link: with-custom-metrics
:link-type: ref

We have a lot of built-in metrics, and you can also create your own metrics. This tutorial will help you use all of the available metrics and create your custom metrics.
```

```{grid-item-card} 📁 With Custom Data
:link: data-ingestion
:link-type: ref

There are primarily two types of data objects available in Ragrank. These are `DataNode`, which carries a single data point, and `Dataset`, which carries multiple data points. Explore here.
```

````

````{grid}
:gutter: 2

```{grid-item-card} 🔥 With Custom LLM
:link: with-custom-llm
:link-type: ref

You can use various LLMs if you need to. Additionally, you can create a new `LLM` object and use that as well. By default, we are using OpenAI LLM for internal operations.
```

```{grid-item-card} 🧮 With Custom Embedding
:link: with-custom-embedding
:link-type: ref

We are using embedding models internally to calculate the scores. Here, you can explore the customization options available for the embedding object.
```

````

```{toctree}
:hidden:

detailed_evaluation
data_ingestion
with_metrics
with_llm
with_embedding
result
```
