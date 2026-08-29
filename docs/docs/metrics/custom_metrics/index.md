# Custom Metrics

Ragrank allow to build custom metrics to evaluate your rag models.

````{grid}
:gutter: 2

```{grid-item-card} ⛏ Custom Metric
:link: custom-metric
:link-type: ref

You can create custom metrics by inheriting the `CustomMetric` class. You'll need to implement three methods for your metric.
```

```{grid-item-card} ⛏ Custom Instruct
:link: custom-instruct
:link-type: ref

Differs from predefined metrics. Initialize with config before use. Allows custom evaluation instructions as prompts.
```

````

````{grid}
:gutter: 2

```{grid-item-card} ⚡ Metrics in one expression
:link: simple-metrics
:link-type: ref

A decorator, a rubric-based judge, or a rule in plain English. Three ways to define a metric, none of which need a class.
```

```{grid-item-card} 🎲 When one judge is not enough
:link: jury-and-pairwise
:link-type: ref

Judges are noisy. Poll a committee, compare instead of scoring, or repeat and measure the spread.
```

````

```{toctree}
:hidden:

simple_metrics
jury_and_pairwise
custom_metric
custom_instruct

```