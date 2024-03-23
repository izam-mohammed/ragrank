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

```{toctree}
:hidden:

custom_metric
custom_instruct

```