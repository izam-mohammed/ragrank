>This repo is under development ❄️

<p align="center">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Auto-Playground/Ragrank/main/docs/docs/_static/imgs/ragrank_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Auto-Playground/Ragrank/main/docs/docs/_static/imgs/ragrank_light.png">
    <img alt="Hashnode logo" src="https://raw.githubusercontent.com/Auto-Playground/Ragrank/main/docs/docs/_static/imgs/ragrank_light.png" height="130">
    </picture>

</p>

<p align="center">
    <a href="">
        <img alt="GitHub" src="https://img.shields.io/github/license/Auto-Playground/ragrank">
    </a>
    <a href="https://pypi.org/project/ragrank/">
        <img alt="Build" src="https://img.shields.io/pypi/pyversions/ragrank">
    </a>
    <a href="https://ragrank.readthedocs.io/latest/">
        <img alt="GitHub" src="https://img.shields.io/readthedocs/ragrank">
    </a>
    <a href="https://pypi.org/project/ragrank/">
        <img alt="GitHub" src="https://img.shields.io/github/v/release/Auto-Playground/Ragrank?color=orange">
    </a>
    <a href="https://github.com/Auto-Playground/Ragrank/actions">
        <img alt="GitHub" src="https://img.shields.io/github/actions/workflow/status/Auto-Playground/ragrank/.github%2Fworkflows%2Ftests.yml">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://ragrank.readthedocs.io/latest/">Documentation</a> |
        <a href="https://api-ragrank.readthedocs.io/">API reference</a> |
        <a href="https://ragrank.readthedocs.io/latest/get_started/basic_evaluation.html">Quickstart</a> |
        <a href="https://discord.gg/zDzM5hFS">Join the Community</a> |
        <a href="https://pypi.org/project/ragrank/">Pypi</a> |
    <p>
</h4>

Welcome to Ragrank! This toolkit is designed to assist you in evaluating the performance of your Retrieval-Augmented Generation (RAG) applications. You will get proper metrics for evaluate RAG model. The product is still in `beta` stage.

## 🔥 Installation

Ragrank is available as a PyPi package. To install it, simply run:

```bash
pip install ragrank
```

If you prefer to install it from the source:

```bash
git clone https://github.com/Auto-Playground/ragrank.git && cd ragrank
uv sync
```

## 🚀 Quick Start

Set your `OPENAI_API_KEY` as an environment variable (you can also evaluate using your own custom model, refer [docs](https://ragrank.readthedocs.io/)):
```bash
export OPENAI_API_KEY="..."
```

Here's a quick example of how you can use Ragrank to evaluate the relevance of generated responses:

```python
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.metric import response_relevancy

# Define your dataset
data = from_dict({
    "question": "What is the capital of France?",
    "context": ["France is famous for its iconic landmarks such as the Eiffel Tower and its rich culinary tradition."],
    "response": "The capital of France is Paris.",
})

# Evaluate the response relevance metric
result = evaluate(data, metrics=[response_relevancy])

# Display the evaluation results
result.to_dataframe()
```

For more information on how to use Ragrank and its various features, please refer to the [documentation](https://ragrank.readthedocs.io/). 📚

## License

This project is licensed under the [Apache License](https://github.com/Auto-Playground/Ragrank/blob/main/LICENSE). Feel free to use and modify it according to your needs.

## Feedback and Support

If you encounter any issues, have questions, or would like to provide feedback, please don't hesitate to open an issue on the GitHub repository. Your contributions and suggestions are highly appreciated!

Join our community on Discord to connect with other users, ask questions, and share your experiences with Ragrank. We're here to help you make the most out of your NLP projects! 💬

> Happy evaluating! 🙂
