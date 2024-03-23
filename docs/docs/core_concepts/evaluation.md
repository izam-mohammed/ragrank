(why-evaluation)=
# Evaluation

Creating a simple [RAG (Retrieval Augmented Generation)](./rag_model.md) system typically doesn't require a ton of time. However, enhancing the accuracy of its output, like with any machine learning model, is the real challenge. This involves ensuring that the output from the RAG system:

- Contains high-quality content that is coherent and factually accurate.
- Is relevant and comprehensive.
- Is free from excessive noise.
- Is not harmful, malicious, or toxic.
- Achieves good performance in terms of speed.

Enhancing these aspects requires continuous refinement and optimization of the model, which can be a more time-consuming task compared to the initial development phase.

## Making RAG better

Before we talk about making RAG systems better, let's first understand how we measure them. In machine learning, we use clear numbers like Gini, Precision, Recall, and F1 score to evaluate performance. But RAG systems deal with unstructured text, so we need a mix of qualitative and quantitative measures to assess them.

It's important to remember that evaluating Large Language Models (LLMs) is different from assessing RAG systems. RAG evaluation involves looking at how well it retrieves and generates responses based on given text. LLM evaluation, on the other hand, looks at a broader range of uses and metrics. Some frameworks for evaluating LLMs include [HELM](https://github.com/stanford-crfm/helm), [OpenAI/eval](https://github.com/openai/evals), and [Alpaca Evaluation](https://github.com/tatsu-lab/alpaca_eval). But in here, we're only focusing on evaluating RAG systems.

## Metric Driven Development

Creating a basic LLM application might be easy, but keeping it up-to-date and making it better over time is the real challenge. we wants to improve LLM and RAG applications continuously by using Metrics-Driven Development (MDD).

MDD is a way of developing products that relies on data to make smart decisions. With MDD, you keep an eye on important metrics over time to understand how well your application is doing.

Our goal is to set a standard for using MDD in LLM and RAG applications that anyone can use and contribute to.

- **Evaluation**: This lets you check how well your LLM applications are doing and try out changes in a way that's guided by metrics, so you can trust that your results are accurate and repeatable.

- **Monitoring**: It helps you learn from the data your application generates in the real world, so you can keep making your LLM application better and better.

> Okay, then why do I choose ragrank for this ? 

Le't discuss [Why Ragrank](./why_ragrank.md)