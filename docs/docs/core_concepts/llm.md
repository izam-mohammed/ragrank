(what-is-llm)=
# LLM

## Large Language Models

LLMs are like smart autocomplete tools. You input some text, and they predict what might come next. They're great for generating ideas, overcoming writer's block, and even assisting coders in writing code faster. [GitHub Copilot](https://github.com/features/copilot) is a prime example of how powerful LLMs can be.

## How LLM work

- LLMs use [deep learning](https://en.wikipedia.org/wiki/Deep_learning) techniques and vast textual data, often based on [transformer architecture](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) like GPT.
- They consist of multiple layers of neural networks, including an [attention mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning)).
- During training, they predict the next word in a sentence based on preceding context, using [tokenization](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) and [embeddings](https://en.wikipedia.org/wiki/Word_embedding).
- Training involves massive corpora of text to learn grammar, semantics, and relationships.
- Once trained, LLMs autonomously generate text based on input and acquired patterns.
- [Prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering), tuning, and [fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)) enhance model performance and mitigate biases and errors.
- These tactics ensure enterprise-grade LLMs are suitable for various [NLU](https://en.wikipedia.org/wiki/Natural-language_understanding) and content generation tasks without causing unwanted consequences.


## Local LLMs

Most people use LLMs through [OpenAI's API](https://openai.com/blog/openai-api) because it's easier than deploying them themselves. But when you're not using a hosted LLM, things get more complicated. You have to decide which LLM to use and think about the computing power needed.

The main issues with deploying your own LLM are cost and speed. For good quality, you usually need a big model with 7 billion parameters or more, which requires a lot of memory, preferably on a GPU to avoid slow generation times.

To make models smaller, you can use techniques like quantization, which compresses the model. This can reduce memory usage from about 28GB to 4GB, but it might affect accuracy and speed, especially with certain methods like [bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes). However, if GPU memory is limited, quantization can be helpful.

After deciding whether to use quantization, you need to deploy the LLM locally. There are services designed specifically for this, like [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated), [Anyscale Endpoints](https://www.anyscale.com/blog/anyscale-endpoints-fast-and-scalable-llm-apis), or [AWS SageMaker](https://aws.amazon.com/sagemaker/).

## LLM use cases 

LLMs are reshaping many business processes and showing their adaptability across a wide range of tasks and industries.

Here is a list of some of the most important areas where LLMs benefit organizations:

- **Text Generation**: LLMs can write emails, blog posts, and other content based on prompts, like the retrieval-augmented generation (RAG) technique.
- **Content Summarization**: LLMs can condense long articles, reports, and documents into shorter, tailored texts.
- **AI Assistants**: LLMs function as chatbots, handling customer queries, tasks, and providing information in natural language for customer support.
- **Code Generation**: LLMs help developers create applications, detect errors, and identify security issues in different programming languages.
- **Sentiment Analysis**: LLMs analyze text to understand customer feedback and manage brand reputation by determining the tone.
- **Language Translation**: LLMs offer language translation services, facilitating communication across different languages and regions.


## A Note on Hallucination

LLMs face significant challenges, with one prominent issue being hallucination.

![](https://cdn.sanity.io/images/vr8gru94/production/e16a232e13c17c9b2934b65c486914922208d1b1-2404x1208.png)

We'll explore various components that can reduce hallucination and enhance result quality. While complete elimination of hallucinations may not be possible, leveraging these components can bring us close to achieving that goal.

> Understand the [RAG models](./rag_model.md) and how they minimize common issues in LLM.
