"""Include all of the prompts for the ragrank"""

# ruff: noqa: E501
from ragrank.prompt.base import Prompt

RESPONSE_RELEVANCY_PROMPT = Prompt(
    name="Answer Relevancy",
    instructions="You are tasked with determining the relevancy of an answer given the context provided. Provide a score between 0 and 1, indicating how relevant the response is to the given question and context. You don't need to explain your score, just provide a single float value. Your response to this should be a number.",
    examples=[
        {
            "question": "How are you?",
            "context": [
                "Sarah slumped into the chair at the cafe, sighing heavily. 'That meeting dragged on forever,' she said to her friend Emily.",
            ],
            "response": "I am fine.",
            "relevancy": "0.99",
        },
        {
            "question": "How are you?",
            "context": [
                "John nervously tapped his foot in the doctor's waiting room, glancing at the clock every few minutes. He clutched the crumpled referral slip in his sweaty hand.",
            ],
            "response": "The weather is very bad.",
            "relevancy": "0.02",
        },
        {
            "question": "What is the capital of France?",
            "context": [
                "Ms. Rousseau pointed to a map of Europe hanging on the classroom wall. 'Today, class, we'll be learning about the capital cities of major European countries. Can anyone tell me the capital of France?'",
            ],
            "response": "The capital of France is Paris.",
            "relevancy": "0.95",
        },
    ],
    input_keys=["question", "context", "response"],
    output_key="relevancy",
)

RESPONSE_CONCISENESS_PROMPT = Prompt(
    name="Response Conciseness",
    instructions="Determine the conciseness of a response on a scale from 0 to 1. A score of 1 indicates a completely concise response, while a score of 0 signifies a response lacking conciseness. Provide only a single float value without explaining th score. Your response to this should be a number.",
    examples=[
        {
            "question": "What is the capital of France?",
            "context": [
                "During a conversation about European culture, Maria mentioned, 'I love French pastries, especially those from the capital.'",
            ],
            "response": "The capital of France is Paris.",
            "conciseness": 0.95,
        },
        {
            "question": "What is the main theme of 'To Kill a Mockingbird'?",
            "context": [
                "In a literature discussion, Sarah remarked, 'The central theme of 'To Kill a Mockingbird' is deeply rooted in social justice.'",
            ],
            "response": "Racial injustice and moral growth.",
            "conciseness": 0.9,
        },
        {
            "question": "What is the capital of Spain?",
            "context": [
                "While planning a trip to Europe, Mark exclaimed, 'I can't wait to visit the beautiful capital city known for its stunning architecture and vibrant culture.'",
            ],
            "response": "Barcelona.",
            "conciseness": 0.85,
        },
        {
            "question": "Describe the impact of the Industrial Revolution.",
            "context": [
                "During a history lesson, the teacher asked the students to describe the impact of the Industrial Revolution.",
            ],
            "response": "The Industrial Revolution had significant social, economic, and technological impacts, leading to urbanization, mass production, and changes in labor practices.",
            "conciseness": 0.6,
        },
    ],
    input_keys=["question", "context", "response"],
    output_key="conciseness",
)


CONTEXT_RELEVANCY_PROMPT = Prompt(
    name="Context Relevancy",
    instructions="You are tasked with determining the relevancy of a context provided for a given question. Provide a score between 0 and 1, indicating how relevant the context is to the given question. You don't need to explain your score, just provide a single float value. Your response to this should be a number.",
    examples=[
        {
            "question": "How do I create a neural network using TensorFlow?",
            "context": [
                "As a beginner in machine learning, I'm interested in learning how to build neural networks. I've heard TensorFlow is a popular library for this purpose."
                "I've already installed TensorFlow and I'm ready to start coding."
            ],
            "response": "To create a neural network using TensorFlow, you can start by importing the necessary modules and defining the layers of your model. Then, compile the model specifying the optimizer, loss function, and metrics. Finally, train the model using the fit method on your training data.",
            "relevancy": 0.95,
        },
        {
            "question": "What are some common interview questions for data science positions?",
            "context": [
                "I'm currently preparing for data science job interviews and I want to make sure I'm ready for common questions."
                "I've already gone through some online resources, but I'm looking for more insights."
            ],
            "response": "Common interview questions for data science positions often include topics like machine learning algorithms, data cleaning techniques, and statistical analysis methods. You may also be asked to explain your past projects and how you handled specific challenges.",
            "relevancy": 0.92,
        },
        {
            "question": "What's the best restaurant in town?",
            "context": [
                "I'm planning a romantic dinner for my anniversary and I want to impress my partner with the perfect restaurant."
                "I've been searching online for reviews, but I'm still unsure where to go."
            ],
            "response": "I don't know",
            "relevancy": 0.05,
        },
        {
            "question": "How do I implement gradient descent in Python?",
            "context": [
                "I'm currently studying optimization algorithms in machine learning and I want to understand how gradient descent works in Python."
                "I've already reviewed the theory behind gradient descent and now I'm looking for practical examples."
            ],
            "response": "To implement gradient descent in Python, you can start by defining a cost function and its gradient. Then, initialize the parameters and update them iteratively using the gradient descent formula. Finally, monitor the convergence of the algorithm and adjust hyperparameters as needed.",
            "relevancy": 0.96,
        },
    ],
    input_keys=["question", "context", "response"],
    output_key="relevancy",
)

CONTEXT_UTILIZATION_PROMPT = Prompt(
    name="Context Utilization",
    instructions="Determine the utilization of context with respect to the response on a scale from 0 to 1. A score of 1 indicates optimal utilization of context, while a score of 0 signifies poor utilization. Provide only a single float value without explaining the score. Your response to this should be a number.",
    examples=[
        {
            "question": "What is the capital of France?",
            "context": [
                "During a conversation about European culture, Maria mentioned, 'I love French pastries, especially those from the capital.'",
            ],
            "response": "The capital of France is Paris.",
            "utilization": 0.95,
        },
        {
            "question": "Describe the impact of the Industrial Revolution.",
            "context": [
                "During a history lesson, the teacher asked the students to describe the impact of the Industrial Revolution.",
            ],
            "response": "The Industrial Revolution had significant social, economic, and technological impacts, leading to urbanization, mass production, and changes in labor practices.",
            "utilization": 0.6,
        },
        {
            "question": "What is the main theme of 'To Kill a Mockingbird'?",
            "context": [
                "In a literature discussion, Sarah remarked, 'The central theme of 'To Kill a Mockingbird' is deeply rooted in social justice.'",
            ],
            "response": "Racial injustice and moral growth.",
            "utilization": 0.9,
        },
        {
            "question": "What's the best way to learn a new language?",
            "context": [
                "During a language learning workshop, John shared his experience, saying, 'When I traveled to Spain, immersing myself in the local culture and conversing with native speakers helped me improve my language skills significantly.'",
            ],
            "response": "Immersing yourself in the local culture and conversing with native speakers is one of the most effective ways to learn a new language.",
            "utilization": 0.98,
        },
    ],
    input_keys=["question", "context", "response"],
    output_key="utilization",
)
