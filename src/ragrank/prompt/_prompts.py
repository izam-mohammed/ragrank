"""Private Module that include all of the prompts for the ragrank"""

# ruff: noqa: E501
from ragrank.prompt.base import Prompt

NONE_PROMPT = Prompt(
    name="None Prompt",
    instructions="",
    examples=[{"input": "", "output": ""}],
    input_keys=["input"],
    output_key="output",
)

BINARY_PROMPT_ADDON = "Provide a score should either 0 or 1, indicating the metric for given question and context. You don't need to explain your score, just provide a single float value. Your response to this should be 0 or 1."
NON_BINARY_PROMPT_ADDON = "Provide a score between 0 and 1, indicating the metric for given question and context. You don't need to explain your score, just provide a single float value. Your response to this should be a number."

RESPONSE_RELEVANCY_PROMPT = Prompt(
    name="Answer Relevancy",
    instructions="You are tasked with determining the relevancy of an answer given the context provided. Provide a score between 0 and 1, indicating how relevant the response is to the given question and context. You don't need to explain your score, just provide a single float value. Your response to this should be a number.",
    examples=[
        {
            "question": "How do plants obtain energy?",
            "context": [
                "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. The process occurs in two stages: the light-dependent reactions and the light-independent reactions. In the light-dependent reactions, light energy is converted into chemical energy, while in the light-independent reactions (Calvin cycle), carbon dioxide and water are converted into glucose.",
            ],
            "response": "Plants obtain energy through a process called photosynthesis, where they convert sunlight into chemical energy.",
            "relevancy": 0.90,
        },
        {
            "question": "How do plants obtain energy?",
            "context": [
                "Gardening enthusiasts often wonder about the mechanisms behind plant growth and nutrition. One of the key processes involved is photosynthesis, where plants convert light energy into chemical energy through the utilization of pigments like chlorophyll.",
            ],
            "response": "The earthworms in the soil help aerate the soil, making it easier for plants to absorb nutrients.",
            "relevancy": 0.15,
        },
        {
            "question": "What are some common programming languages?",
            "context": [
                "In the rapidly evolving field of technology, knowing the right programming languages can make a significant difference in career opportunities. Some widely used programming languages include Python, Java, and JavaScript, each with its unique features and applications.",
            ],
            "response": "Some common programming languages include Python, Java, and JavaScript.",
            "relevancy": 0.95,
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
