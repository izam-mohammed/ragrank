"""Include all of the prompts for the ragrank"""

# ruff: noqa: E501

from ragrank.prompt.base import Prompt

RESPONSE_RELEVANCY_PROMPT = Prompt(
    name="Answer Relevancy",
    instructions="You are tasked with determining the relevancy of an answer given the context provided. Provide a score between 0 and 1, indicating how relevant the response is to the given question and context. You don't need to explain your score, just provide a single float value.",  # noqa: E501
    examples=[
        {
            "question": "How are you?",
            "context": [
                "Sarah slumped into the chair at the cafe, sighing heavily. 'That meeting dragged on forever,' she said to her friend Emily."  # noqa: E501
            ],
            "response": "I am fine.",
            "relevancy": "0.99",
        },
        {
            "question": "How are you?",
            "context": [
                "John nervously tapped his foot in the doctor's waiting room, glancing at the clock every few minutes. He clutched the crumpled referral slip in his sweaty hand."  # noqa: E501
            ],
            "response": "The weather is very bad.",
            "relevancy": "0.02",
        },
        {
            "question": "What is the capital of France?",
            "context": [
                "Ms. Rousseau pointed to a map of Europe hanging on the classroom wall. 'Today, class, we'll be learning about the capital cities of major European countries. Can anyone tell me the capital of France?'"  # noqa: E501
            ],
            "response": "The capital of France is Paris.",
            "relevancy": "0.95",
        },
    ],
    input_keys=["question", "context", "response"],
    output_key="relevancy",
)
