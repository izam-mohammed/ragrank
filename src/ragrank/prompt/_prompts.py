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

BINARY_PROMPT_ADDON = (
    "Output only 0 or 1. Do not include any explanation."
)
NON_BINARY_PROMPT_ADDON = (
    "Output only a single float between 0.0 and 1.0."
    " Do not include any explanation."
)
