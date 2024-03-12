"""Openai module for the integrations"""

from __future__ import annotations

from ragrank.llm import BaseLLM, LLMResult
from ragrank.utils.llm import get_env_var

try:
    from openai import OpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install `openai` module by `pip install openai`"
    ) from None


class OpenaiLLM(BaseLLM):
    """OpenAI LLM class"""

    def generate_text(
        self,
        text: str,
        llm_name: str = "gpt-3.5-turbo",
        initial_msg: str = "You are a helpful assistant",
    ) -> LLMResult:
        """Generate the text abstract method"""

        client = OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model=llm_name,
            messages=[
                {"role": "system", "content": initial_msg},
                {"role": "user", "content": text},
            ],
        )
        message: str = completion.choices[0].message.content
        response: LLMResult = LLMResult(
            response=message,
            response_time=0.1,
            llm=self,
            llm_config=None,
        )
        return response
