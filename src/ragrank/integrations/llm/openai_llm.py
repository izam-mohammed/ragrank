"""Module for integrating OpenAI language model"""

from time import time

from ragrank.llm import BaseLLM, LLMResult
from ragrank.utils.llm import get_env_var

try:
    from openai import OpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install `openai` module by `pip install openai`"
    ) from None


class OpenaiLLM(BaseLLM):
    """Represents an OpenAI Language Model (LLM) for generating text.

    This class provides methods to interact with the OpenAI language model
    to generate text based on user input.

    Attributes:
        name (str): The name of the language model.

    Methods:
        generate_text: Generates text using the OpenAI language model.

    """

    @property
    def name(self) -> str:
        """Get the name of the language model"""
        return "OpenAI LLM"

    def generate_text(
        self,
        text: str,
        llm_name: str = "gpt-3.5-turbo",
        initial_msg: str = "You are a helpful assistant",
    ) -> LLMResult:
        """
        Generate text using the OpenAI language model.

        Args:
            text (str): The user's input text.
            llm_name (str, optional): Name of the language model.
                Defaults to "gpt-3.5-turbo".
            initial_msg (str, optional): Initial message for the conversation.
                Defaults to "You are a helpful assistant".

        Returns:
            LLMResult: Result containing the generated text and other metadata.
        """
        start_time = time()
        client = OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model=llm_name,
            messages=[
                {"role": "system", "content": initial_msg},
                {"role": "user", "content": text},
            ],
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            seed=self.llm_config.seed,
            top_p=self.llm_config.top_p,
            stop=self.llm_config.stop,
        )

        if not completion.choices:
            raise ValueError("Unable to generate the output")

        message: str = completion.choices[0].message.content
        finish_reason: str = completion.choices[0].finish_reason
        response_tokens: int = completion.usage.completion_tokens
        response_time = time() - start_time

        response = LLMResult(
            response=message,
            response_time=response_time,
            llm=self,
            llm_config=self.llm_config,
            finish_reason=finish_reason,
            response_tokens=response_tokens,
        )
        return response
