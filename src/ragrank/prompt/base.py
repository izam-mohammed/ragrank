"""base fo the the prompt module"""

from __future__ import annotations

from typing import Any

from ragrank.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

Example = dict[str, Any]


class Prompt(BaseModel):
    """Represents a prompt for the Language Learning Model (LLM).

    Attributes:
        name (str): The name of the prompt.
        instructions (str): The instructions for the prompt.
        examples (List[Example]): List of example inputs and outputs.
        input_keys (List[str]): List of input keys.
        output_key (str): Key for the output.
    """

    model_config: ConfigDict = ConfigDict(frozen=True)

    name: str = Field(description="The name of the prompt.")
    instructions: str = Field(
        repr=False, description="The instructions for the prompt."
    )
    examples: list[Example] = Field(
        repr=False,
        default_factory=list,
        description="List of example inputs and outputs.",
    )
    input_keys: list[str] = Field(
        repr=False, description="List of input keys."
    )
    output_key: str = Field(
        repr=False, description="Key for the output."
    )

    @model_validator(mode="after")
    def validate_prompt(self) -> Prompt:
        """Validate the prompt using Pydantic.

        Raises:
            ValueError: If input keys are empty or
                example keys do not match input and output keys.
        """
        if not self.input_keys:
            raise ValueError(
                "Input keys cannot be empty. \n"
                "Please provide non-empty input keys."
            )

        keys = self.input_keys + [self.output_key]
        for example in self.examples:
            if [*example] != keys:
                raise ValueError(
                    "The keys should match with the example. \n"
                    "Please ensure the example keys match with"
                    " the input and output keys."
                )
        return self

    def to_string(self) -> str:
        """Convert the prompt to a string representation.

        Returns:
            str: String representation of the prompt.
        """

        prompt_str = self.name + "\n\n"
        prompt_str += self.instructions + "\n"

        for example in self.examples:
            for key, value in example.items():
                prompt_str += f"\n{key}: {value}"
            prompt_str += "\n"

        if self.input_keys:
            prompt_str += "".join(
                f"\n{key}: {{{key}}}" for key in self.input_keys
            )

        if self.output_key:
            prompt_str += f"\n{self.output_key}:\n"

        return prompt_str

    def get_examples(self, example_no: int | None = None) -> str:
        """Retrieve examples from the prompt.

        Args:
            example_no (int | None): The number of examples to retrieve.

        Returns:
            List[Example]: List of example inputs and outputs.

        Raises:
            IndexError: If example number is out of range.
        """

        if example_no is None:
            return self.examples
        if example_no > len(self.examples):
            raise IndexError("Example Number is out of range")
        return self.examples[0:example_no]

    def save(self) -> None:
        """Save the object locally"""
        raise NotImplementedError

    def load(self) -> None:
        """Load the saved object from the system"""
        raise NotImplementedError
