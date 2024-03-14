"""base fo the the prompt module"""

from typing import Any, Dict, List, Optional

from ragrank.bridge.pydantic import BaseModel, model_validator

Example = Dict[str, Any]


class Prompt(BaseModel):
    """Prompt for the LLM"""

    name: str
    instructions: str
    examples: List[Example] = []
    input_keys: List[str]
    output_key: str

    @model_validator(mode="after")
    def validate_prompt(self) -> "Prompt":
        """Pydantic validation function"""

        if not self.input_keys:
            raise ValueError("Input keys Cannot be empty")

        keys = self.input_keys + [self.output_key]
        for example in self.examples:
            if [*example] != keys:
                raise ValueError("The keys should match with the example")
        return self

    def to_string(self) -> str:
        """Convert the Prompt to a string"""

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

    def get_examples(self, example_no: Optional[int] = None) -> str:
        """Return the examples in the class"""

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
