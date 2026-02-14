"""Custom Instruct Metric"""

from __future__ import annotations

import logging
from time import time
from typing import Any

from ragrank.bridge.pydantic import BaseModel, ConfigDict, Field
from ragrank.dataset import DataNode
from ragrank.llm import BaseLLM, default_llm
from ragrank.metric.base import BaseMetric, MetricResult, MetricType
from ragrank.prompt import Prompt
from ragrank.prompt._prompts import (
    BINARY_PROMPT_ADDON,
    NON_BINARY_PROMPT_ADDON,
    NONE_PROMPT,
)
from ragrank.prompt.base import Example as PromptExample

logger = logging.getLogger(__name__)


class InstructConfig(BaseModel):
    """
    Configuration settings for the CustomInstruct metric.

    Attributes:
        metric_type (MetricType): The type of the metric.
        name (str): The name of the Instruct metric.
        instructions (str): Instructions for the metric.
        examples (List[PromptExample]): Examples for the Instruction metric.
        input_fields (List[str]): Input fields for the Instruct Metric.
        output_field (str): Output field for the Instruct Metric.

    Methods:
        to_prompt() -> Prompt: get a Prompt object from the given
            configuration.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True,
    )

    metric_type: MetricType = Field(
        description="The type of the metric"
    )
    name: str = Field(description="The name of the Instruct metric")
    instructions: str = Field(
        description="Instructions for the metric"
    )
    examples: list[PromptExample] = Field(
        description="Examples for the Instruction metric",
        default_factory=list,
    )
    input_fields: list[str] = Field(
        description="Input fields for the Instruct Metric"
    )
    output_field: str = Field(
        description="ouput field for the Instruct Metric",
        default="output",
    )

    def to_prompt(self) -> Prompt:
        """Convert the configuration to a Prompt object.

        Returns:
            Prompt: The generated prompt object.
        """
        return Prompt(
            name=self.name,
            instructions=self.instructions,
            examples=self.examples,
            input_keys=self.input_fields,
            output_key=self.output_field,
        )

    def __repr__(self) -> str:
        """Representation of the InstructConfig object.

        Returns:
            str: String representation of the object.
        """
        return f"Instruct Config - {self.name}"


class CustomInstruct(BaseMetric):
    """
    A custom metric for evaluating responses based on instructions.

    Attributes:
        llm (BaseLLM): The language model used to generate the response.
        prompt (Prompt): The prompt provided for generating the response.
        config (InstructConfig): Instructions and configuration for the metric.

    Methods:
        name(self) -> str:
            Get the name for the metric.

        score(self, data: DataNode) -> MetricResult:
            Calculate the score for the custom metric based
            on the provided data.

        _reason(self, data: DataNode, score: float) -> str | None:
            Determine the reason for the given score. (Not implemented yet)
    """

    metric_type: MetricType = Field(
        default_factory=lambda: MetricType.NON_BINARY,
        description="The type of metric, which is non-binary.",
    )
    llm: BaseLLM = Field(
        default_factory=lambda: default_llm(),
        description="The language model used to generate the response.",
    )
    prompt: Prompt = Field(
        default_factory=lambda: NONE_PROMPT,
        description="The prompt provided for generating the response",
    )
    config: InstructConfig = Field(
        description="Instructions for the Custom Instruct Metric"
    )

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Post-initialization hook to configure the metric."""
        self.metric_type = self.config.metric_type
        self.prompt = Prompt(
            name=self.config.name,
            instructions=(
                self.config.instructions + BINARY_PROMPT_ADDON
                if self.config.metric_type == MetricType.BINARY
                else NON_BINARY_PROMPT_ADDON
            ),
            examples=self.config.examples,
            input_keys=self.config.input_fields,
            output_key=self.config.output_field,
        )

    @property
    def name(self) -> str:
        """Get the name for the metric.

        Returns:
            str: The name of the metric.
        """

        return f"Custom Instruct - {self.config.name}"

    def score(self, data: DataNode) -> MetricResult:
        """Calculate the score for the custom metric based
        on the provided data.

        Args:
            data (DataNode): The input data to be used for scoring.

        Returns:
            MetricResult: The result of the metric calculation.
        """

        tm = time()
        prompt_str = self.prompt.to_string()
        prompt_dt = prompt_str.format(**data.model_dump())
        response = self.llm.generate_text(
            prompt_dt,
        )
        try:
            score = float(response.response)
        except ValueError:
            logger.error(
                f"Got unexpected LLM response - '{response.response}'"
            )
            raise ValueError(
                "Got unexpected response from the LLM"
            ) from ValueError
        delta = tm - time()
        return MetricResult(
            datanode=data,
            metric=self,
            score=score,
            reason=None,
            process_time=delta,
        )

    def _reason(self, data: DataNode, score: float) -> str | None:
        """Determine the reason for the given score.
        Not implemented yet.

        Args:
            data (DataNode): The input data used for scoring.
            score (float): The score indicating the
                relevancy of the response.

        Returns:
            str: The reason for the given score.
        """

        raise NotImplementedError
