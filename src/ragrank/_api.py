from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class TraceType(Enum):
    LLM = "LLM"
    RETRIEVER = "Retriever"
    EMBEDDING = "Embedding"
    TOOL = "Tool"
    AGENT = "Agent"
    CHAIN = "Chain"


class TraceStatus(Enum):
    SUCCESS = "Success"
    ERROR = "Error"


@dataclass
class LLMMetadata:
    model: str


@dataclass
class EmbeddingMetadata:
    model: str


@dataclass
class Event:
    name: str


@dataclass
class BaseTrace:
    type: TraceType | str
    executionTime: float
    name: str
    input: dict
    output: dict
    status: TraceStatus
    traces: list[TraceData]


@dataclass
class LlmTrace(BaseTrace):
    input: str
    llmMetadata: LLMMetadata = None


@dataclass
class EmbeddingTrace(BaseTrace):
    embeddingMetadata: EmbeddingMetadata


@dataclass
class GenericTrace(BaseTrace):
    type: str


TraceData = Union[LlmTrace, EmbeddingTrace, GenericTrace]


class TraceManager:
    def __init__(self) -> None: ...


def trace(event: Event) -> None: ...
