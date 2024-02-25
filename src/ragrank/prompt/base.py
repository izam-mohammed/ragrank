from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class PromptType(Enum):
    RAG = "rag"
    LLM = "llm"
    single = "single"


@dataclass
class Prompt:
    prompt_type:PromptType = PromptType.RAG
    
    def get_json(self):
        ...

    def save(self):
        ...
    
    def from_string(self, string:str):
        ...

    def from_file(self, file_path:Path):
        ...

