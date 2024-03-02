"""All of the contants for ragrank"""

from typing import Final, List

KEYFILE: Final[str] = ".ragrank"
DEFAULT_EMBEDDING_DIMENTION: Final[int] = 64
DEFAULT_LLM_MODEL: Final[str] = "openai-3.5"
DEFAULT_METRIC: Final[str] = "relevancy"

QUESTION_FIELD: str = "question"
CONTEXT_FIELD: str = "context"
RESPONSE_FIELD: str = "response"
DATA_FIELDS: List[str] = [
    QUESTION_FIELD,
    CONTEXT_FIELD,
    RESPONSE_FIELD,
]
