"""All of the contants for ragrank"""


KEYFILE: str = ".ragrank"
DEFAULT_EMBEDDING_DIMENTION: int = 64
DEFAULT_LLM_MODEL: str = "openai-3.5"
DEFAULT_METRIC: str = "relevancy"

QUESTION_FIELD: str = "question"
CONTEXT_FIELD: str = "context"
RESPONSE_FIELD: str = "response"
DATA_FIELDS: list[str] = [
    QUESTION_FIELD,
    CONTEXT_FIELD,
    RESPONSE_FIELD,
]

SERVER_URL: str = "https://ragrank-trace.onrender.com/api/"
DEBUG_MODE: str = "DEBUG_MODE_ON"
REQUEST_TIME_OUT: float = 10.0
