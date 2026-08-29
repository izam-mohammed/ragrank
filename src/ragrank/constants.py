"""Field name constants for ragrank.

These are derived from `DataNode` rather than written out by hand, so
they cannot drift out of step with the model the way a duplicated list
does.
"""

from ragrank.dataset.base import DataNode

QUESTION_FIELD: str = "question"
CONTEXT_FIELD: str = "context"
RESPONSE_FIELD: str = "response"
REFERENCE_FIELD: str = "reference"
RETRIEVED_IDS_FIELD: str = "retrieved_ids"
REFERENCE_IDS_FIELD: str = "reference_ids"

#: Environment variable that turns on verbose evaluation logging.
DEBUG_MODE: str = "RAGRANK_DEBUG"

#: Fields every data point must have.
REQUIRED_FIELDS: list[str] = [
    QUESTION_FIELD,
    CONTEXT_FIELD,
    RESPONSE_FIELD,
]

#: Fields a data point may have, for metrics that need ground truth.
OPTIONAL_FIELDS: list[str] = [
    REFERENCE_FIELD,
    RETRIEVED_IDS_FIELD,
    REFERENCE_IDS_FIELD,
]

#: Every field a data point can carry.
DATA_FIELDS: list[str] = list(DataNode.model_fields)
