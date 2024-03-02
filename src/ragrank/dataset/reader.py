"""Reader module for Ragrank"""

from __future__ import annotations

from typing import Dict, List

from ragrank.constants import DATA_FIELDS
from ragrank.dataset import Dataset
from ragrank.exceptions import ValidationError


def from_dict(data: Dict[str, List[str] | str]) -> Dataset:
    """create a dataset object from dictionary"""

    for data_field in DATA_FIELDS:
        if data_field not in data:
            raise ValidationError("%s not in the data" % data_field)
