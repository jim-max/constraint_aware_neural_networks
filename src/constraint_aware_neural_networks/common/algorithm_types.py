# with python >=3.11 use:
# from enum import StrEnum
from enum import Enum


class Algorithm_Type(str, Enum):
    """
    An enum that lists all available training algorithms.
    """

    STANDARD = "standard"
    CONSTRAINT_ADAPTED_LOSS = "constraint_adapted_loss"
    CONSTRAINT_RESOLVING = "constraint_resolving"
