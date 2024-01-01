# with python >=3.11 use:
# from enum import StrEnum
from enum import Enum


class Model_Type(str, Enum):
    """
    An enum that lists all available model types.
    """

    CUBIC = "cubic"
    ISOTHERMAL_EULER = "isothermal_euler"
    EULER = "euler"
