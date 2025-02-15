"""
Numpy and scipy utilities.
"""

from typing import Any


def item(value: Any) -> Any:
    """
    Meant to return the single value from a numpy array if it's defined.
    """
    try:
        return value.item()
    except AttributeError:
        pass
    return value
