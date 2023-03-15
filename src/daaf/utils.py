"""
Utilities, helpers.
"""
import dataclasses
import math
import uuid
from typing import Any, Mapping, Sequence


def create_task_id(timestamp: int):
    """
    Creates a task id using a given timestamp (epoch)
    and a partial uuid.
    """
    _uuid = next(iter(str(uuid.uuid4()).split("-")))
    return f"{timestamp}_{_uuid}"


def split(items: Sequence[Any], num_partitions: int) -> Sequence[Sequence[Any]]:
    """
    Attempts to split a list of items into sublists of equal size.

    If the numbers of items is not divisible by the number of
    partition sizes, the first partitions will have more items.
    If the number of partitions is higher than the number of items,
    only non-empty partitions are returned.
    """
    partition_size = math.ceil(len(items) / num_partitions)
    splits = []
    for idx in range(0, num_partitions - 1):
        splits.append(items[idx * partition_size : (idx + 1) * partition_size])
    splits.append(items[(num_partitions - 1) * partition_size :])
    return [partition for partition in splits if partition]


def dataclass_from_dict(clazz: Any, data: Mapping[str, Any]):  # type: ignore [arg-type]
    """
    Creates an instance of a dataclass from a dictionary.
    """
    fields = list(dataclasses.fields(clazz))
    return clazz(**{field.name: data[field.name] for field in fields})
