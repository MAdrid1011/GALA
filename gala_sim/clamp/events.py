"""Versioned event vocabulary for the functional and cycle paths."""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np


EVENT_SCHEMA_VERSION: Final[str] = "gala-clamp-events-v1"
NULL_ID: Final[int] = -1


class PrimitiveKind(IntEnum):
    RELATION_CANDIDATE = 1
    RELATION = 2
    QUERY_CLOSE = 3
    FORWARD = 4
    QUERY_REDUCTION = 5
    CONSUMER = 6
    ADJOINT = 7
    CACHE_REQUEST = 8
    CACHE_RETURN = 9
    GRADIENT_REDUCTION = 10
    UPDATE_COMMIT = 11
    SET_MODIFICATION = 12


class ResourceClass(IntEnum):
    RELATION = 1
    ISSUE = 2
    CACHE = 3
    COMPUTE = 4
    QUERY = 5
    UPDATE = 6
    SRAM = 7
    MEMORY = 8


def event_dtype() -> np.dtype:
    """Return the frozen column layout used by every trace writer."""

    return np.dtype(
        [
            ("event_id", "<u8"),
            ("iteration_id", "<u4"),
            ("primitive_kind", "<u2"),
            ("query_id", "<i8"),
            ("gaussian_id", "<i8"),
            ("state_version", "<u4"),
            ("relation_id", "<i8"),
            ("consumer_id", "<i8"),
            ("reduction_key", "<i8"),
            ("resource_class", "<u2"),
            ("dependency_begin", "<u8"),
            ("dependency_count", "<u4"),
            ("template_id", "<u2"),
            ("field_mask", "<u4"),
            ("address_token", "<u8"),
            ("data_bytes", "<u4"),
            ("payload_offset", "<u8"),
            ("payload_length", "<u4"),
            ("flags", "<u4"),
        ],
        align=False,
    )


def dependency_dtype() -> np.dtype:
    return np.dtype("<u8")


class TraceEvent:
    """Typed constructor for one event; storage remains a NumPy row."""

    __slots__ = ("values",)

    def __init__(self, **values: int) -> None:
        dtype_names = set(event_dtype().names or ())
        unknown = set(values).difference(dtype_names)
        if unknown:
            raise ValueError(f"unknown event fields: {sorted(unknown)}")
        self.values = values

    def as_tuple(self) -> tuple[int, ...]:
        dtype = event_dtype()
        defaults = {
            name: 0
            for name in dtype.names or ()
        }
        defaults.update(
            {
                "query_id": NULL_ID,
                "gaussian_id": NULL_ID,
                "relation_id": NULL_ID,
                "consumer_id": NULL_ID,
                "reduction_key": NULL_ID,
            }
        )
        defaults.update(self.values)
        return tuple(int(defaults[name]) for name in dtype.names or ())
