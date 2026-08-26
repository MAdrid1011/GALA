"""Structured trace storage and validation."""

from .model import TRACE_SCHEMA_VERSION, Trace
from .io import TraceReader, TraceWriter
from .validator import (
    TraceValidationConfig, TraceValidationError, TraceValidationReport, validate_trace,
)
from .sink import DeviceTraceSink, NumpyChunkSink
from .sample import QueryRange, TraceSampleConfig, dependency_closed_query_sample

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "Trace",
    "TraceReader",
    "TraceWriter",
    "TraceValidationError",
    "TraceValidationConfig",
    "TraceValidationReport",
    "validate_trace",
    "DeviceTraceSink",
    "NumpyChunkSink",
    "QueryRange",
    "TraceSampleConfig",
    "dependency_closed_query_sample",
]
