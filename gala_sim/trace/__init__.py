"""Structured trace storage and validation."""

from .model import TRACE_SCHEMA_VERSION, Trace
from .io import TraceReader, TraceWriter
from .validator import TraceValidationError, TraceValidationReport, validate_trace
from .sink import DeviceTraceSink, NumpyChunkSink

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "Trace",
    "TraceReader",
    "TraceWriter",
    "TraceValidationError",
    "TraceValidationReport",
    "validate_trace",
    "DeviceTraceSink",
    "NumpyChunkSink",
]
