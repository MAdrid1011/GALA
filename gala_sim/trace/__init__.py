"""Structured trace storage and validation."""

from .model import TRACE_SCHEMA_VERSION, Trace
from .io import TraceReader, TraceWriter
from .validator import (
    TraceValidationConfig, TraceValidationError, TraceValidationReport, validate_trace,
)
from .sink import DeviceTraceSink, NumpyChunkSink
from .sample import QueryRange, TraceSampleConfig, dependency_closed_query_sample
from .virtual import (
    MASK_WORD_BITS,
    VirtualTracePacket,
    VirtualTraceProgress,
    VirtualTraceRun,
    VirtualTraceStream,
    VirtualEventPacket,
    VirtualRelationEventExpander,
    VirtualEventStreamValidator,
    VirtualQueryEventExpander,
    VirtualLifecycleKind,
    VirtualLifecycleRecord,
    VirtualIterationLedger,
    VirtualTraceLifecycleValidator,
    TRANSACTION_COLLECTION,
    TRANSACTION_OPTIMIZER,
)

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
    "MASK_WORD_BITS",
    "VirtualTracePacket",
    "VirtualTraceProgress",
    "VirtualTraceRun",
    "VirtualTraceStream",
    "VirtualEventPacket",
    "VirtualRelationEventExpander",
    "VirtualEventStreamValidator",
    "VirtualQueryEventExpander",
    "VirtualLifecycleKind",
    "VirtualLifecycleRecord",
    "VirtualIterationLedger",
    "VirtualTraceLifecycleValidator",
    "TRANSACTION_COLLECTION",
    "TRANSACTION_OPTIMIZER",
]
