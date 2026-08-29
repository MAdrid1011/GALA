"""Structured trace storage and validation."""

from .model import TRACE_SCHEMA_VERSION, Trace
from .io import TraceReader, TraceWriter
from .validator import (
    TraceValidationConfig, TraceValidationError, TraceValidationReport, validate_trace,
)
from .sink import DeviceTraceSink, NumpyChunkSink
from .sample import (
    QUERY_PACKET_SAMPLE_SCHEMA_VERSION,
    QUERY_PACKET_SAMPLE_SCHEMA_VERSIONS,
    QueryPacketSampleConfig,
    QueryRange,
    TraceSampleConfig,
    dependency_closed_query_sample,
    real_query_packet_sample,
)
from .packetize import (
    PACKET_DERIVATION_SCHEMA_VERSION,
    QueryDomain,
    derive_quick_relation_packets,
    validate_packet_derivation,
)
from .captured_packets import (
    CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION,
    CapturedPacketSpec,
    captured_virtual_packet,
    complete_captured_packet_sample,
)
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
from .compare import VirtualRecordComparison, compare_virtual_packet_records
from .archive import (
    ARCHIVE_SCHEMA_VERSION, VirtualPacketArchiveReader, VirtualPacketArchiveWriter,
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
    "QueryPacketSampleConfig",
    "TraceSampleConfig",
    "dependency_closed_query_sample",
    "QUERY_PACKET_SAMPLE_SCHEMA_VERSION",
    "QUERY_PACKET_SAMPLE_SCHEMA_VERSIONS",
    "real_query_packet_sample",
    "PACKET_DERIVATION_SCHEMA_VERSION",
    "QueryDomain",
    "derive_quick_relation_packets",
    "validate_packet_derivation",
    "CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION",
    "CapturedPacketSpec",
    "captured_virtual_packet",
    "complete_captured_packet_sample",
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
    "VirtualRecordComparison",
    "compare_virtual_packet_records",
    "ARCHIVE_SCHEMA_VERSION", "VirtualPacketArchiveReader", "VirtualPacketArchiveWriter",
]
