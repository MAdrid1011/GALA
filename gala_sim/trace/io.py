"""Chunk-oriented trace persistence; one file per columnar array."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import TRACE_SCHEMA_VERSION, Trace
from .validator import validate_trace


class TraceWriter:
    """Write a trace directory without per-event JSON or object arrays."""

    def write(self, trace: Trace, root: Path, *, validate: bool = True) -> None:
        if validate:
            validate_trace(trace)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "events.npy", trace.events, allow_pickle=False)
        np.save(root / "dependencies.npy", trace.dependencies, allow_pickle=False)
        np.save(root / "payload.npy", trace.payload, allow_pickle=False)
        metadata = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event_schema_version": trace.metadata["schema_version"],
            **{key: value for key, value in trace.metadata.items() if key != "schema_version"},
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


class TraceReader:
    def read(self, root: Path, *, validate: bool = True) -> Trace:
        root = Path(root)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace schema")
        event_metadata = {
            "schema_version": metadata.get("event_schema_version"),
            **{key: value for key, value in metadata.items()
               if key not in {"schema_version", "event_schema_version"}},
        }
        trace = Trace(
            events=np.load(root / "events.npy", allow_pickle=False),
            dependencies=np.load(root / "dependencies.npy", allow_pickle=False),
            payload=np.load(root / "payload.npy", allow_pickle=False),
            metadata=event_metadata,
        )
        if validate:
            validate_trace(trace)
        return trace
