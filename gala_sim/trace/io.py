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
        _write_array_if_needed(trace.events, root / "events.npy")
        _write_array_if_needed(trace.dependencies, root / "dependencies.npy")
        _write_array_if_needed(trace.payload, root / "payload.npy")
        metadata = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event_schema_version": trace.metadata["schema_version"],
            **{key: value for key, value in trace.metadata.items() if key != "schema_version"},
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


def _write_array_if_needed(array: np.ndarray, path: Path) -> None:
    """Avoid truncating a final mmap that already lives at the target path."""
    filename = getattr(array, "filename", None)
    if filename is not None:
        try:
            if Path(filename).resolve() == path.resolve():
                return
        except OSError:
            pass
    np.save(path, array, allow_pickle=False)


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
