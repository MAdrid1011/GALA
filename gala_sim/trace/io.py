"""Chunk-oriented trace persistence; one file per columnar array."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gala_sim.clamp.events import dependency_dtype, event_dtype

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
    def read(
        self, root: Path, *, validate: bool = True, mmap_mode: str | None = None
    ) -> Trace:
        root = Path(root)
        chunk_manifest_path = root / "chunk_manifest.json"
        if not (root / "metadata.json").is_file() and chunk_manifest_path.is_file():
            return self._read_raw_manifest(root, chunk_manifest_path, validate, mmap_mode)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace schema")
        event_metadata = {
            "schema_version": metadata.get("event_schema_version"),
            **{key: value for key, value in metadata.items()
               if key not in {"schema_version", "event_schema_version"}},
        }
        trace = Trace(
            events=np.load(root / "events.npy", mmap_mode=mmap_mode, allow_pickle=False),
            dependencies=np.load(root / "dependencies.npy", mmap_mode=mmap_mode, allow_pickle=False),
            payload=np.load(root / "payload.npy", mmap_mode=mmap_mode, allow_pickle=False),
            metadata=event_metadata,
        )
        if validate:
            validate_trace(trace)
        return trace

    @staticmethod
    def _read_raw_manifest(
        root: Path, manifest_path: Path, validate: bool, mmap_mode: str | None,
    ) -> Trace:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "gala-trace-chunks-v1":
            raise ValueError("unsupported trace chunk schema")
        if manifest.get("storage_format") != "raw_columns":
            raise ValueError("chunk manifest is not a raw-column trace")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("schema_version") is None:
            raise ValueError("raw trace metadata is malformed")
        def raw_array(name: str, dtype: np.dtype, count: int) -> np.ndarray:
            if count == 0:
                return np.empty(0, dtype=dtype)
            if mmap_mode is None:
                return np.fromfile(root / name, dtype=dtype, count=count)
            return np.memmap(root / name, dtype=dtype, mode="r", shape=(count,))

        trace = Trace(
            events=raw_array(
                manifest["event_chunks"][0], event_dtype(), int(manifest["event_count"])
            ),
            dependencies=raw_array(
                manifest["dependency_chunks"][0], dependency_dtype(),
                int(manifest["dependency_count"]),
            ),
            payload=raw_array(
                manifest["payload_chunks"][0], np.dtype("<f4"),
                int(manifest["payload_count"]),
            ),
            metadata=metadata,
        )
        if validate:
            validate_trace(trace)
        return trace
