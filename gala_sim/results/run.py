"""Write the machine-readable files required for one cycle run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gala_sim.timing import CycleResult

from .manifest import write_json


ALLOWED_STATUS = {
    "planned", "running", "passed", "failed_quality", "failed_trace", "failed_cycle",
    "failed_preflight", "unavailable_source", "unavailable_data", "license_blocked",
}


class RunOutputWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_cycles(self, result: CycleResult) -> None:
        write_json({
            "total_cycles": result.total_cycles,
            "policy": result.policy,
            "oracle_status": result.oracle_status,
            "module_counters": result.module_counters,
            "completion_cycles": {str(key): value for key, value in result.completion_cycles.items()},
        }, self.root / "cycles.json")
        write_json({"event_counts": result.event_counts}, self.root / "events.json")
        self._write_stalls(result)

    def write_quality(self, quality: dict[str, Any]) -> None:
        write_json(quality, self.root / "quality.json")

    def write_gpu_reference(self, reference: dict[str, Any]) -> None:
        write_json(reference, self.root / "gpu_reference.json")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(manifest, self.root / "manifest.json")

    def write_status(self, status: str, *, reason: str | None = None,
                     checks: dict[str, Any] | None = None) -> None:
        if status not in ALLOWED_STATUS:
            raise ValueError(f"unsupported run status: {status}")
        value: dict[str, Any] = {"status": status, "reason": reason, "checks": checks or {}}
        write_json(value, self.root / "status.json")

    def _write_stalls(self, result: CycleResult) -> None:
        rows = [
            {"cycle": item.cycle, "module": item.module, "reason": item.reason,
             "event_ids": list(item.event_ids), "count": item.count}
            for item in result.stalls
        ]
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("pyarrow is required to write stalls.parquet") from error
        table = pa.Table.from_pylist(rows, schema=pa.schema([
            ("cycle", pa.int64()), ("module", pa.string()), ("reason", pa.string()),
            ("event_ids", pa.list_(pa.int64())), ("count", pa.int64()),
        ]))
        pq.write_table(table, self.root / "stalls.parquet")
