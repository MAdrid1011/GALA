"""Write the machine-readable files required for one cycle run."""

from __future__ import annotations

from dataclasses import asdict
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
            "oracle_portfolio": (
                asdict(result.oracle_portfolio)
                if result.oracle_portfolio is not None else None
            ),
            "module_counters": result.module_counters,
            "completion_cycles": {str(key): value for key, value in result.completion_cycles.items()},
            "compute_telemetry": (
                {
                    "cluster_count": result.compute_telemetry.cluster_count,
                    "occupancy_metric": result.compute_telemetry.occupancy_metric,
                    "event_timing_records": len(result.compute_telemetry.event_timings),
                    "cluster_occupancy_runs": len(
                        result.compute_telemetry.cluster_occupancy_runs
                    ),
                }
                if result.compute_telemetry is not None else None
            ),
            "oracle_member_artifacts": (
                {
                    name: f"oracle_members/{name}"
                    for name in result.oracle_member_results
                }
                if result.oracle_member_results else None
            ),
        }, self.root / "cycles.json")
        write_json({"event_counts": result.event_counts}, self.root / "events.json")
        self._write_stalls(result)
        self._write_memory_requests(result)
        self._write_compute_telemetry(result)
        if result.oracle_member_results:
            for name, member_result in result.oracle_member_results.items():
                RunOutputWriter(self.root / "oracle_members" / name).write_cycles(
                    member_result
                )

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
             "event_ids": list(item.event_ids), "count": item.count,
             "resource": item.resource, "pod": item.pod, "cluster": item.cluster,
             "resource_cycle": item.resource_cycle,
             "resource_in_use": item.resource_in_use,
             "resource_demand": item.resource_demand,
             "resource_capacity": item.resource_capacity}
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
            ("resource", pa.string()), ("pod", pa.int32()),
            ("cluster", pa.int32()), ("resource_cycle", pa.int64()),
            ("resource_in_use", pa.int32()), ("resource_demand", pa.int32()),
            ("resource_capacity", pa.int32()),
        ]))
        pq.write_table(table, self.root / "stalls.parquet")

    def _write_memory_requests(self, result: CycleResult) -> None:
        rows = [
            {
                "request_id": item.request_id,
                "address": item.address,
                "size_bytes": item.size_bytes,
                "is_write": item.is_write,
                "arrival_cycle": item.arrival_cycle,
                "completion_cycle": item.completion_cycle,
            }
            for item in result.memory_requests
        ]
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("pyarrow is required to write memory_requests.parquet") from error
        table = pa.Table.from_pylist(rows, schema=pa.schema([
            ("request_id", pa.int64()),
            ("address", pa.uint64()),
            ("size_bytes", pa.int64()),
            ("is_write", pa.bool_()),
            ("arrival_cycle", pa.int64()),
            ("completion_cycle", pa.int64()),
        ]))
        pq.write_table(table, self.root / "memory_requests.parquet")

    def _write_compute_telemetry(self, result: CycleResult) -> None:
        telemetry = result.compute_telemetry
        if telemetry is None:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "pyarrow is required to write ComputePod telemetry"
            ) from error
        timing_rows = [asdict(item) for item in telemetry.event_timings]
        timing_table = pa.Table.from_pylist(timing_rows, schema=pa.schema([
            ("event_id", pa.int64()),
            ("primitive_kind", pa.string()),
            ("pod", pa.int32()),
            ("cluster", pa.int32()),
            ("dependency_ready_cycle", pa.int64()),
            ("fusion_issue_cycle", pa.int64()),
            ("query_issue_cycle", pa.int64()),
            ("compute_issue_cycle", pa.int64()),
            ("finish_cycle", pa.int64()),
        ]))
        pq.write_table(timing_table, self.root / "compute_event_timing.parquet")
        occupancy_rows = [asdict(item) for item in telemetry.cluster_occupancy_runs]
        occupancy_table = pa.Table.from_pylist(occupancy_rows, schema=pa.schema([
            ("start_cycle", pa.int64()),
            ("end_cycle", pa.int64()),
            ("active_microcontexts", pa.list_(pa.int32())),
        ]))
        pq.write_table(
            occupancy_table, self.root / "compute_cluster_occupancy.parquet"
        )
        write_json({
            "schema_version": "gala-compute-telemetry-v1",
            "cluster_count": telemetry.cluster_count,
            "occupancy_metric": telemetry.occupancy_metric,
            "occupancy_encoding": "lossless_half_open_run_length",
            "event_timing_path": "compute_event_timing.parquet",
            "cluster_occupancy_path": "compute_cluster_occupancy.parquet",
            "event_timing_records": len(telemetry.event_timings),
            "cluster_occupancy_runs": len(telemetry.cluster_occupancy_runs),
        }, self.root / "compute_telemetry.json")
