"""Persistent, compact archives for exact virtual packet captures.

The archive stores source packets and lifecycle records only.  Expanded CLAMP
event/dependency columns are intentionally never written here; a reader can
recreate fresh :class:`VirtualTracePacket` objects for each policy replay.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Iterator
import zipfile
import zlib

import numpy as np

from .virtual import (
    VirtualLifecycleKind,
    VirtualLifecycleRecord,
    VirtualTraceLifecycleValidator,
    VirtualTracePacket,
)


ARCHIVE_SCHEMA_VERSION = "gala-virtual-packet-archive-v1"


class VirtualPacketArchiveWriter:
    """Write an ordered virtual packet stream in byte-bounded compressed chunks."""

    def __init__(
        self, root: Path, *, max_chunk_bytes: int, max_inflight_chunks: int = 1,
    ) -> None:
        if max_chunk_bytes <= 0 or max_inflight_chunks <= 0:
            raise ValueError("archive chunk capacities must be positive")
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("virtual packet archive root must be empty")
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.max_inflight_chunks = int(max_inflight_chunks)
        self._chunk_executor = ThreadPoolExecutor(
            max_workers=self.max_inflight_chunks,
            thread_name_prefix="gala-archive",
        )
        self._chunk_futures: deque[Future[None]] = deque()
        self._chunk_packets: list[dict[str, Any]] = []
        self._chunk_ids: list[np.ndarray] = []
        self._chunk_keys: list[np.ndarray] = []
        self._chunk_masks: list[np.ndarray] = []
        self._chunk_bytes = 0
        self._chunk_candidate_count = 0
        self._chunk_mask_count = 0
        self._peak_chunk_bytes = 0
        self._chunk_index = 0
        self._initial_gaussian_count: int | None = None
        self._last_packet_iteration: int | None = None
        self._packet_count = 0
        self._query_count = 0
        self._candidate_count = 0
        self._relation_count = 0
        self._physical_stream_bytes = 0
        self._closed_iterations: list[int] = []
        self._finished = False
        self.root.mkdir(parents=True, exist_ok=True)
        self._stream_file = (self.root / "stream.jsonl").open("x", encoding="utf-8")

    def initialize_gaussians(self, count: int) -> None:
        if count < 0:
            raise ValueError("initial Gaussian count must be non-negative")
        if self._initial_gaussian_count is not None and self._initial_gaussian_count != count:
            raise ValueError("archive Gaussian count changed during initialization")
        self._initial_gaussian_count = int(count)

    def append_packet(self, packet: VirtualTracePacket) -> None:
        self._ensure_open()
        if self._initial_gaussian_count is None:
            raise RuntimeError("archive Gaussian count is not initialized")
        if (
            self._last_packet_iteration is not None
            and packet.iteration_id < self._last_packet_iteration
        ):
            raise ValueError("archive packet iteration order is not monotonic")
        self._last_packet_iteration = int(packet.iteration_id)
        ids = np.asarray(packet.point_ids, dtype=np.int64)
        keys = np.asarray(packet.point_keys, dtype=np.uint64)
        masks = np.asarray(packet.masks, dtype=np.dtype("<u4"))
        packet_bytes = int(ids.nbytes + keys.nbytes + masks.nbytes)
        if self._chunk_packets and self._chunk_bytes + packet_bytes > self.max_chunk_bytes:
            self._flush_chunk()
        packet_meta = {
            "iteration_id": int(packet.iteration_id),
            "template_id": int(packet.template_id),
            "query_base": int(packet.query_base),
            "query_shape": [int(value) for value in packet.query_shape],
            "state_version": int(packet.state_version),
            "field_mask": int(packet.field_mask),
            "loss_flags": int(packet.loss_flags),
            "ssim_radius": int(packet.ssim_radius),
            "backward_confirmed": bool(packet.backward_confirmed),
            "candidate_offset": self._chunk_candidate_count,
            "mask_offset": self._chunk_mask_count,
            "candidate_count": int(ids.size),
            "mask_words": int(masks.shape[1]),
        }
        self._chunk_packets.append(packet_meta)
        self._chunk_ids.append(ids.copy())
        self._chunk_keys.append(keys.copy())
        self._chunk_masks.append(masks.reshape(-1).copy())
        self._chunk_bytes += packet_bytes
        self._chunk_candidate_count += int(ids.size)
        self._chunk_mask_count += int(masks.size)
        self._peak_chunk_bytes = max(self._peak_chunk_bytes, self._chunk_bytes)
        self._append_stream({"type": "packet", "chunk": self._chunk_index,
                             "index": len(self._chunk_packets) - 1})
        self._packet_count += 1
        self._query_count += packet.query_count
        self._candidate_count += packet.candidate_count
        self._relation_count += packet.logical_relation_count
        self._physical_stream_bytes += packet.physical_bytes
        if self._chunk_bytes >= self.max_chunk_bytes:
            self._flush_chunk()

    def append_lifecycle(self, record: VirtualLifecycleRecord) -> None:
        self._ensure_open()
        self._append_stream({"type": "lifecycle", "record": _lifecycle_to_dict(record)})

    def close_iteration(self, iteration_id: int) -> None:
        self._ensure_open()
        if self._closed_iterations and iteration_id <= self._closed_iterations[-1]:
            raise ValueError("archive iteration close order is not strictly increasing")
        self._closed_iterations.append(int(iteration_id))
        self._append_stream({"type": "close_iteration", "iteration_id": int(iteration_id)})

    def finish(
        self, *, metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        if self._initial_gaussian_count is None:
            raise RuntimeError("archive Gaussian count is not initialized")
        observed_complete_30k = self._closed_iterations == list(range(1, 30_001))
        self._flush_chunk()
        self._stream_file.flush()
        self._stream_file.close()
        self._finish_chunks()
        result: dict[str, Any] = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "status": "passed",
            "formal_performance_eligible": False,
            "complete_30k": observed_complete_30k,
            "validation_passed": False,
            "initial_gaussian_count": self._initial_gaussian_count,
            "packet_count": self._packet_count,
            "query_count": self._query_count,
            "candidate_count": self._candidate_count,
            "relation_count": self._relation_count,
            "physical_stream_bytes": self._physical_stream_bytes,
            "iteration_count": len(self._closed_iterations),
            "chunk_count": self._chunk_index,
            "max_chunk_bytes": self.max_chunk_bytes,
            "max_inflight_chunks": self.max_inflight_chunks,
            "peak_uncompressed_chunk_bytes": self._peak_chunk_bytes,
            "chunk_storage": "npz_deflate",
            "chunk_compression_level": zlib.Z_BEST_SPEED,
            "stream_path": "stream.jsonl",
            "chunks": [f"chunks/chunk-{index:06d}.npz" for index in range(self._chunk_index)],
            "metadata": dict(metadata or {}),
        }
        (self.root / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        self._finished = True
        return result

    def _append_stream(self, item: dict[str, Any]) -> None:
        self._stream_file.write(
            json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n"
        )

    def _flush_chunk(self) -> None:
        if not self._chunk_packets:
            return
        chunk_dir = self.root / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ids = np.concatenate(self._chunk_ids) if self._chunk_ids else np.empty(0, dtype=np.int64)
        keys = np.concatenate(self._chunk_keys) if self._chunk_keys else np.empty(0, dtype=np.uint64)
        masks = np.concatenate(self._chunk_masks) if self._chunk_masks else np.empty(0, dtype=np.dtype("<u4"))
        path = chunk_dir / f"chunk-{self._chunk_index:06d}.npz"
        packets = np.asarray(json.dumps(self._chunk_packets), dtype=np.str_)
        while len(self._chunk_futures) >= self.max_inflight_chunks:
            self._chunk_futures.popleft().result()
        self._chunk_futures.append(self._chunk_executor.submit(
            _write_archive_chunk, path, ids, keys, masks, packets,
        ))
        self._chunk_packets.clear()
        self._chunk_ids.clear()
        self._chunk_keys.clear()
        self._chunk_masks.clear()
        self._chunk_bytes = 0
        self._chunk_candidate_count = 0
        self._chunk_mask_count = 0
        self._chunk_index += 1

    def _finish_chunks(self) -> None:
        try:
            while self._chunk_futures:
                self._chunk_futures.popleft().result()
        finally:
            self._chunk_executor.shutdown(wait=True, cancel_futures=False)

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("virtual packet archive is already finalized")


class VirtualPacketArchiveReader:
    """Read an archive and feed a newly supplied consumer in source order."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("virtual packet archive manifest is missing")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
            raise ValueError("unsupported virtual packet archive schema")
        self._chunk_index: int | None = None
        self._chunk: dict[str, Any] | None = None

    @property
    def initial_gaussian_count(self) -> int:
        return int(self.manifest["initial_gaussian_count"])

    @property
    def formal_performance_eligible(self) -> bool:
        return bool(self.manifest.get("formal_performance_eligible", False))

    def records(self) -> Iterator[tuple[str, Any]]:
        stream_path = self.root / str(self.manifest.get("stream_path", "stream.jsonl"))
        with stream_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                kind = item.get("type")
                if kind == "packet":
                    yield "packet", self._packet(item)
                elif kind == "lifecycle":
                    yield "lifecycle", _lifecycle_from_dict(item["record"])
                elif kind == "close_iteration":
                    yield "close_iteration", int(item["iteration_id"])
                else:
                    raise ValueError("virtual packet archive stream record is malformed")

    def replay(self, consumer: Any, *, finish: bool = False) -> None:
        """Feed one independent consumer; callers may replay this reader again."""
        if hasattr(consumer, "initialize_gaussians"):
            consumer.initialize_gaussians(self.initial_gaussian_count)
        for kind, value in self.records():
            if kind == "packet":
                _call(consumer, "accept_query_packet", value)
            elif kind == "lifecycle":
                _call(consumer, "accept_lifecycle", value)
            else:
                _call(consumer, "close_iteration", value)
        if finish and hasattr(consumer, "finish"):
            consumer.finish()

    def replay_session(self, engine: Any, **kwargs: Any) -> Any:
        """Create and finish a fresh online session for one policy.

        Semantic totals are derived per iteration by the same bounded adapter
        used during live capture.  Future-visible Oracle policies intentionally
        remain unsupported; they require a complete expanded trace.
        """
        from gala_sim.timing.engine import BufferedVirtualCycleConsumer

        session = engine.online_session(
            initial_gaussian_count=self.initial_gaussian_count,
            total_iterations=int(self.manifest["iteration_count"]),
            **kwargs,
        )
        consumer = BufferedVirtualCycleConsumer(session)
        self.replay(consumer, finish=True)
        return consumer.result

    def validate(self, *, promote: bool = False) -> dict[str, Any]:
        """Re-read and validate every compact packet and lifecycle record.

        Formal eligibility is derived exclusively from the observed archive
        contents. ``promote`` records a successful validation in the archive
        manifest, but cannot make an incomplete iteration stream eligible.
        """
        validator = VirtualTraceLifecycleValidator(self.initial_gaussian_count)
        packet_count = 0
        query_count = 0
        candidate_count = 0
        relation_count = 0
        physical_stream_bytes = 0
        closed_iterations: list[int] = []
        for kind, value in self.records():
            if kind == "packet":
                validator.accept_packet(value)
                packet_count += 1
                query_count += value.query_count
                candidate_count += value.candidate_count
                relation_count += value.logical_relation_count
                physical_stream_bytes += value.physical_bytes
            elif kind == "lifecycle":
                validator.accept_lifecycle(value)
            else:
                validator.close_iteration(value)
                closed_iterations.append(value)
        ledgers = validator.finalize()
        observed = {
            "packet_count": packet_count,
            "query_count": query_count,
            "candidate_count": candidate_count,
            "relation_count": relation_count,
            "physical_stream_bytes": physical_stream_bytes,
            "iteration_count": len(closed_iterations),
        }
        for name, value in observed.items():
            recorded = self.manifest.get(name)
            if recorded is not None and int(recorded) != value:
                raise ValueError(f"virtual packet archive {name} does not match its stream")
        complete_30k = closed_iterations == list(range(1, 30_001))
        if bool(self.manifest.get("complete_30k", False)) != complete_30k:
            raise ValueError("virtual packet archive complete_30k marker is inconsistent")
        audit = self.manifest.get("metadata", {}).get("capture_audit", {})
        audit_contract = {
            "captured_query_kernel_calls": packet_count,
            "captured_backward_calls": packet_count,
            "captured_logical_queries": query_count,
            "captured_consumers": query_count,
            "cuda_relation_candidates": candidate_count,
            "cuda_valid_relations": relation_count,
            "captured_backward_relations": relation_count,
        }
        audit_binding_passed = (
            isinstance(audit, dict)
            and all(
                name in audit and int(audit[name]) == expected
                for name, expected in audit_contract.items()
            )
            and sum(
                int(audit.get(name, 0))
                for name in (
                    "captured_raster_kernel_calls",
                    "captured_voxel_kernel_calls",
                )
            ) == packet_count
        )
        if complete_30k and not audit_binding_passed:
            raise ValueError(
                "formal virtual packet archive is not bound to complete capture audit counts"
            )
        formal_eligible = complete_30k and audit_binding_passed
        report: dict[str, Any] = {
            "schema_version": "gala-virtual-packet-archive-validation-v1",
            "status": "passed",
            "validation_scope": "complete_compact_packet_and_lifecycle_stream",
            "validation_passed": True,
            "complete_30k": complete_30k,
            "formal_performance_eligible": formal_eligible,
            "capture_audit_binding_passed": audit_binding_passed,
            **observed,
            "state_version_end": validator.state_version,
            "active_gaussian_count_end": len(validator.active_gaussians or ()),
            "first_iteration": closed_iterations[0] if closed_iterations else None,
            "last_iteration": closed_iterations[-1] if closed_iterations else None,
            "ledger_count": len(ledgers),
        }
        if promote:
            self.manifest.update({
                "validation_passed": True,
                "formal_performance_eligible": formal_eligible,
                "validation_report_path": "validation.json",
            })
            _write_json_atomic(self.root / "validation.json", report)
            _write_json_atomic(self.root / "manifest.json", self.manifest)
        return report

    def _packet(self, item: dict[str, Any]) -> VirtualTracePacket:
        chunk_index = int(item["chunk"])
        index = int(item["index"])
        chunk = self._load_chunk(chunk_index)
        meta = chunk["packets"][index]
        candidate_offset = int(meta["candidate_offset"])
        candidate_count = int(meta["candidate_count"])
        mask_offset = int(meta["mask_offset"])
        mask_words = int(meta["mask_words"])
        ids = chunk["point_ids"][candidate_offset:candidate_offset + candidate_count].copy()
        keys = chunk["point_keys"][candidate_offset:candidate_offset + candidate_count].copy()
        mask_end = mask_offset + candidate_count * mask_words
        masks = chunk["masks"][mask_offset:mask_end].reshape(candidate_count, mask_words).copy()
        return VirtualTracePacket(
            iteration_id=int(meta["iteration_id"]), template_id=int(meta["template_id"]),
            query_base=int(meta["query_base"]), query_shape=tuple(meta["query_shape"]),
            point_ids=ids, point_keys=keys, masks=masks,
            state_version=int(meta["state_version"]), field_mask=int(meta["field_mask"]),
            loss_flags=int(meta["loss_flags"]), ssim_radius=int(meta["ssim_radius"]),
            backward_confirmed=bool(meta["backward_confirmed"]),
        )

    def _load_chunk(self, index: int) -> dict[str, Any]:
        if self._chunk_index != index:
            path = self.root / "chunks" / f"chunk-{index:06d}.npz"
            with np.load(path, allow_pickle=False) as data:
                packet_json = str(data["packets"].item())
                self._chunk = {
                    "point_ids": np.asarray(data["point_ids"], dtype=np.int64),
                    "point_keys": np.asarray(data["point_keys"], dtype=np.uint64),
                    "masks": np.asarray(data["masks"], dtype=np.dtype("<u4")),
                    "packets": json.loads(packet_json),
                }
                self._chunk_index = index
        if self._chunk is None:
            raise RuntimeError("virtual packet archive chunk failed to load")
        return self._chunk


def _call(consumer: Any, method: str, value: Any) -> None:
    callback = getattr(consumer, method, None)
    if callback is None:
        raise TypeError(f"archive consumer lacks {method}()")
    callback(value)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_archive_chunk(
    path: Path,
    point_ids: np.ndarray,
    point_keys: np.ndarray,
    masks: np.ndarray,
    packets: np.ndarray,
) -> None:
    """Write one standard NPZ chunk without blocking the capture producer."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    arrays = {
        "point_ids": point_ids,
        "point_keys": point_keys,
        "masks": masks,
        "packets": packets,
    }
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=zlib.Z_BEST_SPEED,
            allowZip64=True,
        ) as archive:
            for name, value in arrays.items():
                with archive.open(name + ".npy", mode="w", force_zip64=True) as stream:
                    np.lib.format.write_array(
                        stream, np.asanyarray(value), allow_pickle=False,
                    )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _lifecycle_to_dict(record: VirtualLifecycleRecord) -> dict[str, Any]:
    return {
        "iteration_id": int(record.iteration_id),
        "kind": int(record.kind),
        "state_version": int(record.state_version),
        "field_mask": int(record.field_mask),
        "gaussian_id": int(record.gaussian_id),
        "parent_id": int(record.parent_id),
        "child_ids": list(record.child_ids),
        "transaction_kind": int(record.transaction_kind),
        "all_active": bool(record.all_active),
        "dependency_ids": list(record.dependency_ids),
        "active_ids": list(record.active_ids),
    }


def _lifecycle_from_dict(value: dict[str, Any]) -> VirtualLifecycleRecord:
    return VirtualLifecycleRecord(
        iteration_id=int(value["iteration_id"]), kind=VirtualLifecycleKind(int(value["kind"])),
        state_version=int(value["state_version"]), field_mask=int(value.get("field_mask", 0)),
        gaussian_id=int(value.get("gaussian_id", -1)), parent_id=int(value.get("parent_id", -1)),
        child_ids=tuple(int(item) for item in value.get("child_ids", ())),
        transaction_kind=int(value.get("transaction_kind", 0)),
        all_active=bool(value.get("all_active", False)),
        dependency_ids=tuple(int(item) for item in value.get("dependency_ids", ())),
        active_ids=tuple(int(item) for item in value.get("active_ids", ())),
    )
