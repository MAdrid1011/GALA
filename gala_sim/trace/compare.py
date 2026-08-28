"""Bounded field-by-field comparisons for CUDA relation records."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .virtual import VirtualTracePacket


@dataclass(frozen=True)
class VirtualRecordComparison:
    """Result of comparing one virtual packet with legacy decoder columns."""

    candidate_count: int
    relation_count: int
    candidate_mismatches: int
    relation_mismatches: int
    first_mismatches: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            self.candidate_count >= 0
            and self.relation_count >= 0
            and self.candidate_mismatches == 0
            and self.relation_mismatches == 0
        )


def compare_virtual_packet_records(
    packet: VirtualTracePacket,
    candidate_records: np.ndarray,
    relation_records: np.ndarray,
    *,
    gaussian_ids: np.ndarray | None = None,
    candidate_index_base: int = 0,
    relation_batch_size: int = 1_000_000,
    max_mismatches: int = 16,
) -> VirtualRecordComparison:
    """Compare a packet with legacy ``[kind, index, value, key]`` records.

    The legacy CUDA writer emits candidates and relations in candidate-major
    order.  Relation rows are checked in bounded vectorized batches against
    the exact mask bits.  Strict candidate/local-query ordering, uniqueness,
    and the mask popcount prove that neither side has an extra or missing bit;
    no relation-sized expected array is allocated.
    ``gaussian_ids`` maps legacy point-list indexes to the stable IDs stored in
    the packet.  Passing it is required when the capture has undergone Gaussian
    densification or pruning.
    """

    if max_mismatches < 0:
        raise ValueError("max_mismatches must be non-negative")
    if candidate_index_base < 0:
        raise ValueError("candidate_index_base must be non-negative")
    if relation_batch_size <= 0:
        raise ValueError("relation_batch_size must be positive")
    candidates = np.asarray(candidate_records)
    relations = np.asarray(relation_records)
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError("candidate records must have shape [N, 4]")
    if relations.ndim != 2 or relations.shape[1] != 4:
        raise ValueError("relation records must have shape [N, 4]")
    if candidates.dtype.kind not in "iu" or relations.dtype.kind not in "iu":
        raise ValueError("legacy records must use integer columns")
    if gaussian_ids is None:
        stable_ids = None
    else:
        stable_ids = np.asarray(gaussian_ids, dtype=np.int64)
        if stable_ids.ndim != 1:
            raise ValueError("gaussian_ids must be one-dimensional")

    mismatch_messages: list[str] = []

    def mismatch(message: str) -> None:
        if len(mismatch_messages) < max_mismatches:
            mismatch_messages.append(message)

    candidate_mismatches = 0
    if candidates.shape[0] != packet.candidate_count:
        candidate_mismatches += abs(candidates.shape[0] - packet.candidate_count)
        mismatch(
            f"candidate count old={candidates.shape[0]} new={packet.candidate_count}"
        )
    compare_candidates = min(candidates.shape[0], packet.candidate_count)
    if compare_candidates:
        old_keys = np.ascontiguousarray(
            candidates[:compare_candidates, 3], dtype=np.int64
        ).view(np.uint64)
        expected_ids = np.asarray(packet.point_ids[:compare_candidates], dtype=np.int64)
        if stable_ids is not None:
            indexes = np.asarray(candidates[:compare_candidates, 2], dtype=np.int64)
            valid = (indexes >= 0) & (indexes < stable_ids.size)
            invalid = int((~valid).sum())
            candidate_mismatches += invalid
            if invalid:
                mismatch("candidate Gaussian index is outside gaussian_ids")
            mapped = np.full(compare_candidates, -1, dtype=np.int64)
            mapped[valid] = stable_ids[indexes[valid]]
            old_ids = mapped
        else:
            old_ids = np.asarray(candidates[:compare_candidates, 2], dtype=np.int64)
        checks = (
            np.asarray(candidates[:compare_candidates, 0]) != 0,
            np.asarray(candidates[:compare_candidates, 1])
            != candidate_index_base + np.arange(
                compare_candidates, dtype=candidates.dtype
            ),
            old_ids != expected_ids,
            old_keys != np.asarray(packet.point_keys[:compare_candidates], dtype=np.uint64),
        )
        for name, failed in zip(
            ("kind", "candidate_index", "gaussian_id", "point_key"), checks, strict=True
        ):
            count = int(np.asarray(failed, dtype=bool).sum())
            candidate_mismatches += count
            if count:
                mismatch(f"candidate {name} mismatches: {count}")

    relation_mismatches = 0
    relation_count = packet.logical_relation_count
    if relations.shape[0] != relation_count:
        relation_mismatches += abs(relations.shape[0] - relation_count)
        mismatch(f"relation count old={relations.shape[0]} new={relation_count}")
    old_cursor = 0
    candidate_batch = max(1, relation_batch_size // packet.local_query_count)
    for candidate_start in range(0, packet.candidate_count, candidate_batch):
        candidate_end = min(candidate_start + candidate_batch, packet.candidate_count)
        mask_bits = np.unpackbits(
            np.asarray(packet.masks[candidate_start:candidate_end], dtype=np.uint32)
            .view(np.uint8),
            bitorder="little",
        ).reshape(candidate_end - candidate_start, packet.local_query_count)
        expected_candidates, expected_local = np.nonzero(mask_bits)
        old_end = old_cursor + expected_candidates.size
        rows = relations[old_cursor:old_end]
        if rows.shape[0] != expected_candidates.size:
            mismatch(
                f"relation count for candidates [{candidate_start}, {candidate_end}) "
                f"old={rows.shape[0]} new={expected_candidates.size}"
            )
            relation_mismatches += abs(rows.shape[0] - expected_candidates.size)
            # Continue from the expected stream position when the old input is
            # truncated; remaining candidates cannot be compared safely.
            if rows.shape[0] < expected_candidates.size:
                break
        if rows.size:
            expected_candidate_values = expected_candidates + candidate_start + candidate_index_base
            expected_local_values = expected_local.astype(np.int64, copy=False)
            old_values = np.asarray(rows, dtype=np.int64)
            checks = (
                ("kind", old_values[:, 0] != 1),
                ("candidate_index", old_values[:, 1] != expected_candidate_values),
                ("local_query", old_values[:, 2] != expected_local_values),
                ("reserved_key", old_values[:, 3] != 0),
            )
            for name, failed in checks:
                count = int(failed.sum())
                relation_mismatches += count
                if count:
                    mismatch(
                        f"relation {name} mismatches: {count} "
                        f"in [{old_cursor}, {old_end})"
                    )
        old_cursor = old_end
    if old_cursor != relations.shape[0]:
        extra = relations.shape[0] - old_cursor
        relation_mismatches += max(extra, 0)
        if extra > 0:
            mismatch(f"relation records remain after mask stream: {extra}")

    return VirtualRecordComparison(
        candidate_count=int(candidates.shape[0]),
        relation_count=int(relations.shape[0]),
        candidate_mismatches=candidate_mismatches,
        relation_mismatches=relation_mismatches,
        first_mismatches=tuple(mismatch_messages),
    )
