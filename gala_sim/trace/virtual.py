"""Bounded, lossless virtual trace packets for dense CUDA workloads.

The official rasterizer exposes a point list, a sorted point key and a
per-candidate validity mask.  Expanding every set bit into a structured event
array is useful for a small audit, but it is not a viable representation for a
30,000-iteration run.  This module keeps that exact information in a bounded
packet and exposes deterministic lazy relation iterators for consumers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
import queue
import threading
import time
from typing import Any

import numpy as np

from gala_sim.clamp.events import (
    PrimitiveKind,
    ResourceClass,
    TraceEvent,
    dependency_dtype,
    event_dtype,
)


MASK_WORD_BITS = 32
"""Number of query bits represented by one mask word."""

RASTER_TEMPLATE_ID = 1
VOXEL_TEMPLATE_ID = 2
RASTER_BLOCK = (16, 16)
VOXEL_BLOCK = (8, 8, 8)
LOSS_SSIM = 1 << 1
LOSS_TV = 1 << 2
TRANSACTION_COLLECTION = 1
TRANSACTION_OPTIMIZER = 2


class VirtualLifecycleKind(IntEnum):
    """Compact lifecycle records that are not represented by query masks."""

    UPDATE_BEGIN = 1
    UPDATE_COMMIT = 2
    UPDATE_END = 3
    PRUNE = 4
    CLONE = 5
    SPLIT = 6


@dataclass(frozen=True)
class VirtualLifecycleRecord:
    """One exact optimizer or Gaussian-set transition."""

    iteration_id: int
    kind: VirtualLifecycleKind
    state_version: int
    field_mask: int = 0
    gaussian_id: int = -1
    parent_id: int = -1
    child_ids: tuple[int, ...] = ()
    transaction_kind: int = 0
    all_active: bool = False

    def __post_init__(self) -> None:
        if self.iteration_id < 0 or self.state_version < 0 or self.field_mask < 0:
            raise ValueError("virtual lifecycle fields must be non-negative")
        if not isinstance(self.kind, VirtualLifecycleKind):
            raise ValueError("virtual lifecycle kind is invalid")
        if any(child < 0 for child in self.child_ids):
            raise ValueError("virtual lifecycle child IDs must be non-negative")
        if self.transaction_kind not in {0, TRANSACTION_COLLECTION, TRANSACTION_OPTIMIZER}:
            raise ValueError("virtual lifecycle transaction kind is invalid")
        if self.all_active and self.kind is not VirtualLifecycleKind.UPDATE_COMMIT:
            raise ValueError("all-active lifecycle records must be optimizer commits")


@dataclass(frozen=True)
class VirtualIterationLedger:
    """Audit counters retained after an iteration's packets are released."""

    iteration_id: int
    packet_count: int
    query_count: int
    candidate_count: int
    relation_count: int
    backward_relation_count: int
    optimizer_commits: int
    collection_transactions: int
    state_version_start: int
    state_version_end: int
    active_gaussian_count_start: int
    active_gaussian_count_end: int
    physical_stream_bytes: int


@dataclass
class VirtualTraceLifecycleValidator:
    """Validate packet counts and lifecycle transitions with bounded state."""

    initial_gaussian_count: int
    active_gaussians: set[int] | None = None
    state_version: int = 0
    current_iteration: int | None = None
    current_packets: int = 0
    current_queries: int = 0
    current_candidates: int = 0
    current_relations: int = 0
    current_backward_relations: int = 0
    current_optimizer_commits: int = 0
    current_collection_transactions: int = 0
    current_physical_bytes: int = 0
    _iteration_start_version: int = 0
    _iteration_start_gaussians: int = 0
    _open_update: tuple[int, int, int] | None = None
    _open_commit_expected: tuple[int, ...] = ()
    _open_commit_seen: set[int] = field(default_factory=set)
    _open_commit_all_active: bool = False
    _next_query_base: int | None = None
    _confirmed_query_bases: set[int] = field(default_factory=set)
    last_closed_iteration: int | None = None
    _ledgers: list[VirtualIterationLedger] | None = None

    def __post_init__(self) -> None:
        if self.initial_gaussian_count < 0:
            raise ValueError("initial Gaussian count must be non-negative")
        if self.active_gaussians is None:
            self.active_gaussians = set(range(self.initial_gaussian_count))
        if self._ledgers is None:
            self._ledgers = []

    @property
    def ledgers(self) -> tuple[VirtualIterationLedger, ...]:
        return tuple(self._ledgers or ())

    def accept_packet(self, packet: VirtualTracePacket) -> None:
        self._select_iteration(packet.iteration_id)
        if packet.state_version != self.state_version:
            raise ValueError("virtual packet state version does not match lifecycle state")
        if self._open_update is not None:
            raise ValueError("virtual query packet arrived inside an update transaction")
        expected_query_base = self._next_query_base
        if expected_query_base is not None and packet.query_base != expected_query_base:
            raise ValueError("virtual packet query base is not contiguous")
        self._next_query_base = packet.query_base + packet.query_count
        point_ids = np.asarray(packet.point_ids, dtype=np.int64)
        if point_ids.size and not set(int(value) for value in np.unique(point_ids)).issubset(
            self.active_gaussians or set()
        ):
            raise ValueError("virtual packet refers to an inactive Gaussian")
        relations = packet.logical_relation_count
        self.current_packets += 1
        self.current_queries += packet.query_count
        self.current_candidates += packet.candidate_count
        self.current_relations += relations
        self.current_physical_bytes += packet.physical_bytes
        if packet.backward_confirmed:
            self.accept_backward_confirmation(packet)

    def accept_backward_confirmation(self, packet: VirtualTracePacket) -> None:
        """Record the independent backward hook for an accepted packet."""

        self._select_iteration(packet.iteration_id)
        if self._next_query_base is None or (
            packet.query_base + packet.query_count > self._next_query_base
        ):
            raise ValueError("backward confirmation refers to an unknown query packet")
        marker = packet.query_base
        if marker in self._confirmed_query_bases:
            raise ValueError("virtual backward confirmation is duplicated")
        self._confirmed_query_bases.add(marker)
        self.current_backward_relations += packet.logical_relation_count

    def accept_lifecycle(self, record: VirtualLifecycleRecord) -> None:
        self._select_iteration(record.iteration_id)
        if record.state_version != self.state_version:
            raise ValueError("virtual lifecycle state version is not current")
        if record.kind is VirtualLifecycleKind.UPDATE_BEGIN:
            if self._open_update is not None:
                raise ValueError("virtual update transactions cannot overlap")
            if record.transaction_kind not in {TRANSACTION_COLLECTION, TRANSACTION_OPTIMIZER}:
                raise ValueError("virtual update begin requires a transaction kind")
            self._open_update = (
                record.transaction_kind, record.field_mask, record.iteration_id
            )
            self._open_commit_expected = (
                tuple(sorted(self.active_gaussians or ()))
                if record.transaction_kind == TRANSACTION_OPTIMIZER and record.field_mask
                else ()
            )
            self._open_commit_seen.clear()
            self._open_commit_all_active = False
            return
        if record.kind is VirtualLifecycleKind.UPDATE_COMMIT:
            if self._open_update is None:
                raise ValueError("virtual update commit has no begin")
            transaction_kind, begin_mask, _ = self._open_update
            if transaction_kind != TRANSACTION_OPTIMIZER:
                raise ValueError("virtual update commit is outside an optimizer transaction")
            if record.transaction_kind != transaction_kind or record.field_mask != begin_mask:
                raise ValueError("virtual update commit does not match its transaction")
            if not begin_mask:
                raise ValueError("no-op update cannot contain a commit")
            if transaction_kind != TRANSACTION_OPTIMIZER:
                raise ValueError("collection transaction cannot contain an optimizer commit")
            if record.all_active:
                if self._open_commit_all_active or self._open_commit_seen:
                    raise ValueError("virtual optimizer commit is duplicated")
                if record.gaussian_id >= 0:
                    raise ValueError("all-active optimizer commit cannot name one Gaussian")
                self._open_commit_all_active = True
                self.current_optimizer_commits += len(self.active_gaussians or ())
                return
            if record.gaussian_id not in (self.active_gaussians or set()):
                raise ValueError("virtual update commit refers to an inactive Gaussian")
            if record.gaussian_id in self._open_commit_seen:
                raise ValueError("virtual optimizer Gaussian commit is duplicated")
            if record.gaussian_id not in self._open_commit_expected:
                raise ValueError("virtual optimizer commit is outside its begin snapshot")
            self._open_commit_seen.add(record.gaussian_id)
            self.current_optimizer_commits += 1
            return
        if record.kind is VirtualLifecycleKind.PRUNE:
            self._require_collection()
            if record.gaussian_id not in (self.active_gaussians or set()):
                raise ValueError("virtual prune refers to an inactive Gaussian")
            self.active_gaussians.remove(record.gaussian_id)  # type: ignore[union-attr]
            return
        if record.kind in {VirtualLifecycleKind.CLONE, VirtualLifecycleKind.SPLIT}:
            self._require_collection()
            if record.parent_id not in (self.active_gaussians or set()):
                raise ValueError("virtual lineage parent is inactive")
            if not record.child_ids:
                raise ValueError("virtual lineage transition has no children")
            for child in record.child_ids:
                if child in (self.active_gaussians or set()):
                    raise ValueError("virtual lineage child is already active")
                self.active_gaussians.add(child)  # type: ignore[union-attr]
            if record.kind is VirtualLifecycleKind.SPLIT:
                self.active_gaussians.remove(record.parent_id)  # type: ignore[union-attr]
            return
        if record.kind is VirtualLifecycleKind.UPDATE_END:
            if self._open_update is None:
                raise ValueError("virtual update end has no begin")
            transaction_kind, begin_mask, _ = self._open_update
            if record.transaction_kind != transaction_kind:
                raise ValueError("virtual update end does not match its transaction")
            if record.field_mask != begin_mask:
                raise ValueError("virtual update masks do not match")
            if transaction_kind == TRANSACTION_OPTIMIZER and begin_mask:
                if not self._open_commit_all_active and (
                    self._open_commit_seen != set(self._open_commit_expected)
                ):
                    raise ValueError("virtual optimizer transaction has incomplete commits")
            if begin_mask:
                self.state_version += 1
            if transaction_kind == TRANSACTION_COLLECTION:
                self.current_collection_transactions += 1
            self._open_update = None
            self._open_commit_expected = ()
            self._open_commit_seen.clear()
            self._open_commit_all_active = False
            return
        raise ValueError("unsupported virtual lifecycle record")

    def close_iteration(self, iteration_id: int) -> VirtualIterationLedger:
        if self.current_iteration != iteration_id:
            raise ValueError("virtual iteration close does not match the active iteration")
        if self._open_update is not None:
            raise ValueError("virtual iteration ended with an open update transaction")
        if self.current_relations != self.current_backward_relations:
            raise ValueError("virtual forward/backward relation counts differ")
        ledger = VirtualIterationLedger(
            iteration_id=iteration_id,
            packet_count=self.current_packets,
            query_count=self.current_queries,
            candidate_count=self.current_candidates,
            relation_count=self.current_relations,
            backward_relation_count=self.current_backward_relations,
            optimizer_commits=self.current_optimizer_commits,
            collection_transactions=self.current_collection_transactions,
            state_version_start=self._iteration_start_version,
            state_version_end=self.state_version,
            active_gaussian_count_start=self._iteration_start_gaussians,
            active_gaussian_count_end=len(self.active_gaussians or ()),
            physical_stream_bytes=self.current_physical_bytes,
        )
        self._ledgers.append(ledger)  # type: ignore[union-attr]
        self.last_closed_iteration = iteration_id
        self.current_iteration = None
        self._reset_iteration_counters()
        return ledger

    def finalize(self) -> tuple[VirtualIterationLedger, ...]:
        if self.current_iteration is not None:
            raise ValueError("virtual trace ended before its active iteration closed")
        if self._open_update is not None:
            raise ValueError("virtual trace ended with an open update transaction")
        return self.ledgers

    def _select_iteration(self, iteration_id: int) -> None:
        if self.last_closed_iteration is not None and iteration_id <= self.last_closed_iteration:
            raise ValueError("virtual iteration is not strictly increasing")
        if self.current_iteration is None:
            self.current_iteration = iteration_id
            self._iteration_start_version = self.state_version
            self._iteration_start_gaussians = len(self.active_gaussians or ())
            return
        if iteration_id != self.current_iteration:
            raise ValueError("virtual iteration changed without an explicit close")

    def _require_collection(self) -> None:
        if self._open_update is None or self._open_update[0] != TRANSACTION_COLLECTION:
            raise ValueError("Gaussian set modification requires a collection transaction")

    def _reset_iteration_counters(self) -> None:
        self.current_packets = 0
        self.current_queries = 0
        self.current_candidates = 0
        self.current_relations = 0
        self.current_backward_relations = 0
        self.current_optimizer_commits = 0
        self.current_collection_transactions = 0
        self.current_physical_bytes = 0


@dataclass(frozen=True)
class VirtualTracePacket:
    """One exact raster or voxel work-buffer packet.

    ``masks[candidate, word]`` has the same bit numbering as the CUDA
    ``atomicOr`` buffers: bit ``word * MASK_WORD_BITS + bit`` identifies a
    local query in the candidate's tile.  The packet owns no expanded event or
    dependency array, so its resident size is bounded by the input work
    buffer.
    """

    iteration_id: int
    template_id: int
    query_base: int
    query_shape: tuple[int, ...]
    point_ids: np.ndarray
    point_keys: np.ndarray
    masks: np.ndarray
    state_version: int = 0
    field_mask: int = 0
    loss_flags: int = 0
    ssim_radius: int = 0
    backward_confirmed: bool = False

    def __post_init__(self) -> None:
        if self.iteration_id < 0 or self.template_id < 0 or self.query_base < 0:
            raise ValueError("virtual trace packet identifiers must be non-negative")
        if not self.query_shape or any(int(value) <= 0 for value in self.query_shape):
            raise ValueError("virtual trace packet query shape must be positive")
        if (
            self.state_version < 0 or self.field_mask < 0
            or self.loss_flags < 0 or self.ssim_radius < 0
        ):
            raise ValueError("virtual trace packet state fields must be non-negative")
        point_ids = np.asarray(self.point_ids)
        point_keys = np.asarray(self.point_keys)
        masks = np.asarray(self.masks)
        if point_ids.ndim != 1 or point_keys.ndim != 1 or masks.ndim != 2:
            raise ValueError("virtual trace packet arrays have invalid dimensions")
        if point_ids.size != point_keys.size or masks.shape[0] != point_ids.size:
            raise ValueError("virtual trace packet candidate arrays have different lengths")
        if point_ids.dtype.kind not in "iu" or point_keys.dtype.kind not in "iu":
            raise ValueError("virtual trace point arrays must be integer arrays")
        if masks.dtype != np.dtype("<u4"):
            raise ValueError("virtual trace masks must use little-endian uint32 words")
        required_words = self.local_query_count // MASK_WORD_BITS
        if masks.shape[1] != required_words:
            raise ValueError(
                "virtual trace mask word count does not match the packet query shape"
            )
        if np.any(point_ids < 0) or np.any(point_keys < 0):
            raise ValueError("virtual trace point identifiers must be non-negative")
        # Mask off padding bits.  Padding is not part of the official query
        # domain and accepting it would create relations with invalid IDs.
        # Edge-tile bits outside the full query shape must be clear.  The
        # allowed words are built once per tile extent and checked in a
        # vectorized candidate pass; validating every set bit in Python would
        # recreate the dense-trace bottleneck this representation avoids.
        tiles = np.right_shift(np.asarray(point_keys, dtype=np.uint64), 32)
        tile_count = self._tile_count()
        if tiles.size and int(tiles.max()) >= tile_count:
            raise ValueError("virtual trace point key refers to an invalid tile")
        allowed = self._allowed_tile_words()
        if masks.size and np.any(masks & ~allowed[tiles]):
            raise ValueError("virtual trace mask refers outside the output")

    @property
    def query_count(self) -> int:
        return int(np.prod(self.query_shape, dtype=np.int64))

    @property
    def candidate_count(self) -> int:
        return int(self.point_ids.size)

    @property
    def local_query_count(self) -> int:
        if self.template_id == RASTER_TEMPLATE_ID:
            return int(np.prod(RASTER_BLOCK, dtype=np.int64))
        if self.template_id == VOXEL_TEMPLATE_ID:
            return int(np.prod(VOXEL_BLOCK, dtype=np.int64))
        raise ValueError(f"unsupported virtual trace template: {self.template_id}")

    def _tile_count(self) -> int:
        if self.template_id == RASTER_TEMPLATE_ID:
            height, width = self.query_shape
            return (
                (width + RASTER_BLOCK[1] - 1) // RASTER_BLOCK[1]
                * ((height + RASTER_BLOCK[0] - 1) // RASTER_BLOCK[0])
            )
        if self.template_id == VOXEL_TEMPLATE_ID:
            voxel_x, voxel_y, voxel_z = self.query_shape
            return (
                (voxel_x + VOXEL_BLOCK[0] - 1) // VOXEL_BLOCK[0]
                * ((voxel_y + VOXEL_BLOCK[1] - 1) // VOXEL_BLOCK[1])
                * ((voxel_z + VOXEL_BLOCK[2] - 1) // VOXEL_BLOCK[2])
            )
        raise ValueError(f"unsupported virtual trace template: {self.template_id}")

    def _allowed_tile_words(self) -> np.ndarray:
        if self.template_id == RASTER_TEMPLATE_ID:
            height, width = self.query_shape
            blocks_x = (width + RASTER_BLOCK[1] - 1) // RASTER_BLOCK[1]
            blocks_y = (height + RASTER_BLOCK[0] - 1) // RASTER_BLOCK[0]
            extents = tuple(
                (
                    min(RASTER_BLOCK[0], height - (tile // blocks_x) * RASTER_BLOCK[0]),
                    min(RASTER_BLOCK[1], width - (tile % blocks_x) * RASTER_BLOCK[1]),
                )
                for tile in range(blocks_x * blocks_y)
            )
            return np.asarray([
                _local_valid_words(RASTER_TEMPLATE_ID, *extent)
                for extent in extents
            ], dtype=np.dtype("<u4"))
        voxel_x, voxel_y, voxel_z = self.query_shape
        blocks_x = (voxel_x + VOXEL_BLOCK[0] - 1) // VOXEL_BLOCK[0]
        blocks_y = (voxel_y + VOXEL_BLOCK[1] - 1) // VOXEL_BLOCK[1]
        blocks_z = (voxel_z + VOXEL_BLOCK[2] - 1) // VOXEL_BLOCK[2]
        extents = []
        for tile in range(blocks_x * blocks_y * blocks_z):
            tile_x = tile % blocks_x
            tile_y = (tile // blocks_x) % blocks_y
            tile_z = tile // (blocks_x * blocks_y)
            extents.append((
                min(VOXEL_BLOCK[0], voxel_x - tile_x * VOXEL_BLOCK[0]),
                min(VOXEL_BLOCK[1], voxel_y - tile_y * VOXEL_BLOCK[1]),
                min(VOXEL_BLOCK[2], voxel_z - tile_z * VOXEL_BLOCK[2]),
            ))
        return np.asarray([
            _local_valid_words(VOXEL_TEMPLATE_ID, *extent) for extent in extents
        ], dtype=np.dtype("<u4"))

    @property
    def mask_bytes(self) -> int:
        return int(self.masks.nbytes)

    @property
    def physical_bytes(self) -> int:
        return int(self.point_ids.nbytes + self.point_keys.nbytes + self.masks.nbytes)

    @property
    def logical_relation_count(self) -> int:
        """Count exact set bits without constructing relation records."""

        if self.masks.size == 0:
            return 0
        # A byte lookup avoids relying on a NumPy version-specific bit_count
        # ufunc while keeping the temporary bounded to the mask packet.
        table = _POPCOUNT8
        return int(table[self.masks.view(np.uint8)].sum(dtype=np.uint64))

    def iter_candidates(self) -> Iterator[tuple[int, int, int, int, bool]]:
        """Yield ``(index, point_id, key, state_version, has_relation)``."""

        for index, (point_id, point_key, words) in enumerate(
            zip(self.point_ids, self.point_keys, self.masks, strict=True)
        ):
            yield (
                index,
                int(point_id),
                int(point_key),
                int(self.state_version),
                bool(np.any(words)),
            )

    def iter_relations(self) -> Iterator[tuple[int, int, int, int]]:
        """Yield exact relations in the decoder's stable query-major order.

        The CUDA decoder writes relation records candidate-major and the
        capture path then performs a stable sort by local query offset.  A
        query-major scan of each mask bit produces the same ordering without
        retaining a relation-sized index array.  Each tuple is
        ``(candidate_index, global_query_id, gaussian_id, point_key)``.
        """

        for batch in self.iter_relation_arrays(max_relations=65536):
            for relation in zip(*batch, strict=True):
                yield tuple(int(value) for value in relation)

    def iter_relation_arrays(
        self, max_relations: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield bounded query-major relation columns without per-relation callbacks."""

        if max_relations <= 0:
            raise ValueError("virtual trace relation batch size must be positive")
        tiles = np.right_shift(np.asarray(self.point_keys, dtype=np.uint64), 32)
        order = np.argsort(tiles, kind="stable")
        sorted_tiles = tiles[order]
        tile_count = self._tile_count()
        starts = np.searchsorted(sorted_tiles, np.arange(tile_count), side="left")
        ends = np.searchsorted(sorted_tiles, np.arange(tile_count), side="right")
        candidate_parts: list[np.ndarray] = []
        query_parts: list[np.ndarray] = []
        buffered = 0

        def flush() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            candidates = np.concatenate(candidate_parts).astype(np.int64, copy=False)
            queries = np.concatenate(query_parts).astype(np.int64, copy=False)
            return (
                candidates,
                queries,
                np.asarray(self.point_ids[candidates], dtype=np.int64),
                np.asarray(self.point_keys[candidates], dtype=np.uint64),
            )

        for query_offset in range(self.query_count):
            tile, local_query = self._tile_and_local_query(query_offset)
            tile_candidates = order[starts[tile]:ends[tile]]
            if tile_candidates.size == 0:
                continue
            word_index, bit_index = divmod(local_query, MASK_WORD_BITS)
            bit = np.uint32(1 << bit_index)
            selected = tile_candidates[
                (self.masks[tile_candidates, word_index] & bit) != 0
            ]
            selected_offset = 0
            while selected_offset < selected.size:
                capacity = max_relations - buffered
                take = min(capacity, selected.size - selected_offset)
                part = np.asarray(
                    selected[selected_offset:selected_offset + take], dtype=np.int64
                )
                candidate_parts.append(part)
                query_parts.append(np.full(
                    take, self.query_base + query_offset, dtype=np.int64
                ))
                buffered += take
                selected_offset += take
                if buffered == max_relations:
                    yield flush()
                    candidate_parts.clear()
                    query_parts.clear()
                    buffered = 0
        if buffered:
            yield flush()

    def _tile_and_local_query(self, query_offset: int) -> tuple[int, int]:
        if self.template_id == RASTER_TEMPLATE_ID:
            height, width = self.query_shape
            blocks_x = (width + RASTER_BLOCK[1] - 1) // RASTER_BLOCK[1]
            y, x = divmod(query_offset, width)
            tile = (y // RASTER_BLOCK[0]) * blocks_x + x // RASTER_BLOCK[1]
            local = (y % RASTER_BLOCK[0]) * RASTER_BLOCK[1] + x % RASTER_BLOCK[1]
            return tile, local
        if self.template_id == VOXEL_TEMPLATE_ID:
            voxel_x, voxel_y, voxel_z = self.query_shape
            blocks_x = (voxel_x + VOXEL_BLOCK[0] - 1) // VOXEL_BLOCK[0]
            blocks_y = (voxel_y + VOXEL_BLOCK[1] - 1) // VOXEL_BLOCK[1]
            x, remainder = divmod(query_offset, voxel_y * voxel_z)
            y, z = divmod(remainder, voxel_z)
            tile = (
                (z // VOXEL_BLOCK[2]) * blocks_x * blocks_y
                + (y // VOXEL_BLOCK[1]) * blocks_x
                + x // VOXEL_BLOCK[0]
            )
            local = (
                (x % VOXEL_BLOCK[0]) * VOXEL_BLOCK[1] * VOXEL_BLOCK[2]
                + (y % VOXEL_BLOCK[1]) * VOXEL_BLOCK[2]
                + z % VOXEL_BLOCK[2]
            )
            return tile, local
        raise ValueError(f"unsupported virtual trace template: {self.template_id}")

    def iter_relations_candidate_order(self) -> Iterator[tuple[int, int, int, int]]:
        """Yield the raw CUDA candidate-major relation order."""

        for candidate, words in enumerate(self.masks):
            tile = int(np.uint64(self.point_keys[candidate]) >> 32)
            for word_index, raw_word in enumerate(words):
                bits = int(raw_word)
                while bits:
                    bit_index = (bits & -bits).bit_length() - 1
                    local_query = word_index * MASK_WORD_BITS + bit_index
                    query_id = self.query_base + self._query_offset(tile, local_query)
                    yield (
                        candidate,
                        query_id,
                        int(self.point_ids[candidate]),
                        int(self.point_keys[candidate]),
                    )
                    bits &= bits - 1

    def _query_offset(self, tile: int, local_query: int) -> int:
        if self.template_id == RASTER_TEMPLATE_ID:
            height, width = self.query_shape
            blocks_x = (width + RASTER_BLOCK[1] - 1) // RASTER_BLOCK[1]
            x = (tile % blocks_x) * RASTER_BLOCK[1] + local_query % RASTER_BLOCK[1]
            y = (tile // blocks_x) * RASTER_BLOCK[0] + local_query // RASTER_BLOCK[1]
            if x >= width or y >= height:
                raise ValueError("virtual raster mask refers outside the output")
            return y * width + x
        if self.template_id == VOXEL_TEMPLATE_ID:
            voxel_x, voxel_y, voxel_z = self.query_shape
            blocks_x = (voxel_x + VOXEL_BLOCK[0] - 1) // VOXEL_BLOCK[0]
            blocks_y = (voxel_y + VOXEL_BLOCK[1] - 1) // VOXEL_BLOCK[1]
            tile_x = tile % blocks_x
            tile_y = (tile // blocks_x) % blocks_y
            tile_z = tile // (blocks_x * blocks_y)
            local_x = local_query % VOXEL_BLOCK[0]
            local_y = (local_query // VOXEL_BLOCK[0]) % VOXEL_BLOCK[1]
            local_z = local_query // (VOXEL_BLOCK[0] * VOXEL_BLOCK[1])
            x = tile_x * VOXEL_BLOCK[0] + local_x
            y = tile_y * VOXEL_BLOCK[1] + local_y
            z = tile_z * VOXEL_BLOCK[2] + local_z
            if x >= voxel_x or y >= voxel_y or z >= voxel_z:
                raise ValueError("virtual voxel mask refers outside the output")
            return x * voxel_y * voxel_z + y * voxel_z + z
        raise ValueError(f"unsupported virtual trace template: {self.template_id}")

    def iter_relation_batches(
        self, max_relations: int
    ) -> Iterator[tuple[tuple[int, int, int, int], ...]]:
        """Yield deterministic bounded batches of exact relation tuples."""

        if max_relations <= 0:
            raise ValueError("virtual trace relation batch size must be positive")
        for columns in self.iter_relation_arrays(max_relations):
            yield tuple(
                tuple(int(value) for value in relation)
                for relation in zip(*columns, strict=True)
            )

    def materialize_relations(self, *, max_relations: int | None = None) -> np.ndarray:
        """Materialize only when explicitly bounded by the caller."""

        relation_count = self.logical_relation_count
        if max_relations is not None and relation_count > max_relations:
            raise ValueError("virtual trace relation materialization exceeds its bound")
        if relation_count == 0:
            return np.empty((0, 4), dtype=np.int64)
        parts = [np.column_stack(columns) for columns in self.iter_relation_arrays(
            max(relation_count, 1)
        )]
        return np.concatenate(parts, axis=0).astype(np.int64, copy=False)


@dataclass(frozen=True)
class VirtualEventPacket:
    """A bounded event batch with IDs valid in one global event stream.

    Dependency offsets are local to ``dependencies`` while dependency values
    themselves are global dense event IDs.  The explicit packet boundary is
    therefore transport-only: it cannot silently rebase or drop a dependency
    belonging to an earlier packet.
    """

    packet_id: int
    global_event_start: int
    events: np.ndarray
    dependencies: np.ndarray
    final_packet: bool = False
    frontier_complete: bool = False

    def __post_init__(self) -> None:
        if self.packet_id < 0 or self.global_event_start < 0:
            raise ValueError("virtual event packet identifiers must be non-negative")
        if self.events.dtype != event_dtype() or self.events.ndim != 1:
            raise ValueError("virtual event packet events do not use the frozen schema")
        if self.dependencies.dtype != dependency_dtype() or self.dependencies.ndim != 1:
            raise ValueError("virtual event packet dependencies use an invalid schema")
        expected_ids = np.arange(
            self.global_event_start,
            self.global_event_start + self.events.size,
            dtype=np.uint64,
        )
        if not np.array_equal(self.events["event_id"], expected_ids):
            raise ValueError("virtual event packet event IDs are not globally contiguous")
        if self.events.size:
            for row in self.events:
                begin = int(row["dependency_begin"])
                end = begin + int(row["dependency_count"])
                if begin < 0 or end > self.dependencies.size:
                    raise ValueError("virtual event packet dependency range is invalid")
                if end > begin and np.any(self.dependencies[begin:end] >= row["event_id"]):
                    raise ValueError("virtual event packet has a forward dependency")
        if self.final_packet and not self.frontier_complete:
            raise ValueError("final virtual event packet must close its dependency frontier")

    @property
    def event_count(self) -> int:
        return int(self.events.size)

    @property
    def global_event_end(self) -> int:
        return self.global_event_start + self.event_count

    @property
    def external_dependencies(self) -> np.ndarray:
        """Return dependencies outside this packet's global event range."""

        if self.dependencies.size == 0:
            return np.empty(0, dtype=dependency_dtype())
        external = self.dependencies[self.dependencies < self.global_event_start]
        return np.unique(external).astype(dependency_dtype(), copy=False)

    def dependency_ids(self, event_index: int) -> np.ndarray:
        if event_index < 0 or event_index >= self.event_count:
            raise IndexError("virtual event packet event index is out of range")
        row = self.events[event_index]
        begin = int(row["dependency_begin"])
        end = begin + int(row["dependency_count"])
        return self.dependencies[begin:end]


@dataclass
class VirtualRelationEventExpander:
    """Expand one work-buffer packet into bounded global-ID event packets.

    This deliberately emits only the relation-construction prefix.  Later
    CLAMP stages must consume the same global IDs and carry their state across
    packet boundaries; pretending that this prefix is a complete trace would
    make cache and update lifetimes unverifiable.
    """

    max_events: int
    state_record_bytes: int = 128
    relation_candidate_bytes: int = 0
    next_event_id: int = 0
    next_relation_id: int = 0
    next_packet_id: int = 0

    def __post_init__(self) -> None:
        if self.max_events <= 0 or self.state_record_bytes <= 0:
            raise ValueError("virtual event expander limits must be positive")
        if self.relation_candidate_bytes < 0:
            raise ValueError("relation candidate bytes must be non-negative")

    def expand(self, packet: VirtualTracePacket) -> Iterator[VirtualEventPacket]:
        candidate_start = self.next_event_id
        relation_start = candidate_start + packet.candidate_count
        relation_count = packet.logical_relation_count
        self.next_event_id = relation_start + relation_count
        relation_base = self.next_relation_id
        self.next_relation_id += relation_count

        candidate_ids = np.arange(
            candidate_start, relation_start, dtype=np.uint64
        )
        for start in range(0, packet.candidate_count, self.max_events):
            end = min(start + self.max_events, packet.candidate_count)
            rows = np.empty(end - start, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = candidate_ids[start:end]
            rows["iteration_id"] = packet.iteration_id
            rows["primitive_kind"] = int(PrimitiveKind.RELATION_CANDIDATE)
            rows["gaussian_id"] = packet.point_ids[start:end]
            rows["state_version"] = packet.state_version
            rows["resource_class"] = int(ResourceClass.RELATION)
            rows["address_token"] = packet.point_keys[start:end]
            rows["data_bytes"] = self.relation_candidate_bytes
            rows["template_id"] = packet.template_id
            rows["field_mask"] = packet.field_mask
            rows["flags"] = np.any(packet.masks[start:end], axis=1)
            yield self._make_event_packet(rows, np.empty(0, dtype=dependency_dtype()))

        emitted_relations = 0
        for columns in packet.iter_relation_arrays(self.max_events):
            yield self._relation_packet(
                columns, packet, candidate_start, relation_start,
                relation_base, emitted_relations,
            )
            emitted_relations += int(columns[0].size)

    def _relation_packet(
        self,
        relations: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        source: VirtualTracePacket,
        candidate_start: int,
        relation_start: int,
        relation_base: int,
        emitted_before: int,
    ) -> VirtualEventPacket:
        candidates, query_ids, gaussian_ids, _point_keys = relations
        relation_count = int(candidates.size)
        rows = np.empty(relation_count, dtype=event_dtype())
        rows[:] = TraceEvent().as_tuple()
        rows["event_id"] = np.arange(
            relation_start + emitted_before,
            relation_start + emitted_before + relation_count,
            dtype=np.uint64,
        )
        rows["iteration_id"] = source.iteration_id
        rows["primitive_kind"] = int(PrimitiveKind.RELATION)
        rows["query_id"] = query_ids
        rows["gaussian_id"] = gaussian_ids
        rows["state_version"] = source.state_version
        rows["relation_id"] = np.arange(
            relation_base + emitted_before,
            relation_base + emitted_before + relation_count,
            dtype=np.int64,
        )
        rows["resource_class"] = int(ResourceClass.RELATION)
        rows["template_id"] = source.template_id
        rows["field_mask"] = source.field_mask
        rows["address_token"] = np.asarray(
            gaussian_ids * self.state_record_bytes, dtype=np.uint64
        )
        dependencies = np.asarray(
            candidate_start + candidates, dtype=dependency_dtype()
        )
        return self._make_event_packet(rows, dependencies)

    def _make_event_packet(
        self, rows: np.ndarray, dependencies: np.ndarray
    ) -> VirtualEventPacket:
        rows = np.asarray(rows, dtype=event_dtype())
        if dependencies.size:
            rows["dependency_begin"] = np.arange(
                dependencies.size, dtype=np.uint64
            )
            rows["dependency_count"] = 1
        packet = VirtualEventPacket(
            packet_id=self.next_packet_id,
            global_event_start=int(rows["event_id"][0]) if rows.size else self.next_event_id,
            events=rows,
            dependencies=np.asarray(dependencies, dtype=dependency_dtype()),
        )
        self.next_packet_id += 1
        return packet


@dataclass
class VirtualQueryEventExpander:
    """Expand one packet through the complete query/backward event chain.

    The expansion is still bounded: relation columns are rescanned in
    ``max_events`` batches for each stage, and only query-sized dependency
    frontiers are retained while a packet is emitted.  Update transactions and
    cross-iteration lineage are intentionally supplied by the lifecycle
    expander, not synthesized here.
    """

    max_events: int
    state_record_bytes: int = 128
    relation_candidate_bytes: int = 0
    next_event_id: int = 0
    next_relation_id: int = 0
    next_packet_id: int = 0

    def __post_init__(self) -> None:
        if self.max_events <= 0 or self.state_record_bytes <= 0:
            raise ValueError("virtual query expander limits must be positive")
        if self.relation_candidate_bytes < 0:
            raise ValueError("relation candidate bytes must be non-negative")

    def expand(self, source: VirtualTracePacket) -> Iterator[VirtualEventPacket]:
        if source.loss_flags == 0:
            raise ValueError("virtual query expansion requires a captured loss consumer")
        candidate_start = self.next_event_id
        relation_start = candidate_start + source.candidate_count
        relation_count = source.logical_relation_count
        relation_base = self.next_relation_id
        query_count = source.query_count
        close_start = relation_start + relation_count
        request_start = close_start + query_count
        return_start = request_start + relation_count
        forward_start = return_start + relation_count
        reduction_start = forward_start + relation_count
        consumer_start = reduction_start + query_count
        adjoint_start = consumer_start + query_count
        gradient_start = adjoint_start + relation_count
        self.next_event_id = gradient_start + relation_count
        self.next_relation_id += relation_count

        yield from self._candidates(source, candidate_start)
        yield from self._relations(
            source, candidate_start, relation_start, relation_base, emitted_before=0
        )
        counts = self._relation_counts(source)
        yield from self._query_closes(source, close_start, relation_start, counts)
        yield from self._one_dependency_stage(
            source, PrimitiveKind.CACHE_REQUEST, ResourceClass.CACHE,
            request_start, relation_start, source.candidate_count,
            relation_base, source,
        )
        yield from self._one_dependency_stage(
            source, PrimitiveKind.CACHE_RETURN, ResourceClass.CACHE,
            return_start, request_start, source.candidate_count,
            relation_base, source,
        )
        yield from self._forward_stage(
            source, forward_start, relation_start, return_start, relation_base
        )
        yield from self._reductions(source, reduction_start, close_start, forward_start, counts)
        yield from self._consumers(source, consumer_start, reduction_start)
        yield from self._adjoint_gradient(
            source, adjoint_start, gradient_start, consumer_start,
            relation_base,
        )

    def _candidates(
        self, source: VirtualTracePacket, event_start: int
    ) -> Iterator[VirtualEventPacket]:
        for start in range(0, source.candidate_count, self.max_events):
            end = min(start + self.max_events, source.candidate_count)
            rows = np.empty(end - start, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(event_start + start, event_start + end, dtype=np.uint64)
            rows["iteration_id"] = source.iteration_id
            rows["primitive_kind"] = int(PrimitiveKind.RELATION_CANDIDATE)
            rows["gaussian_id"] = source.point_ids[start:end]
            rows["state_version"] = source.state_version
            rows["resource_class"] = int(ResourceClass.RELATION)
            rows["address_token"] = source.point_keys[start:end]
            rows["data_bytes"] = self.relation_candidate_bytes
            rows["template_id"] = source.template_id
            rows["field_mask"] = source.field_mask
            rows["flags"] = np.any(source.masks[start:end], axis=1)
            yield self._make_event_packet(rows, np.empty(0, dtype=dependency_dtype()))

    def _relations(
        self,
        source: VirtualTracePacket,
        candidate_start: int,
        event_start: int,
        relation_base: int,
        *,
        emitted_before: int,
    ) -> Iterator[VirtualEventPacket]:
        for columns in source.iter_relation_arrays(self.max_events):
            candidates, query_ids, gaussian_ids, _keys = columns
            count = int(candidates.size)
            rows = np.empty(count, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(
                event_start + emitted_before,
                event_start + emitted_before + count,
                dtype=np.uint64,
            )
            rows["iteration_id"] = source.iteration_id
            rows["primitive_kind"] = int(PrimitiveKind.RELATION)
            rows["query_id"] = query_ids
            rows["gaussian_id"] = gaussian_ids
            rows["state_version"] = source.state_version
            rows["relation_id"] = np.arange(
                relation_base + emitted_before,
                relation_base + emitted_before + count,
                dtype=np.int64,
            )
            rows["resource_class"] = int(ResourceClass.RELATION)
            rows["address_token"] = gaussian_ids * self.state_record_bytes
            rows["template_id"] = source.template_id
            rows["field_mask"] = source.field_mask
            dependencies = np.asarray(
                candidate_start + candidates, dtype=dependency_dtype()
            )
            emitted_before += count
            yield self._make_event_packet(rows, dependencies)

    def _relation_counts(self, source: VirtualTracePacket) -> np.ndarray:
        counts = np.zeros(source.query_count, dtype=np.uint64)
        for _candidates, query_ids, _gaussians, _keys in source.iter_relation_arrays(self.max_events):
            offsets = query_ids - source.query_base
            np.add.at(counts, offsets, 1)
        return counts

    def _query_closes(
        self,
        source: VirtualTracePacket,
        event_start: int,
        relation_start: int,
        counts: np.ndarray,
    ) -> Iterator[VirtualEventPacket]:
        relation_prefix = np.empty(counts.size + 1, dtype=np.uint64)
        relation_prefix[0] = 0
        np.cumsum(counts, out=relation_prefix[1:])
        for start in range(0, source.query_count, self.max_events):
            end = min(start + self.max_events, source.query_count)
            rows = np.empty(end - start, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(event_start + start, event_start + end, dtype=np.uint64)
            rows["iteration_id"] = source.iteration_id
            rows["primitive_kind"] = int(PrimitiveKind.QUERY_CLOSE)
            rows["query_id"] = source.query_base + np.arange(start, end, dtype=np.int64)
            rows["state_version"] = source.state_version
            rows["resource_class"] = int(ResourceClass.RELATION)
            rows["template_id"] = source.template_id
            rows["field_mask"] = source.field_mask
            dependency_parts = [
                np.arange(
                    relation_start + int(relation_prefix[index]),
                    relation_start + int(relation_prefix[index + 1]),
                    dtype=dependency_dtype(),
                )
                for index in range(start, end)
            ]
            dependencies = np.concatenate(
                dependency_parts or [np.empty(0, dtype=dependency_dtype())]
            )
            local_prefix = np.empty((end - start) + 1, dtype=np.uint64)
            local_prefix[0] = 0
            np.cumsum(counts[start:end], out=local_prefix[1:])
            rows["dependency_begin"] = np.asarray(
                local_prefix[:-1], dtype=np.uint64,
            )
            rows["dependency_count"] = counts[start:end].astype(np.uint32, copy=False)
            yield self._make_event_packet(rows, dependencies)

    def _one_dependency_stage(
        self,
        source: VirtualTracePacket,
        primitive: PrimitiveKind,
        resource: ResourceClass,
        event_start: int,
        dependency_start: int,
        candidate_count: int,
        relation_base: int,
        _unused_source: VirtualTracePacket,
    ) -> Iterator[VirtualEventPacket]:
        emitted = 0
        for columns in source.iter_relation_arrays(self.max_events):
            _candidates, query_ids, gaussian_ids, _keys = columns
            count = int(query_ids.size)
            rows = np.empty(count, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(event_start + emitted, event_start + emitted + count, dtype=np.uint64)
            rows["iteration_id"] = source.iteration_id
            rows["primitive_kind"] = int(primitive)
            rows["query_id"] = query_ids
            rows["gaussian_id"] = gaussian_ids
            rows["state_version"] = source.state_version
            rows["relation_id"] = np.arange(relation_base + emitted, relation_base + emitted + count, dtype=np.int64)
            rows["resource_class"] = int(resource)
            rows["address_token"] = gaussian_ids * self.state_record_bytes
            if primitive in {PrimitiveKind.CACHE_REQUEST, PrimitiveKind.CACHE_RETURN}:
                rows["data_bytes"] = self.state_record_bytes
            rows["template_id"] = source.template_id
            rows["field_mask"] = source.field_mask
            dependencies = np.arange(dependency_start + emitted, dependency_start + emitted + count, dtype=dependency_dtype())
            emitted += count
            yield self._make_event_packet(rows, dependencies)

    def _forward_stage(
        self, source: VirtualTracePacket, event_start: int,
        relation_start: int, return_start: int, relation_base: int,
    ) -> Iterator[VirtualEventPacket]:
        emitted = 0
        for columns in source.iter_relation_arrays(self.max_events):
            _candidates, query_ids, gaussian_ids, _keys = columns
            count = int(query_ids.size)
            rows = np.empty(count, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(event_start + emitted, event_start + emitted + count, dtype=np.uint64)
            rows["iteration_id"] = source.iteration_id
            rows["primitive_kind"] = int(PrimitiveKind.FORWARD)
            rows["query_id"] = query_ids
            rows["gaussian_id"] = gaussian_ids
            rows["state_version"] = source.state_version
            rows["relation_id"] = np.arange(
                relation_base + emitted, relation_base + emitted + count,
                dtype=np.int64,
            )
            rows["resource_class"] = int(ResourceClass.ISSUE)
            rows["address_token"] = gaussian_ids * self.state_record_bytes
            rows["template_id"] = source.template_id
            rows["field_mask"] = source.field_mask
            dependencies = np.empty(count * 2, dtype=dependency_dtype())
            relation_ids = relation_start + emitted + np.arange(
                count, dtype=dependency_dtype()
            )
            dependencies[0::2] = relation_ids
            dependencies[1::2] = return_start + emitted + np.arange(
                count, dtype=dependency_dtype()
            )
            rows["dependency_begin"] = np.arange(0, count * 2, 2, dtype=np.uint64)
            rows["dependency_count"] = 2
            emitted += count
            yield self._make_event_packet(rows, dependencies)

    def _reductions(
        self, source: VirtualTracePacket, event_start: int,
        close_start: int, forward_start: int, counts: np.ndarray,
    ) -> Iterator[VirtualEventPacket]:
        prefix = np.empty(counts.size + 1, dtype=np.uint64)
        prefix[0] = 0
        np.cumsum(counts, out=prefix[1:])
        for start in range(0, source.query_count, self.max_events):
            end = min(start + self.max_events, source.query_count)
            rows = np.empty(end - start, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(event_start + start, event_start + end, dtype=np.uint64)
            rows["iteration_id"] = source.iteration_id
            rows["primitive_kind"] = int(PrimitiveKind.QUERY_REDUCTION)
            rows["query_id"] = source.query_base + np.arange(start, end, dtype=np.int64)
            rows["state_version"] = source.state_version
            rows["reduction_key"] = rows["query_id"]
            rows["resource_class"] = int(ResourceClass.QUERY)
            rows["template_id"] = source.template_id
            rows["field_mask"] = source.field_mask
            dependency_parts = []
            for query in range(start, end):
                dependency_parts.append(np.concatenate((
                    np.asarray([close_start + query], dtype=dependency_dtype()),
                    np.arange(
                        forward_start + int(prefix[query]),
                        forward_start + int(prefix[query + 1]),
                        dtype=dependency_dtype(),
                    ),
                )))
            dependencies = np.concatenate(
                dependency_parts or [np.empty(0, dtype=dependency_dtype())]
            )
            local_counts = counts[start:end] + 1
            local_prefix = np.empty((end - start) + 1, dtype=np.uint64)
            local_prefix[0] = 0
            np.cumsum(local_counts, out=local_prefix[1:])
            rows["dependency_begin"] = np.asarray(
                local_prefix[:-1], dtype=np.uint64,
            )
            rows["dependency_count"] = local_counts.astype(np.uint32, copy=False)
            yield self._make_event_packet(rows, dependencies)

    def _consumers(
        self, source: VirtualTracePacket, event_start: int, reduction_start: int,
    ) -> Iterator[VirtualEventPacket]:
        for start in range(0, source.query_count, self.max_events):
            end = min(start + self.max_events, source.query_count)
            rows = np.empty(end - start, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(event_start + start, event_start + end, dtype=np.uint64)
            rows["iteration_id"] = source.iteration_id
            rows["primitive_kind"] = int(PrimitiveKind.CONSUMER)
            rows["query_id"] = source.query_base + np.arange(start, end, dtype=np.int64)
            rows["state_version"] = source.state_version
            rows["consumer_id"] = rows["query_id"]
            rows["reduction_key"] = rows["query_id"]
            rows["resource_class"] = int(ResourceClass.QUERY)
            rows["template_id"] = source.template_id
            rows["flags"] = source.loss_flags
            dependency_lists = [
                _consumer_offsets(source, query) for query in range(start, end)
            ]
            dependencies = np.concatenate([
                reduction_start + np.asarray(offsets, dtype=dependency_dtype())
                for offsets in dependency_lists
            ] or [np.empty(0, dtype=dependency_dtype())])
            local_counts = np.asarray(
                [len(offsets) for offsets in dependency_lists], dtype=np.uint64
            )
            local_prefix = np.empty((end - start) + 1, dtype=np.uint64)
            local_prefix[0] = 0
            np.cumsum(local_counts, out=local_prefix[1:])
            rows["dependency_begin"] = np.asarray(
                local_prefix[:-1], dtype=np.uint64,
            )
            rows["dependency_count"] = np.asarray(
                local_counts, dtype=np.uint32
            )
            yield self._make_event_packet(rows, dependencies)

    def _adjoint_gradient(
        self, source: VirtualTracePacket, adjoint_start: int, gradient_start: int,
        consumer_start: int, relation_base: int,
    ) -> Iterator[VirtualEventPacket]:
        emitted = 0
        for columns in source.iter_relation_arrays(self.max_events):
            _candidates, query_ids, gaussian_ids, _keys = columns
            count = int(query_ids.size)
            local_queries = query_ids - source.query_base
            rows = np.empty(count * 2, dtype=event_dtype())
            rows[:] = TraceEvent().as_tuple()
            rows["event_id"] = np.arange(
                adjoint_start + emitted * 2,
                adjoint_start + (emitted + count) * 2,
                dtype=np.uint64,
            )
            rows["iteration_id"] = source.iteration_id
            rows["query_id"] = np.repeat(query_ids, 2)
            rows["gaussian_id"] = np.repeat(gaussian_ids, 2)
            rows["state_version"] = source.state_version
            rows["relation_id"] = np.repeat(
                np.arange(relation_base + emitted, relation_base + emitted + count), 2
            )
            rows["address_token"] = np.repeat(
                gaussian_ids * self.state_record_bytes, 2
            )
            rows["template_id"] = source.template_id
            rows["field_mask"] = source.field_mask
            rows["primitive_kind"][0::2] = int(PrimitiveKind.ADJOINT)
            rows["primitive_kind"][1::2] = int(PrimitiveKind.GRADIENT_REDUCTION)
            rows["resource_class"][0::2] = int(ResourceClass.ISSUE)
            rows["resource_class"][1::2] = int(ResourceClass.QUERY)
            rows["reduction_key"][1::2] = gaussian_ids
            dependencies = np.empty(count * 2, dtype=dependency_dtype())
            dependencies[0::2] = consumer_start + local_queries
            dependencies[1::2] = np.arange(
                adjoint_start + emitted * 2,
                adjoint_start + (emitted + count) * 2,
                2,
                dtype=dependency_dtype(),
            )
            rows["dependency_begin"] = np.arange(0, count * 2, dtype=np.uint64)
            rows["dependency_count"] = 1
            emitted += count
            yield self._make_event_packet(rows, dependencies)

    def _make_event_packet(
        self, rows: np.ndarray, dependencies: np.ndarray
    ) -> VirtualEventPacket:
        rows = np.asarray(rows, dtype=event_dtype())
        if dependencies.size and int(rows["dependency_count"].sum()) == 0:
            rows["dependency_begin"] = np.arange(
                dependencies.size, dtype=np.uint64
            )
            rows["dependency_count"] = 1
        packet = VirtualEventPacket(
            packet_id=self.next_packet_id,
            global_event_start=int(rows["event_id"][0]) if rows.size else self.next_event_id,
            events=rows,
            dependencies=np.asarray(dependencies, dtype=dependency_dtype()),
        )
        self.next_packet_id += 1
        return packet


def _consumer_offsets(source: VirtualTracePacket, query: int) -> np.ndarray:
    """Return local reduction offsets for one exact loss consumer."""

    if source.loss_flags & LOSS_SSIM:
        if len(source.query_shape) != 2:
            raise ValueError("SSIM consumer requires a raster query shape")
        height, width = source.query_shape
        y, x = divmod(query, width)
        radius = source.ssim_radius
        ys = np.arange(max(0, y - radius), min(height, y + radius + 1))
        xs = np.arange(max(0, x - radius), min(width, x + radius + 1))
        return (ys[:, None] * width + xs[None, :]).reshape(-1).astype(np.int64)
    if source.loss_flags & LOSS_TV:
        if len(source.query_shape) != 3:
            raise ValueError("TV consumer requires a voxel query shape")
        voxel_x, voxel_y, voxel_z = source.query_shape
        x, remainder = divmod(query, voxel_y * voxel_z)
        y, z = divmod(remainder, voxel_z)
        offsets = [query]
        if x > 0:
            offsets.append(query - voxel_y * voxel_z)
        if x + 1 < voxel_x:
            offsets.append(query + voxel_y * voxel_z)
        if y > 0:
            offsets.append(query - voxel_z)
        if y + 1 < voxel_y:
            offsets.append(query + voxel_z)
        if z > 0:
            offsets.append(query - 1)
        if z + 1 < voxel_z:
            offsets.append(query + 1)
        return np.asarray(offsets, dtype=np.int64)
    return np.asarray([query], dtype=np.int64)


@dataclass
class VirtualEventStreamValidator:
    """Validate global-ID continuity while event packets cross a boundary."""

    next_packet_id: int = 0
    next_event_id: int = 0
    accepted_packets: int = 0
    accepted_events: int = 0
    final_seen: bool = False

    def accept(self, packet: VirtualEventPacket) -> None:
        if self.final_seen:
            raise ValueError("virtual event packet arrived after the final packet")
        if packet.packet_id != self.next_packet_id:
            raise ValueError("virtual event packet IDs are not contiguous")
        if packet.global_event_start != self.next_event_id:
            raise ValueError("virtual event packet event IDs were rebased or skipped")
        if packet.external_dependencies.size and int(packet.external_dependencies.max()) >= packet.global_event_start:
            raise ValueError("virtual event packet external dependency is not from an earlier packet")
        self.next_packet_id += 1
        self.next_event_id = packet.global_event_end
        self.accepted_packets += 1
        self.accepted_events += packet.event_count
        if packet.final_packet:
            self.final_seen = True

    def finalize(self) -> None:
        if not self.final_seen:
            raise ValueError("virtual event stream ended without a final packet")


@dataclass(frozen=True)
class VirtualTraceProgress:
    """Progress emitted by a bounded producer/consumer run."""

    phase: str
    produced_packets: int
    consumed_packets: int
    produced_relations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class VirtualTraceRun:
    """Machine-readable summary of a completed packet stream."""

    produced_packets: int
    consumed_packets: int
    logical_relation_count: int
    logical_expanded_trace_bytes: int
    physical_stream_bytes: int
    peak_resident_packet_bytes: int
    elapsed_seconds: float


class VirtualTraceStream:
    """Run a bounded producer and consumer with a hard inactivity deadline.

    The producer may execute CUDA work while the consumer validates or feeds
    a cycle engine.  At most ``max_inflight_packets`` packet objects are held
    between them.  A producer exception, consumer exception or period with no
    packet progress is propagated to the caller and stops both sides.
    """

    _SENTINEL = object()

    def __init__(
        self,
        packets: Iterable[VirtualTracePacket] | Callable[[], Iterable[VirtualTracePacket]],
        *,
        max_inflight_packets: int = 2,
        inactivity_timeout_seconds: float = 300.0,
        progress: Callable[[VirtualTraceProgress], None] | None = None,
        progress_interval_seconds: float = 30.0,
    ) -> None:
        if max_inflight_packets <= 0:
            raise ValueError("virtual trace in-flight packet capacity must be positive")
        if inactivity_timeout_seconds <= 0:
            raise ValueError("virtual trace inactivity timeout must be positive")
        if progress_interval_seconds <= 0:
            raise ValueError("virtual trace progress interval must be positive")
        self._packets = packets
        self.max_inflight_packets = max_inflight_packets
        self.inactivity_timeout_seconds = inactivity_timeout_seconds
        self.progress = progress
        self.progress_interval_seconds = progress_interval_seconds

    def run(self, consume: Callable[[VirtualTracePacket], Any]) -> VirtualTraceRun:
        started = time.monotonic()
        packet_queue: queue.Queue[Any] = queue.Queue(self.max_inflight_packets)
        stop = threading.Event()
        producer_error: list[BaseException] = []
        produced_packets = 0
        produced_relations = 0
        consumed_packets = 0
        consumed_relations = 0
        physical_bytes = 0
        peak_packet_bytes = 0
        last_progress = started
        counters_lock = threading.Lock()

        def report(phase: str, *, force: bool = False) -> None:
            nonlocal last_progress
            if self.progress is None:
                return
            now = time.monotonic()
            if not force and now - last_progress < self.progress_interval_seconds:
                return
            with counters_lock:
                snapshot = VirtualTraceProgress(
                    phase=phase,
                    produced_packets=produced_packets,
                    consumed_packets=consumed_packets,
                    produced_relations=produced_relations,
                    elapsed_seconds=now - started,
                )
            self.progress(snapshot)
            last_progress = now

        def produce() -> None:
            nonlocal produced_packets, produced_relations, physical_bytes, peak_packet_bytes
            try:
                source = self._packets() if callable(self._packets) else self._packets
                for packet in source:
                    if stop.is_set():
                        break
                    while not stop.is_set():
                        try:
                            packet_queue.put(packet, timeout=min(
                                self.inactivity_timeout_seconds, 1.0
                            ))
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        break
                    with counters_lock:
                        produced_packets += 1
                        produced_relations += packet.logical_relation_count
                        physical_bytes += packet.physical_bytes
                        peak_packet_bytes = max(peak_packet_bytes, packet.physical_bytes)
                    report("produce")
                while not stop.is_set():
                    try:
                        packet_queue.put(self._SENTINEL, timeout=min(
                            self.inactivity_timeout_seconds, 1.0
                        ))
                        break
                    except queue.Full:
                        continue
            except BaseException as error:  # propagate on the consumer thread
                producer_error.append(error)
                stop.set()
                try:
                    packet_queue.put_nowait(self._SENTINEL)
                except queue.Full:
                    pass

        producer = threading.Thread(
            target=produce, name="gala-virtual-trace-producer", daemon=True
        )
        failure: BaseException | None = None
        producer.start()
        try:
            while True:
                try:
                    item = packet_queue.get(timeout=self.inactivity_timeout_seconds)
                except queue.Empty as error:
                    stop.set()
                    if producer_error:
                        raise RuntimeError("virtual trace producer failed") from producer_error[0]
                    raise TimeoutError(
                        "virtual trace producer/consumer made no progress for "
                        f"{self.inactivity_timeout_seconds:g} seconds"
                    ) from error
                if item is self._SENTINEL:
                    break
                packet = item
                consume(packet)
                consumed_packets += 1
                consumed_relations += packet.logical_relation_count
                peak_packet_bytes = max(peak_packet_bytes, packet.physical_bytes)
                report("consume")
        except BaseException as error:
            failure = error
            stop.set()
            raise
        finally:
            stop.set()
            # A user-supplied CUDA iterator cannot always be interrupted.
            # Keep this cleanup bounded so the inactivity gate itself never
            # turns into another long wait.
            producer.join(timeout=min(self.inactivity_timeout_seconds, 1.0))
            if producer.is_alive() and failure is None:
                raise TimeoutError("virtual trace producer did not stop after stream termination")
        if producer_error:
            raise RuntimeError("virtual trace producer failed") from producer_error[0]
        if produced_packets != consumed_packets or produced_relations != consumed_relations:
            raise RuntimeError("virtual trace packet production and consumption diverged")
        report("complete", force=True)
        elapsed = time.monotonic() - started
        # Every valid relation has six relation-specific event records and
        # eight dependency IDs in the current CLAMP schema.  Candidate,
        # query-level and update records are added by the later stream
        # expander, so this is explicitly a lower bound until that consumer
        # reports the exact expanded total.
        relation_event_records = 6
        relation_dependency_ids = 8
        logical_bytes = produced_relations * (
            relation_event_records * event_dtype().itemsize
            + relation_dependency_ids * dependency_dtype().itemsize
        )
        return VirtualTraceRun(
            produced_packets=produced_packets,
            consumed_packets=consumed_packets,
            logical_relation_count=produced_relations,
            logical_expanded_trace_bytes=logical_bytes,
            physical_stream_bytes=physical_bytes,
            peak_resident_packet_bytes=peak_packet_bytes,
            elapsed_seconds=elapsed,
        )


_POPCOUNT8 = np.asarray(
    [value.bit_count() for value in range(256)], dtype=np.uint8
)


@lru_cache(maxsize=None)
def _local_valid_words(template_id: int, *extents: int) -> tuple[int, ...]:
    """Return valid local-query bits for one possible edge-tile extent."""

    if template_id == RASTER_TEMPLATE_ID:
        height, width = extents
        local_count = int(np.prod(RASTER_BLOCK, dtype=np.int64))
        words = [0] * (local_count // MASK_WORD_BITS)
        for local in range(local_count):
            y, x = divmod(local, RASTER_BLOCK[1])
            if y < height and x < width:
                words[local // MASK_WORD_BITS] |= 1 << (local % MASK_WORD_BITS)
        return tuple(words)
    if template_id == VOXEL_TEMPLATE_ID:
        size_x, size_y, size_z = extents
        local_count = int(np.prod(VOXEL_BLOCK, dtype=np.int64))
        words = [0] * (local_count // MASK_WORD_BITS)
        for local in range(local_count):
            x, remainder = divmod(local, VOXEL_BLOCK[1] * VOXEL_BLOCK[2])
            y, z = divmod(remainder, VOXEL_BLOCK[2])
            if x < size_x and y < size_y and z < size_z:
                words[local // MASK_WORD_BITS] |= 1 << (local % MASK_WORD_BITS)
        return tuple(words)
    raise ValueError(f"unsupported virtual trace template: {template_id}")
