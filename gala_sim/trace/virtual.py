"""Bounded, lossless virtual trace packets for dense CUDA workloads.

The official rasterizer exposes a point list, a sorted point key and a
per-candidate validity mask.  Expanding every set bit into a structured event
array is useful for a small audit, but it is not a viable representation for a
30,000-iteration run.  This module keeps that exact information in a bounded
packet and exposes deterministic lazy relation iterators for consumers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
import queue
import threading
import time
from typing import Any

import numpy as np

from gala_sim.clamp.events import dependency_dtype, event_dtype


MASK_WORD_BITS = 32
"""Number of query bits represented by one mask word."""

RASTER_TEMPLATE_ID = 1
VOXEL_TEMPLATE_ID = 2
RASTER_BLOCK = (16, 16)
VOXEL_BLOCK = (8, 8, 8)


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

    def __post_init__(self) -> None:
        if self.iteration_id < 0 or self.template_id < 0 or self.query_base < 0:
            raise ValueError("virtual trace packet identifiers must be non-negative")
        if not self.query_shape or any(int(value) <= 0 for value in self.query_shape):
            raise ValueError("virtual trace packet query shape must be positive")
        if self.state_version < 0 or self.field_mask < 0 or self.loss_flags < 0:
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

        tiles = np.right_shift(np.asarray(self.point_keys, dtype=np.uint64), 32)
        for query_offset in range(self.query_count):
            tile, local_query = self._tile_and_local_query(query_offset)
            word_index, bit_index = divmod(local_query, MASK_WORD_BITS)
            bit = np.uint32(1 << bit_index)
            candidates = np.flatnonzero(
                (tiles == np.uint64(tile)) & ((self.masks[:, word_index] & bit) != 0)
            )
            for candidate in candidates:
                index = int(candidate)
                yield (
                    index,
                    self.query_base + query_offset,
                    int(self.point_ids[index]),
                    int(self.point_keys[index]),
                )

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
        batch: list[tuple[int, int, int, int]] = []
        for relation in self.iter_relations():
            batch.append(relation)
            if len(batch) == max_relations:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch)

    def materialize_relations(self, *, max_relations: int | None = None) -> np.ndarray:
        """Materialize only when explicitly bounded by the caller."""

        relation_count = self.logical_relation_count
        if max_relations is not None and relation_count > max_relations:
            raise ValueError("virtual trace relation materialization exceeds its bound")
        rows = np.fromiter(
            (value for relation in self.iter_relations() for value in relation),
            dtype=np.int64,
            count=relation_count * 4,
        )
        return rows.reshape((-1, 4))


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
