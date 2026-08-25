"""Runtime-only hooks for capturing the official R²-Gaussian CUDA workload."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gala_sim.clamp import (
    ChunkedTraceBuilder,
    ModificationKind,
    PrimitiveKind,
    ResourceClass,
    TraceEvent,
    UpdateBeginKind,
)
from gala_sim.trace import Trace, TraceWriter

from .buffer_decoder import load_buffer_decoder


RASTER_TEMPLATE_ID = 1
VOXEL_TEMPLATE_ID = 2
UPDATE_TEMPLATE_ID = 3
MODIFICATION_TEMPLATE_ID = 4
MOD_PRUNE = int(ModificationKind.PRUNE)
MOD_CLONE_PARENT = int(ModificationKind.CLONE_PARENT)
MOD_CLONE_CHILD = int(ModificationKind.CLONE_CHILD)
MOD_SPLIT_PARENT = int(ModificationKind.SPLIT_PARENT)
MOD_SPLIT_CHILD = int(ModificationKind.SPLIT_CHILD)
UPDATE_BEGIN_COLLECTION = int(UpdateBeginKind.COLLECTION)
UPDATE_BEGIN_OPTIMIZER = int(UpdateBeginKind.OPTIMIZER)
FIELD_POSITION = 1 << 0
FIELD_DENSITY = 1 << 1
FIELD_SCALE = 1 << 2
FIELD_ROTATION = 1 << 3
STATE_FIELD_MASK = FIELD_POSITION | FIELD_DENSITY | FIELD_SCALE | FIELD_ROTATION
LOSS_L1 = 1 << 0
LOSS_SSIM = 1 << 1
LOSS_TV = 1 << 2
RASTER_BLOCK = (16, 16)
VOXEL_BLOCK = (8, 8, 8)


@dataclass
class _QueryContext:
    query_base: int
    query_shape: tuple[int, ...]
    relation_query_offsets: np.ndarray
    gaussian_ids: np.ndarray
    relation_ids: np.ndarray
    reduction_events: np.ndarray
    buffer_pointer: int
    output_pointer: int
    template_id: int
    field_mask: int
    loss_flags: int = 0
    ssim_radius: int = 0


@dataclass
class _PendingQuery:
    records: Any | None
    rendered: int
    query_base: int
    query_shape: tuple[int, ...]
    binning_pointer: int
    output_pointer: int
    template_id: int
    field_mask: int
    loss_flags: int = 0
    ssim_radius: int = 0


@dataclass
class TraceSession:
    """Capture real extension buffers and Python call boundaries in one process."""

    output_root: Path
    state_record_bytes: int = 128
    relation_candidate_bytes: int = 0
    chunk_events: int = 65536
    _builder: ChunkedTraceBuilder = field(init=False)
    _decoder: Any = field(default=None, init=False)
    _iteration: int = field(default=0, init=False)
    _next_query: int = field(default=0, init=False)
    _next_relation: int = field(default=0, init=False)
    _next_gaussian: int = field(default=0, init=False)
    _state_version: int = field(default=0, init=False)
    _contexts: dict[int, _QueryContext] = field(default_factory=dict, init=False)
    _output_contexts: dict[int, _QueryContext] = field(default_factory=dict, init=False)
    _pending_queries: list[_PendingQuery] = field(default_factory=list, init=False)
    _pending_query_by_buffer: dict[int, _PendingQuery] = field(default_factory=dict, init=False)
    _pending_query_by_output: dict[int, _PendingQuery] = field(default_factory=dict, init=False)
    _pending_backwards: list[tuple[int, bool]] = field(default_factory=list, init=False)
    _pending_backward_buffers: set[int] = field(default_factory=set, init=False)
    _pending_gradients: dict[int, list[int]] = field(default_factory=dict, init=False)
    _pending_backward_events: list[int] = field(default_factory=list, init=False)
    _gaussian_ids: list[int] = field(default_factory=list, init=False)
    _gaussian_ids_initialized: bool = field(default=False, init=False)
    _initial_gaussian_count: int = field(default=0, init=False)
    _query_capture_allowed: bool = field(default=False, init=False)
    _audit: dict[str, int] = field(default_factory=dict, init=False)
    _installed: bool = field(default=False, init=False)
    _originals: list[tuple[Any, str, Any]] = field(default_factory=list, init=False)
    _modification_action: str | None = field(default=None, init=False)
    _collection_active: bool = field(default=False, init=False)
    _collection_begin_event: int | None = field(default=None, init=False)
    _collection_events: list[int] = field(default_factory=list, init=False)
    _prior_transition_events: list[int] = field(default_factory=list, init=False)
    _state_ready_event: int | None = field(default=None, init=False)
    _capture_error: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._builder = ChunkedTraceBuilder(
            self.chunk_events, chunk_root=self.output_root / ".capture_chunks"
        )

    def install(self) -> None:
        if self._installed:
            return
        import xray_gaussian_rasterization_voxelization as extension

        self._decoder = load_buffer_decoder()
        self._patch(
            extension.GaussianRasterizer, "forward",
            lambda original: self._wrap_query_forward(original, "raster"),
        )
        self._patch(
            extension.GaussianVoxelizer, "forward",
            lambda original: self._wrap_query_forward(original, "voxel"),
        )
        self._patch(extension._C, "rasterize_gaussians", self._wrap_rasterize)
        self._patch(extension._C, "voxelize_gaussians", self._wrap_voxelize)
        self._patch(extension._C, "rasterize_gaussians_backward", self._wrap_rasterize_backward)
        self._patch(extension._C, "voxelize_gaussians_backward", self._wrap_voxel_backward)

        from r2_gaussian.gaussian.gaussian_model import GaussianModel
        from r2_gaussian.utils import loss_utils

        self._patch(GaussianModel, "update_learning_rate", self._wrap_learning_rate)
        self._patch(GaussianModel, "training_setup", self._wrap_training_setup)
        self._patch(GaussianModel, "prune_points", self._wrap_prune_points)
        self._patch(GaussianModel, "densification_postfix", self._wrap_densification_postfix)
        self._patch(GaussianModel, "densify_and_clone", self._wrap_densify_clone)
        self._patch(GaussianModel, "densify_and_split", self._wrap_densify_split)
        self._patch(GaussianModel, "densify_and_prune", self._wrap_densify_and_prune)
        self._patch(loss_utils, "l1_loss", lambda original: self._wrap_loss(original, LOSS_L1))
        self._patch(loss_utils, "ssim", lambda original: self._wrap_loss(original, LOSS_SSIM))
        self._patch(loss_utils, "tv_3d_loss", lambda original: self._wrap_loss(original, LOSS_TV))
        self._installed = True

    def restore(self) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self._installed = False

    def finish(self) -> Trace:
        if self._capture_error is not None:
            raise RuntimeError(
                "trace capture cannot finish after a collection failure: "
                + self._capture_error
            )
        self._flush_pending_queries()
        audit = dict(self._audit)
        audit.setdefault("relation_record_device_batches", 0)
        audit.setdefault("relation_record_d2h_batches", 0)
        trace = self._builder.finish(metadata={
            "model": "R2-Gaussian",
            "dataset": "Chest",
            "capture_backend": "official_cuda_buffers_and_call_hooks",
            "state_record_bytes": self.state_record_bytes,
            "trace_chunk_events": self.chunk_events,
            "trace_capture_status": "real_extension_buffers",
            "capture_audit_schema_version": "gala-r2-capture-audit-v4",
            "initial_gaussian_count": self._initial_gaussian_count,
            "capture_audit": dict(sorted(audit.items())),
        })
        TraceWriter().write(trace, self.output_root)
        return trace

    def _patch(self, owner: Any, name: str, wrapper_factory: Callable[..., Any]) -> None:
        original = getattr(owner, name)
        self._originals.append((owner, name, original))
        setattr(owner, name, wrapper_factory(original))

    def _wrap_rasterize(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if self._capture_enabled():
                self._capture_raster(args, result)
            return result
        return wrapped

    def _wrap_voxelize(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if self._capture_enabled():
                self._capture_voxel(args, result)
            return result
        return wrapped

    def _wrap_rasterize_backward(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._capture_backward(int(args[14].data_ptr()), voxel=False)
            return result
        return wrapped

    def _wrap_voxel_backward(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._capture_backward(int(args[11].data_ptr()), voxel=True)
            return result
        return wrapped

    def _wrap_query_forward(self, original: Any, query_kind: str) -> Any:
        def wrapped(rasterizer: Any, *args: Any, **kwargs: Any) -> Any:
            previous = self._query_capture_allowed
            import torch

            self._query_capture_allowed = bool(torch.is_grad_enabled())
            self._audit_increment(f"official_{query_kind}_kernel_calls")
            if self._query_capture_allowed:
                self._audit_increment(f"captured_{query_kind}_kernel_calls")
                self._audit_increment("captured_query_kernel_calls")
            else:
                self._audit_increment(f"excluded_no_grad_{query_kind}_kernel_calls")
            try:
                return original(rasterizer, *args, **kwargs)
            finally:
                self._query_capture_allowed = previous
        return wrapped

    def _wrap_loss(self, original: Any, loss_flag: int) -> Any:
        def wrapped(output: Any, *args: Any, **kwargs: Any) -> Any:
            context = self._output_contexts.get(int(output.data_ptr()))
            pending = self._pending_query_by_output.get(int(output.data_ptr()))
            if context is not None:
                context.loss_flags |= loss_flag
                if loss_flag == LOSS_SSIM:
                    window_size = int(kwargs.get(
                        "window_size", args[1] if len(args) > 1 else 11
                    ))
                    context.ssim_radius = max(context.ssim_radius, window_size // 2)
            elif pending is not None:
                pending.loss_flags |= loss_flag
                if loss_flag == LOSS_SSIM:
                    window_size = int(kwargs.get(
                        "window_size", args[1] if len(args) > 1 else 11
                    ))
                    pending.ssim_radius = max(pending.ssim_radius, window_size // 2)
            return original(output, *args, **kwargs)
        return wrapped

    def _capture_enabled(self) -> bool:
        """Return the grad-mode decision captured at the official query boundary."""
        return self._query_capture_allowed

    def _wrap_learning_rate(self, original: Any) -> Any:
        def wrapped(model: Any, iteration: int, *args: Any, **kwargs: Any) -> Any:
            self._flush_pending_queries()
            self._iteration = int(iteration)
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            return original(model, iteration, *args, **kwargs)
        return wrapped

    def _wrap_training_setup(self, original: Any) -> Any:
        def wrapped(model: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(model, *args, **kwargs)
            self._initialize_or_validate_gaussians(int(model.get_xyz.shape[0]))
            optimizer = model.optimizer
            original_step = optimizer.step

            def step(*step_args: Any, **step_kwargs: Any) -> Any:
                self._flush_pending_queries()
                field_mask = self._optimizer_field_mask(optimizer)
                result_step = original_step(*step_args, **step_kwargs)
                self._capture_update(model, field_mask=field_mask)
                return result_step

            optimizer.step = step
            return result
        return wrapped

    def _wrap_prune_points(self, original: Any) -> Any:
        def wrapped(model: Any, mask: Any, *args: Any, **kwargs: Any) -> Any:
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            removed = mask.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
            if removed.size != len(self._gaussian_ids):
                raise RuntimeError("prune mask does not match stable Gaussian ID count")
            removed_ids = [self._gaussian_ids[index] for index, flag in enumerate(removed)
                           if flag and index < len(self._gaussian_ids)]
            result = original(model, mask, *args, **kwargs)
            self._gaussian_ids = [value for index, value in enumerate(self._gaussian_ids)
                                  if index < len(removed) and not removed[index]]
            if len(self._gaussian_ids) != int(model.get_xyz.shape[0]):
                raise RuntimeError("prune result does not match stable Gaussian ID count")
            if self._modification_action != "split":
                for gaussian_id in removed_ids:
                    self._emit_set_modification(
                        gaussian_id, flags=MOD_PRUNE, reduction_key=gaussian_id
                    )
            return result
        return wrapped

    def _wrap_densification_postfix(self, original: Any) -> Any:
        def wrapped(model: Any, new_xyz: Any, *args: Any, **kwargs: Any) -> Any:
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            result = original(model, new_xyz, *args, **kwargs)
            count = int(new_xyz.shape[0])
            new_ids = list(range(self._next_gaussian, self._next_gaussian + count))
            self._next_gaussian += count
            self._gaussian_ids.extend(new_ids)
            if len(self._gaussian_ids) != int(model.get_xyz.shape[0]):
                raise RuntimeError("densification result does not match stable Gaussian ID count")
            return result
        return wrapped

    def _wrap_densify_clone(self, original: Any) -> Any:
        def wrapped(model: Any, grads: Any, grad_threshold: float,
                    densify_scale_threshold: float, *args: Any, **kwargs: Any) -> Any:
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            selected = self._clone_selection(model, grads, grad_threshold, densify_scale_threshold)
            parents = tuple(self._gaussian_ids[index] for index in selected)
            next_gaussian = self._next_gaussian
            self._modification_action = "clone"
            try:
                result = original(model, grads, grad_threshold, densify_scale_threshold,
                                  *args, **kwargs)
            finally:
                self._modification_action = None
            new_ids = list(range(next_gaussian, self._next_gaussian))
            if len(new_ids) != len(parents):
                raise RuntimeError("clone event count does not match selected Gaussian count")
            for parent, child in zip(parents, new_ids):
                parent_event = self._emit_set_modification(
                    parent, flags=MOD_CLONE_PARENT, reduction_key=parent
                )
                self._emit_set_modification(
                    child, flags=MOD_CLONE_CHILD, reduction_key=parent,
                    dependencies=[parent_event],
                )
            return result
        return wrapped

    def _wrap_densify_split(self, original: Any) -> Any:
        def wrapped(model: Any, grads: Any, grad_threshold: float,
                    densify_scale_threshold: float, N: int = 2,
                    *args: Any, **kwargs: Any) -> Any:
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            selected = self._split_selection(model, grads, grad_threshold, densify_scale_threshold)
            parents = tuple(self._gaussian_ids[index] for index in selected)
            next_gaussian = self._next_gaussian
            self._modification_action = "split"
            try:
                result = original(model, grads, grad_threshold, densify_scale_threshold, N,
                                  *args, **kwargs)
            finally:
                self._modification_action = None
            new_ids = list(range(next_gaussian, self._next_gaussian))
            expected = len(parents) * int(N)
            if len(new_ids) != expected:
                raise RuntimeError("split event count does not match selected Gaussian count")
            children_by_parent: dict[int, list[int]] = {parent: [] for parent in parents}
            for child, parent in zip(new_ids, self._split_child_lineage(parents, int(N))):
                event = self._emit_set_modification(
                    child, flags=MOD_SPLIT_CHILD, reduction_key=parent
                )
                children_by_parent[parent].append(event)
            for parent in parents:
                self._emit_set_modification(
                    parent, flags=MOD_SPLIT_PARENT, reduction_key=parent,
                    dependencies=children_by_parent[parent],
                )
            return result
        return wrapped

    @staticmethod
    def _split_child_lineage(parents: tuple[int, ...], children_per_parent: int) -> tuple[int, ...]:
        if children_per_parent <= 0:
            raise ValueError("split children per parent must be positive")
        return parents * children_per_parent

    def _wrap_densify_and_prune(self, original: Any) -> Any:
        def wrapped(model: Any, *args: Any, **kwargs: Any) -> Any:
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            self._flush_pending_queries()
            self._start_collection_transaction()
            try:
                result = original(model, *args, **kwargs)
            except BaseException:
                self._capture_error = "densify_and_prune raised before its collection transaction closed"
                self._collection_active = False
                self._collection_begin_event = None
                self._collection_events = []
                raise
            self._finish_collection_transaction()
            return result
        return wrapped

    @staticmethod
    def _clone_selection(model: Any, grads: Any, grad_threshold: float,
                         densify_scale_threshold: float) -> np.ndarray:
        import torch
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = torch.logical_and(
            selected, torch.max(model.get_scaling, dim=1).values <= densify_scale_threshold
        )
        return selected.detach().cpu().numpy().astype(bool, copy=False).reshape(-1).nonzero()[0]

    @staticmethod
    def _split_selection(model: Any, grads: Any, grad_threshold: float,
                          densify_scale_threshold: float) -> np.ndarray:
        import torch
        padded = torch.zeros((model.get_xyz.shape[0],), device=grads.device)
        padded[:grads.shape[0]] = grads.squeeze()
        selected = padded >= grad_threshold
        selected = torch.logical_and(
            selected, torch.max(model.get_scaling, dim=1).values > densify_scale_threshold
        )
        return selected.detach().cpu().numpy().astype(bool, copy=False).reshape(-1).nonzero()[0]

    def _capture_raster(self, args: tuple[Any, ...], result: tuple[Any, ...]) -> None:
        rendered, output, _, geometry, binning, _ = result
        self._audit_increment("cuda_relation_candidates", int(rendered))
        means = args[0]
        height, width = int(args[10]), int(args[11])
        self._capture_query(
            means, binning, output, int(rendered), (height, width),
            lambda: self._decoder.raster_trace_records(
                geometry, binning, int(means.shape[0]), int(rendered), height, width
            ),
            template_id=RASTER_TEMPLATE_ID,
            field_mask=STATE_FIELD_MASK,
        )

    def _capture_voxel(self, args: tuple[Any, ...], result: tuple[Any, ...]) -> None:
        rendered, output, _, _, _, geometry, binning, _ = result
        self._audit_increment("cuda_relation_candidates", int(rendered))
        means = args[0]
        dimensions = tuple(int(value) for value in args[6:9])
        self._capture_query(
            means, binning, output, int(rendered), dimensions,
            lambda: self._decoder.voxel_trace_records(
                geometry, binning, int(means.shape[0]), int(rendered), *dimensions
            ),
            template_id=VOXEL_TEMPLATE_ID,
            field_mask=STATE_FIELD_MASK,
        )

    def _capture_query(
        self, means: Any, binning: Any, output: Any, rendered: int,
        query_shape: tuple[int, ...], records_fn: Callable[[], Any],
        *, template_id: int, field_mask: int,
    ) -> None:
        if rendered < 0 or not query_shape or any(value <= 0 for value in query_shape):
            raise ValueError("decoded trace dimensions and candidate count must be positive")
        query_base = self._next_query
        image_elements = int(np.prod(query_shape))
        self._next_query += image_elements
        self._audit_increment("captured_logical_queries", image_elements)
        gaussian_count = int(means.shape[0])
        self._ensure_gaussians(gaussian_count)
        binning_pointer = int(binning.data_ptr())
        output_pointer = int(output.data_ptr())
        if binning_pointer in self._pending_query_by_buffer or binning_pointer in self._contexts:
            raise RuntimeError("captured query reused a live binning buffer pointer")
        if output_pointer in self._pending_query_by_output or output_pointer in self._output_contexts:
            raise RuntimeError("captured query reused a live output buffer pointer")
        pending = _PendingQuery(
            records_fn() if rendered > 0 else None,
            rendered, query_base, query_shape, binning_pointer, output_pointer,
            template_id, field_mask,
        )
        if pending.records is not None:
            self._audit_increment("relation_record_device_batches")
        self._pending_queries.append(pending)
        self._pending_query_by_buffer[binning_pointer] = pending
        self._pending_query_by_output[output_pointer] = pending

    def _flush_pending_queries(self) -> None:
        if not self._pending_queries and not self._pending_backwards:
            return
        pending_buffers = {item.binning_pointer for item in self._pending_queries}
        if (
            len(pending_buffers) != len(self._pending_queries)
            or pending_buffers != set(self._pending_query_by_buffer)
            or pending_buffers != self._pending_backward_buffers
            or len(self._pending_backwards) != len(self._pending_queries)
        ):
            raise RuntimeError("pending query batch does not have exactly one backward per query")
        for item in self._pending_queries:
            if item.loss_flags == 0:
                raise RuntimeError(
                    "captured query reached flush without a captured loss consumer"
                )
        for buffer_pointer, voxel in self._pending_backwards:
            item = self._pending_query_by_buffer.get(buffer_pointer)
            if item is None:
                raise RuntimeError("pending backward does not match a captured query")
            expected_template = VOXEL_TEMPLATE_ID if voxel else RASTER_TEMPLATE_ID
            if item.template_id != expected_template:
                raise RuntimeError("captured backward kind does not match its forward context")

        records_by_query: list[np.ndarray] = []
        device_records = [item.records for item in self._pending_queries if item.records is not None]
        if device_records:
            host_records = self._copy_record_batches(device_records)
            self._audit_increment("relation_record_d2h_batches")
            cursor = 0
            for item in self._pending_queries:
                if item.records is None:
                    records_by_query.append(np.empty((0, 4), dtype=np.int64))
                else:
                    record_count = int(item.records.shape[0])
                    records_by_query.append(np.asarray(host_records[cursor:cursor + record_count], dtype=np.int64))
                    cursor += record_count
        else:
            records_by_query = [np.empty((0, 4), dtype=np.int64) for _ in self._pending_queries]
        pending = tuple(self._pending_queries)
        self._pending_queries.clear()
        self._pending_query_by_buffer.clear()
        self._pending_query_by_output.clear()
        self._pending_backward_buffers.clear()
        for item, records in zip(pending, records_by_query):
            self._emit_query_records(
                records, rendered=item.rendered, query_base=item.query_base,
                query_shape=item.query_shape, binning_pointer=item.binning_pointer,
                output_pointer=item.output_pointer, template_id=item.template_id,
                field_mask=item.field_mask,
            )
            context = self._contexts[item.binning_pointer]
            context.loss_flags = item.loss_flags
            context.ssim_radius = item.ssim_radius
        pending_backwards = tuple(self._pending_backwards)
        self._pending_backwards.clear()
        for buffer_pointer, voxel in pending_backwards:
            self._emit_backward(buffer_pointer, voxel=voxel)

    @staticmethod
    def _copy_record_batches(record_batches: list[Any]) -> np.ndarray:
        if not record_batches:
            return np.empty((0, 4), dtype=np.int64)
        if all(isinstance(batch, np.ndarray) for batch in record_batches):
            if len(record_batches) == 1:
                return np.asarray(record_batches[0])
            return np.concatenate(record_batches, axis=0)
        if any(isinstance(batch, np.ndarray) for batch in record_batches):
            raise TypeError("relation record batches must all use the same tensor backend")
        combined = record_batches[0]
        if len(record_batches) > 1:
            import torch

            combined = torch.cat(record_batches, dim=0)
        return combined.detach().cpu().numpy()

    def _emit_query_records(
        self, records: np.ndarray, *, rendered: int, query_base: int,
        query_shape: tuple[int, ...], binning_pointer: int, output_pointer: int,
        template_id: int, field_mask: int,
    ) -> None:
        if records.ndim != 2 or records.shape[1] != 4:
            raise ValueError("decoded trace records must have shape [N, 4]")
        candidate_records = records[:rendered]
        relation_records = records[rendered:]
        if (
            candidate_records.shape[0] != rendered
            or (rendered and not np.array_equal(candidate_records[:, 0], np.zeros(rendered)))
            or (rendered and not np.array_equal(candidate_records[:, 1], np.arange(rendered)))
            or (relation_records.size and np.any(relation_records[:, 0] != 1))
        ):
            raise ValueError("decoded trace record kinds or candidate indexes are invalid")
        gaussian_indexes = candidate_records[:, 2]
        if gaussian_indexes.size and (
            int(gaussian_indexes.min()) < 0
            or int(gaussian_indexes.max()) >= len(self._gaussian_ids)
        ):
            raise ValueError("decoded trace record contains an invalid Gaussian index")
        candidate_keys = candidate_records[:, 3].view(np.uint64)
        relation_candidates = relation_records[:, 1]
        if relation_candidates.size and (
            int(relation_candidates.min()) < 0
            or int(relation_candidates.max()) >= rendered
        ):
            raise ValueError("decoded relation refers to an invalid candidate")
        relation_offsets = self._decode_query_offsets(
            candidate_keys[relation_candidates], relation_records[:, 2],
            query_shape, template_id,
        )
        order = np.argsort(relation_offsets, kind="stable")
        relation_offsets = relation_offsets[order]
        relation_candidates = relation_candidates[order]
        valid_candidates = np.zeros(rendered, dtype=bool)
        valid_candidates[relation_candidates] = True
        candidate_event_ids = np.empty(rendered, dtype=np.int64)
        for index, (_, _, raw_gaussian, _) in enumerate(candidate_records):
            gaussian_id = self._gaussian_ids[int(raw_gaussian)]
            candidate_event = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE),
                gaussian_id=gaussian_id,
                state_version=self._state_version,
                resource_class=int(ResourceClass.RELATION),
                address_token=int(candidate_keys[index]),
                data_bytes=self.relation_candidate_bytes,
                template_id=template_id, field_mask=field_mask,
                flags=int(valid_candidates[index]),
            ), dependencies=(
                () if self._state_ready_event is None else (self._state_ready_event,)
            ))
            candidate_event_ids[index] = candidate_event

        relation_count = int(relation_candidates.size)
        stable_gaussian_ids = np.asarray(self._gaussian_ids, dtype=np.int64)[
            gaussian_indexes[relation_candidates]
        ]
        relation_ids = np.arange(
            self._next_relation, self._next_relation + relation_count, dtype=np.int64
        )
        self._next_relation += relation_count
        relation_event_ids = np.empty(relation_count, dtype=np.int64)
        for index in range(relation_count):
            gaussian_id = int(stable_gaussian_ids[index])
            relation_id = int(relation_ids[index])
            relation = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.RELATION),
                query_id=query_base + int(relation_offsets[index]),
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_id, resource_class=int(ResourceClass.RELATION),
                address_token=gaussian_id * self.state_record_bytes,
                template_id=template_id, field_mask=field_mask,
            ), dependencies=[candidate_event_ids[int(relation_candidates[index])]])
            relation_event_ids[index] = relation
        self._audit_increment("cuda_valid_relations", relation_count)

        query_count = int(np.prod(query_shape))
        counts = np.bincount(relation_offsets, minlength=query_count)
        begins = np.empty(query_count + 1, dtype=np.int64)
        begins[0] = 0
        np.cumsum(counts, out=begins[1:])
        query_close_events = np.empty(query_count, dtype=np.int64)
        for query_offset in range(query_count):
            begin, end = int(begins[query_offset]), int(begins[query_offset + 1])
            close_dependencies: list[int] = [
                int(event) for event in relation_event_ids[begin:end]
            ]
            if self._state_ready_event is not None:
                close_dependencies.append(self._state_ready_event)
            query_close_events[query_offset] = self._emit_query_close(
                query_base + query_offset, close_dependencies,
                template_id=template_id, field_mask=field_mask,
            )

        forward_event_ids = np.empty(relation_count, dtype=np.int64)
        for index in range(relation_count):
            query_id = query_base + int(relation_offsets[index])
            gaussian_id = int(stable_gaussian_ids[index])
            relation_id = int(relation_ids[index])
            relation = int(relation_event_ids[index])
            request = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_id,
                resource_class=int(ResourceClass.CACHE),
                address_token=gaussian_id * self.state_record_bytes,
                data_bytes=self.state_record_bytes,
                template_id=template_id, field_mask=field_mask,
            ), dependencies=[relation])
            returned = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_id,
                resource_class=int(ResourceClass.CACHE),
                address_token=gaussian_id * self.state_record_bytes,
                data_bytes=self.state_record_bytes,
                template_id=template_id, field_mask=field_mask,
            ), dependencies=[request])
            forward = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.FORWARD), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_id, resource_class=int(ResourceClass.ISSUE),
                address_token=gaussian_id * self.state_record_bytes,
                template_id=template_id, field_mask=field_mask,
            ), dependencies=[relation, returned])
            forward_event_ids[index] = forward

        reduction_events = np.empty(query_count, dtype=np.int64)
        for query_offset in range(query_count):
            query_id = query_base + query_offset
            begin, end = int(begins[query_offset]), int(begins[query_offset + 1])
            reduction_events[query_offset] = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.QUERY_REDUCTION), query_id=query_id,
                state_version=self._state_version, reduction_key=query_id,
                resource_class=int(ResourceClass.QUERY),
                template_id=template_id,
            ), dependencies=(
                query_close_events[query_offset],
                *(int(event) for event in forward_event_ids[begin:end]),
            ))
        context = _QueryContext(
            query_base, query_shape, relation_offsets, stable_gaussian_ids,
            relation_ids, reduction_events, binning_pointer, output_pointer,
            template_id, field_mask,
        )
        self._contexts[binning_pointer] = context
        self._output_contexts[output_pointer] = context

    def _emit_query_close(
        self, query_id: int, dependencies: Any,
        *, template_id: int, field_mask: int,
    ) -> int:
        return self._builder.emit(TraceEvent(
            iteration_id=self._iteration,
            primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=query_id,
            state_version=self._state_version, resource_class=int(ResourceClass.RELATION),
            template_id=template_id, field_mask=field_mask,
        ), dependencies=dependencies)

    def _capture_backward(self, buffer_pointer: int, *, voxel: bool) -> None:
        pending = self._pending_query_by_buffer.get(buffer_pointer)
        if pending is not None:
            expected_template = VOXEL_TEMPLATE_ID if voxel else RASTER_TEMPLATE_ID
            if pending.template_id != expected_template:
                raise RuntimeError("captured backward kind does not match its forward context")
            if pending.loss_flags == 0:
                raise RuntimeError(
                    "captured query reached backward without a captured loss consumer"
                )
            if buffer_pointer in self._pending_backward_buffers:
                raise RuntimeError("captured query received duplicate backward calls")
            self._pending_backward_buffers.add(buffer_pointer)
            self._pending_backwards.append((buffer_pointer, voxel))
            return
        self._emit_backward(buffer_pointer, voxel=voxel)

    def _emit_backward(self, buffer_pointer: int, *, voxel: bool) -> None:
        context = self._contexts.get(buffer_pointer)
        if context is None:
            return
        expected_template = VOXEL_TEMPLATE_ID if voxel else RASTER_TEMPLATE_ID
        if context.template_id != expected_template:
            raise RuntimeError("captured backward kind does not match its forward context")
        if context.loss_flags == 0:
            raise RuntimeError("captured query reached backward without a captured loss consumer")
        self._audit_increment("captured_backward_calls")
        relation_count = int(context.relation_ids.size)
        query_count = int(np.prod(context.query_shape))
        self._audit_increment("captured_backward_relations", relation_count)
        self._audit_increment("captured_consumers", query_count)
        consumer_events = np.empty(query_count, dtype=np.int64)
        for query_offset in range(query_count):
            query_id = context.query_base + query_offset
            dependencies = context.reduction_events[
                self._consumer_query_offsets(context, query_offset)
            ]
            consumer_events[query_offset] = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.CONSUMER), query_id=query_id,
                state_version=self._state_version, consumer_id=query_id,
                reduction_key=query_id, resource_class=int(ResourceClass.QUERY),
                template_id=context.template_id, flags=context.loss_flags,
            ), dependencies=dependencies)
        gradient_events = np.empty(relation_count, dtype=np.int64)
        for index in range(relation_count):
            query_offset = int(context.relation_query_offsets[index])
            query_id = context.query_base + query_offset
            gaussian_id = int(context.gaussian_ids[index])
            relation_id = int(context.relation_ids[index])
            adjoint = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.ADJOINT), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_id,
                resource_class=int(ResourceClass.ISSUE),
                address_token=gaussian_id * self.state_record_bytes,
                template_id=context.template_id, field_mask=context.field_mask,
            ), dependencies=[consumer_events[query_offset]])
            gradient = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.GRADIENT_REDUCTION), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_id,
                reduction_key=gaussian_id, resource_class=int(ResourceClass.QUERY),
                address_token=gaussian_id * self.state_record_bytes,
                template_id=context.template_id, field_mask=context.field_mask,
            ), dependencies=[adjoint])
            self._pending_gradients.setdefault(gaussian_id, []).append(gradient)
            gradient_events[index] = gradient
        relation_counts = np.bincount(
            context.relation_query_offsets, minlength=query_count
        )
        self._pending_backward_events.extend(
            int(event) for event in consumer_events[relation_counts == 0]
        )
        self._pending_backward_events.extend(int(event) for event in gradient_events)
        self._output_contexts.pop(context.output_pointer, None)
        self._contexts.pop(buffer_pointer, None)

    @staticmethod
    def _decode_query_offsets(
        candidate_keys: np.ndarray, local_queries: np.ndarray,
        query_shape: tuple[int, ...], template_id: int,
    ) -> np.ndarray:
        tiles = np.right_shift(candidate_keys, np.uint64(32)).astype(np.int64)
        local_queries = np.asarray(local_queries, dtype=np.int64)
        if template_id == RASTER_TEMPLATE_ID:
            height, width = query_shape
            blocks_x = (width + RASTER_BLOCK[1] - 1) // RASTER_BLOCK[1]
            x = (tiles % blocks_x) * RASTER_BLOCK[1] + local_queries % RASTER_BLOCK[1]
            y = (tiles // blocks_x) * RASTER_BLOCK[0] + local_queries // RASTER_BLOCK[1]
            offsets = y * width + x
        elif template_id == VOXEL_TEMPLATE_ID:
            voxel_x, voxel_y, voxel_z = query_shape
            blocks_x = (voxel_x + VOXEL_BLOCK[0] - 1) // VOXEL_BLOCK[0]
            blocks_y = (voxel_y + VOXEL_BLOCK[1] - 1) // VOXEL_BLOCK[1]
            tile_x = tiles % blocks_x
            tile_y = (tiles // blocks_x) % blocks_y
            tile_z = tiles // (blocks_x * blocks_y)
            local_x = local_queries % VOXEL_BLOCK[0]
            local_y = (local_queries // VOXEL_BLOCK[0]) % VOXEL_BLOCK[1]
            local_z = local_queries // (VOXEL_BLOCK[0] * VOXEL_BLOCK[1])
            x = tile_x * VOXEL_BLOCK[0] + local_x
            y = tile_y * VOXEL_BLOCK[1] + local_y
            z = tile_z * VOXEL_BLOCK[2] + local_z
            offsets = x * voxel_y * voxel_z + y * voxel_z + z
        else:
            raise ValueError(f"unsupported query template: {template_id}")
        query_count = int(np.prod(query_shape))
        if offsets.size and (int(offsets.min()) < 0 or int(offsets.max()) >= query_count):
            raise ValueError("decoded relation refers to a query outside the output")
        return offsets.astype(np.int64, copy=False)

    @staticmethod
    def _consumer_query_offsets(context: _QueryContext, query_offset: int) -> np.ndarray:
        if context.loss_flags & LOSS_SSIM:
            height, width = context.query_shape
            y, x = divmod(query_offset, width)
            radius = context.ssim_radius
            ys = np.arange(max(0, y - radius), min(height, y + radius + 1))
            xs = np.arange(max(0, x - radius), min(width, x + radius + 1))
            return (ys[:, None] * width + xs[None, :]).reshape(-1)
        if context.loss_flags & LOSS_TV:
            voxel_x, voxel_y, voxel_z = context.query_shape
            x, remainder = divmod(query_offset, voxel_y * voxel_z)
            y, z = divmod(remainder, voxel_z)
            offsets = [query_offset]
            if x > 0:
                offsets.append(query_offset - voxel_y * voxel_z)
            if x + 1 < voxel_x:
                offsets.append(query_offset + voxel_y * voxel_z)
            if y > 0:
                offsets.append(query_offset - voxel_z)
            if y + 1 < voxel_y:
                offsets.append(query_offset + voxel_z)
            if z > 0:
                offsets.append(query_offset - 1)
            if z + 1 < voxel_z:
                offsets.append(query_offset + 1)
            return np.asarray(offsets, dtype=np.int64)
        return np.asarray([query_offset], dtype=np.int64)

    def _capture_update(self, model: Any, *, field_mask: int) -> None:
        self._ensure_gaussians(int(model.get_xyz.shape[0]))
        self._audit_increment("optimizer_steps")
        begin = self._emit_update_begin(
            flags=UPDATE_BEGIN_OPTIMIZER,
            dependencies=(*self._pending_backward_events, *self._prior_transition_events),
        )
        if field_mask == 0:
            self._audit_increment("optimizer_noop_steps")
            self._state_ready_event = self._emit_update_end(
                flags=UPDATE_BEGIN_OPTIMIZER, begin=begin, dependencies=[begin],
                field_mask=0,
            )
            self._pending_gradients.clear()
            self._pending_backward_events.clear()
            self._prior_transition_events.clear()
            return
        self._audit_increment("optimizer_updated_gaussians", len(self._gaussian_ids))
        commits: list[int] = []
        for gaussian_id in self._gaussian_ids:
            dependencies = (begin, *self._pending_gradients.get(gaussian_id, ()))
            commits.append(self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.UPDATE_COMMIT),
                gaussian_id=gaussian_id, state_version=self._state_version,
                resource_class=int(ResourceClass.UPDATE),
                address_token=gaussian_id * self.state_record_bytes,
                data_bytes=self.state_record_bytes,
                template_id=UPDATE_TEMPLATE_ID, field_mask=field_mask,
            ), dependencies=dependencies))
        self._state_ready_event = self._emit_update_end(
            flags=UPDATE_BEGIN_OPTIMIZER, begin=begin,
            dependencies=commits or [begin], field_mask=field_mask,
        )
        self._pending_gradients.clear()
        self._pending_backward_events.clear()
        self._prior_transition_events.clear()
        self._state_version += 1

    def _emit_update_begin(self, *, flags: int, dependencies: Any) -> int:
        event = self._builder.emit(TraceEvent(
            iteration_id=self._iteration,
            primitive_kind=int(PrimitiveKind.UPDATE_BEGIN),
            state_version=self._state_version,
            resource_class=int(ResourceClass.UPDATE),
            template_id=UPDATE_TEMPLATE_ID,
            flags=int(flags),
        ), dependencies=dependencies)
        self._audit_increment("update_begin_events")
        return event

    def _emit_update_end(self, *, flags: int, begin: int, dependencies: Any,
                         field_mask: int) -> int:
        event = self._builder.emit(TraceEvent(
            iteration_id=self._iteration,
            primitive_kind=int(PrimitiveKind.UPDATE_END),
            state_version=self._state_version,
            resource_class=int(ResourceClass.UPDATE),
            template_id=UPDATE_TEMPLATE_ID,
            flags=int(flags), reduction_key=int(begin), field_mask=int(field_mask),
        ), dependencies=dependencies)
        self._audit_increment("update_end_events")
        return event

    def _ensure_collection_begin(self) -> int:
        if not self._collection_active:
            raise RuntimeError("collection modification occurred outside densify_and_prune")
        if self._collection_begin_event is None:
            self._collection_begin_event = self._emit_update_begin(
                flags=UPDATE_BEGIN_COLLECTION,
                dependencies=(*self._pending_backward_events, *self._prior_transition_events),
            )
            self._pending_backward_events.clear()
        return self._collection_begin_event

    def _start_collection_transaction(self) -> None:
        if self._collection_active:
            raise RuntimeError("nested collection modification transaction")
        self._collection_active = True
        self._collection_begin_event = None
        self._collection_events = []

    def _finish_collection_transaction(self) -> None:
        if not self._collection_active:
            raise RuntimeError("collection modification transaction is not active")
        self._collection_active = False
        if self._collection_events:
            self._audit_increment("collection_modification_transactions")
            if self._collection_begin_event is None:
                raise RuntimeError("collection transaction has no begin event")
            end = self._emit_update_end(
                flags=UPDATE_BEGIN_COLLECTION,
                begin=self._collection_begin_event,
                dependencies=self._collection_events,
                field_mask=STATE_FIELD_MASK,
            )
            self._prior_transition_events = [end]
            self._state_ready_event = end
            self._state_version += 1
        self._collection_begin_event = None
        self._collection_events = []

    def _emit_set_modification(self, gaussian_id: int, *, flags: int,
                               reduction_key: int,
                               dependencies: Any = ()) -> int:
        begin = self._ensure_collection_begin()
        event = self._builder.emit(TraceEvent(
            iteration_id=self._iteration,
            primitive_kind=int(PrimitiveKind.SET_MODIFICATION),
            gaussian_id=int(gaussian_id), state_version=self._state_version,
            resource_class=int(ResourceClass.UPDATE),
            address_token=int(gaussian_id) * self.state_record_bytes,
            data_bytes=self.state_record_bytes,
            template_id=MODIFICATION_TEMPLATE_ID, field_mask=STATE_FIELD_MASK,
            flags=int(flags), reduction_key=int(reduction_key),
        ), dependencies=(begin, *(int(value) for value in dependencies)))
        self._collection_events.append(event)
        self._audit_increment("collection_modification_events")
        self._audit_increment(
            f"collection_{ModificationKind(flags).name.lower()}_events"
        )
        return event

    @staticmethod
    def _optimizer_field_mask(optimizer: Any) -> int:
        fields = {
            "xyz": FIELD_POSITION,
            "density": FIELD_DENSITY,
            "scaling": FIELD_SCALE,
            "rotation": FIELD_ROTATION,
        }
        mask = 0
        for group in optimizer.param_groups:
            parameters = group.get("params", ())
            if parameters and getattr(parameters[0], "grad", None) is not None:
                try:
                    mask |= fields[str(group["name"])]
                except KeyError as error:
                    raise RuntimeError(
                        f"unsupported optimizer parameter group: {group.get('name')}"
                    ) from error
        return mask

    def _initialize_or_validate_gaussians(self, count: int) -> None:
        self._ensure_gaussians(count)

    def _ensure_gaussians(self, count: int) -> None:
        if count < 0:
            raise ValueError("Gaussian count cannot be negative")
        if not self._gaussian_ids_initialized:
            self._gaussian_ids = list(range(count))
            self._next_gaussian = count
            self._initial_gaussian_count = count
            self._gaussian_ids_initialized = True
            return
        if len(self._gaussian_ids) != count:
            raise RuntimeError(
                f"model Gaussian count {count} does not match stable ID count "
                f"{len(self._gaussian_ids)}"
            )

    def _audit_increment(self, name: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("capture audit increments must be non-negative")
        self._audit[name] = self._audit.get(name, 0) + int(amount)
