"""Runtime-only hooks for capturing the official R²-Gaussian CUDA workload."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gala_sim.clamp import ChunkedTraceBuilder, PrimitiveKind, ResourceClass, TraceEvent
from gala_sim.trace import Trace, TraceWriter

from .buffer_decoder import load_buffer_decoder


@dataclass
class _QueryContext:
    query_id: int
    gaussian_ids: tuple[int, ...]
    relation_events: dict[int, int]
    relation_ids: dict[int, int]
    forward_events: dict[int, int]
    consumer_events: dict[int, int]
    cache_events: dict[int, tuple[int, int]]
    buffer_pointer: int


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
    _pending_gradients: dict[int, list[int]] = field(default_factory=dict, init=False)
    _gaussian_ids: list[int] = field(default_factory=list, init=False)
    _query_capture_allowed: bool = field(default=False, init=False)
    _installed: bool = field(default=False, init=False)
    _originals: list[tuple[Any, str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._builder = ChunkedTraceBuilder(
            self.chunk_events, chunk_root=self.output_root / ".capture_chunks"
        )

    def install(self) -> None:
        if self._installed:
            return
        import xray_gaussian_rasterization_voxelization as extension

        self._decoder = load_buffer_decoder()
        self._patch(extension.GaussianRasterizer, "forward", self._wrap_query_forward)
        self._patch(extension.GaussianVoxelizer, "forward", self._wrap_query_forward)
        self._patch(extension._C, "rasterize_gaussians", self._wrap_rasterize)
        self._patch(extension._C, "voxelize_gaussians", self._wrap_voxelize)
        self._patch(extension._C, "rasterize_gaussians_backward", self._wrap_rasterize_backward)
        self._patch(extension._C, "voxelize_gaussians_backward", self._wrap_voxel_backward)

        from r2_gaussian.gaussian.gaussian_model import GaussianModel

        self._patch(GaussianModel, "update_learning_rate", self._wrap_learning_rate)
        self._patch(GaussianModel, "training_setup", self._wrap_training_setup)
        self._patch(GaussianModel, "prune_points", self._wrap_prune_points)
        self._patch(GaussianModel, "densification_postfix", self._wrap_densification_postfix)
        self._installed = True

    def restore(self) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        self._installed = False

    def finish(self) -> Trace:
        trace = self._builder.finish(metadata={
            "model": "R2-Gaussian",
            "dataset": "Chest",
            "capture_backend": "official_cuda_buffers_and_call_hooks",
            "state_record_bytes": self.state_record_bytes,
            "trace_chunk_events": self.chunk_events,
            "trace_capture_status": "real_extension_buffers",
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

    def _wrap_query_forward(self, original: Any) -> Any:
        def wrapped(rasterizer: Any, *args: Any, **kwargs: Any) -> Any:
            previous = self._query_capture_allowed
            import torch

            self._query_capture_allowed = bool(torch.is_grad_enabled())
            try:
                return original(rasterizer, *args, **kwargs)
            finally:
                self._query_capture_allowed = previous
        return wrapped

    def _capture_enabled(self) -> bool:
        """Return the grad-mode decision captured at the official query boundary."""
        return self._query_capture_allowed

    def _wrap_learning_rate(self, original: Any) -> Any:
        def wrapped(model: Any, iteration: int, *args: Any, **kwargs: Any) -> Any:
            self._iteration = int(iteration)
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            return original(model, iteration, *args, **kwargs)
        return wrapped

    def _wrap_training_setup(self, original: Any) -> Any:
        def wrapped(model: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(model, *args, **kwargs)
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            optimizer = model.optimizer
            original_step = optimizer.step

            def step(*step_args: Any, **step_kwargs: Any) -> Any:
                result_step = original_step(*step_args, **step_kwargs)
                self._capture_update(model)
                return result_step

            optimizer.step = step
            return result
        return wrapped

    def _wrap_prune_points(self, original: Any) -> Any:
        def wrapped(model: Any, mask: Any, *args: Any, **kwargs: Any) -> Any:
            self._ensure_gaussians(int(model.get_xyz.shape[0]))
            removed = mask.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
            removed_ids = [self._gaussian_ids[index] for index, flag in enumerate(removed)
                           if flag and index < len(self._gaussian_ids)]
            result = original(model, mask, *args, **kwargs)
            self._gaussian_ids = [value for index, value in enumerate(self._gaussian_ids)
                                  if index < len(removed) and not removed[index]]
            for gaussian_id in removed_ids:
                self._emit_set_modification(gaussian_id)
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
            for gaussian_id in new_ids:
                self._emit_set_modification(gaussian_id)
            return result
        return wrapped

    def _capture_raster(self, args: tuple[Any, ...], result: tuple[Any, ...]) -> None:
        rendered, _, _, geometry, binning, image = result
        means = args[0]
        height, width = int(args[10]), int(args[11])
        self._capture_query(
            means, geometry, binning, image, int(rendered), height * width,
            lambda: self._decoder.copy_raster_point_list(binning, int(rendered)),
            lambda: self._decoder.copy_raster_point_keys(binning, int(rendered)),
            lambda: self._decoder.raster_valid_masks(
                geometry, binning, int(means.shape[0]), int(rendered), height, width
            ),
        )

    def _capture_voxel(self, args: tuple[Any, ...], result: tuple[Any, ...]) -> None:
        rendered, _, _, _, _, geometry, binning, image = result
        means = args[0]
        dimensions = tuple(int(value) for value in args[6:9])
        self._capture_query(
            means, geometry, binning, image, int(rendered), int(np.prod(dimensions)),
            lambda: self._decoder.copy_voxel_point_list(binning, int(rendered)),
            lambda: self._decoder.copy_voxel_point_keys(binning, int(rendered)),
            lambda: self._decoder.voxel_valid_masks(
                geometry, binning, int(means.shape[0]), int(rendered), *dimensions
            ),
        )

    def _capture_query(
        self, means: Any, geometry: Any, binning: Any, image: Any, rendered: int,
        image_elements: int, point_list_fn: Callable[[], Any], point_keys_fn: Callable[[], Any],
        valid_masks_fn: Callable[[], Any],
    ) -> None:
        query_id = self._next_query
        self._next_query += 1
        gaussian_count = int(means.shape[0])
        self._ensure_gaussians(gaussian_count)
        if rendered <= 0:
            self._emit_query_close(query_id, ())
            self._contexts[int(binning.data_ptr())] = _QueryContext(
                query_id, (), {}, {}, {}, {}, {}, int(binning.data_ptr())
            )
            return
        point_list = point_list_fn().detach().cpu().numpy().astype(np.int64, copy=False)
        point_keys = point_keys_fn().detach().cpu().numpy().astype(np.uint64, copy=False)
        masks = valid_masks_fn().detach().cpu().numpy().view(np.uint32)
        candidate_by_gaussian: dict[int, list[int]] = {}
        valid_by_gaussian: dict[int, bool] = {}
        for index, (raw_gaussian, key) in enumerate(zip(point_list, point_keys)):
            gaussian_index = int(raw_gaussian)
            gaussian_id = self._gaussian_ids[gaussian_index]
            mask_nonzero = any(int(word).bit_count() > 0 for word in masks[index])
            candidate_event = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE),
                query_id=query_id, gaussian_id=gaussian_id,
                state_version=self._state_version,
                resource_class=int(ResourceClass.RELATION),
                address_token=int(key), data_bytes=self.relation_candidate_bytes,
                flags=1 if mask_nonzero else 0,
            ))
            candidate_by_gaussian.setdefault(gaussian_index, []).append(candidate_event)
            valid_by_gaussian[gaussian_index] = valid_by_gaussian.get(gaussian_index, False) or mask_nonzero
        relation_events: dict[int, int] = {}
        relation_ids: dict[int, int] = {}
        forward_events: dict[int, int] = {}
        consumer_events: dict[int, int] = {}
        cache_events: dict[int, tuple[int, int]] = {}
        for gaussian_index in sorted(candidate_by_gaussian):
            if not valid_by_gaussian.get(gaussian_index, False):
                continue
            gaussian_id = self._gaussian_ids[gaussian_index]
            relation_id = self._next_relation
            self._next_relation += 1
            relation = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.RELATION), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_id, resource_class=int(ResourceClass.RELATION),
                address_token=gaussian_id * self.state_record_bytes,
            ), dependencies=candidate_by_gaussian[gaussian_index])
            relation_events[gaussian_index] = relation
            relation_ids[gaussian_index] = relation_id
        self._emit_query_close(query_id, tuple(relation_events.values()))
        for gaussian_index, relation in relation_events.items():
            gaussian_id = self._gaussian_ids[gaussian_index]
            request = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_ids[gaussian_index],
                resource_class=int(ResourceClass.CACHE),
                address_token=gaussian_id * self.state_record_bytes,
                data_bytes=self.state_record_bytes,
            ), dependencies=[relation])
            returned = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.CACHE_RETURN), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                resource_class=int(ResourceClass.CACHE),
                address_token=gaussian_id * self.state_record_bytes,
                data_bytes=self.state_record_bytes,
            ), dependencies=[request])
            forward = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.FORWARD), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_ids[gaussian_index], resource_class=int(ResourceClass.ISSUE),
                address_token=gaussian_id * self.state_record_bytes,
            ), dependencies=[relation, returned])
            forward_events[gaussian_index] = forward
            cache_events[gaussian_index] = (request, returned)
        reduction = self._builder.emit(TraceEvent(
            iteration_id=self._iteration,
            primitive_kind=int(PrimitiveKind.QUERY_REDUCTION), query_id=query_id,
            state_version=self._state_version, reduction_key=query_id,
            resource_class=int(ResourceClass.QUERY),
        ), dependencies=tuple(forward_events.values()))
        for gaussian_index, forward in forward_events.items():
            gaussian_id = self._gaussian_ids[gaussian_index]
            consumer_events[gaussian_index] = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.CONSUMER), query_id=query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=relation_ids[gaussian_index], consumer_id=query_id,
                resource_class=int(ResourceClass.QUERY),
            ), dependencies=[reduction])
        self._contexts[int(binning.data_ptr())] = _QueryContext(
            query_id, tuple(self._gaussian_ids[index] for index in forward_events),
            relation_events, relation_ids, forward_events, consumer_events, cache_events,
            int(binning.data_ptr()),
        )

    def _emit_query_close(self, query_id: int, dependencies: tuple[int, ...]) -> None:
        self._builder.emit(TraceEvent(
            iteration_id=self._iteration,
            primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=query_id,
            state_version=self._state_version, resource_class=int(ResourceClass.RELATION),
        ), dependencies=dependencies)

    def _capture_backward(self, buffer_pointer: int, *, voxel: bool) -> None:
        context = self._contexts.get(buffer_pointer)
        if context is None:
            return
        for gaussian_index, consumer in context.consumer_events.items():
            gaussian_id = self._gaussian_ids[gaussian_index] if gaussian_index < len(self._gaussian_ids) else context.gaussian_ids[0]
            adjoint = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.ADJOINT), query_id=context.query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=context.relation_ids[gaussian_index],
                resource_class=int(ResourceClass.ISSUE),
                address_token=gaussian_id * self.state_record_bytes,
            ), dependencies=[consumer])
            gradient = self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.GRADIENT_REDUCTION), query_id=context.query_id,
                gaussian_id=gaussian_id, state_version=self._state_version,
                relation_id=context.relation_ids[gaussian_index],
                reduction_key=gaussian_id, resource_class=int(ResourceClass.QUERY),
                address_token=gaussian_id * self.state_record_bytes,
            ), dependencies=[adjoint])
            self._pending_gradients.setdefault(gaussian_id, []).append(gradient)

    def _capture_update(self, model: Any) -> None:
        self._ensure_gaussians(int(model.get_xyz.shape[0]))
        for index, gaussian_id in enumerate(self._gaussian_ids):
            dependencies = tuple(self._pending_gradients.get(gaussian_id, ()))
            self._builder.emit(TraceEvent(
                iteration_id=self._iteration,
                primitive_kind=int(PrimitiveKind.UPDATE_COMMIT),
                gaussian_id=gaussian_id, state_version=self._state_version,
                resource_class=int(ResourceClass.UPDATE),
                address_token=gaussian_id * self.state_record_bytes,
                data_bytes=self.state_record_bytes,
            ), dependencies=dependencies)
        self._pending_gradients.clear()
        self._state_version += 1

    def _emit_set_modification(self, gaussian_id: int) -> None:
        self._builder.emit(TraceEvent(
            iteration_id=self._iteration,
            primitive_kind=int(PrimitiveKind.SET_MODIFICATION),
            gaussian_id=int(gaussian_id), state_version=self._state_version,
            resource_class=int(ResourceClass.UPDATE),
            address_token=int(gaussian_id) * self.state_record_bytes,
            data_bytes=self.state_record_bytes,
        ))

    def _ensure_gaussians(self, count: int) -> None:
        if count < 0:
            raise ValueError("Gaussian count cannot be negative")
        while len(self._gaussian_ids) < count:
            self._gaussian_ids.append(self._next_gaussian)
            self._next_gaussian += 1
        if len(self._gaussian_ids) > count:
            self._gaussian_ids = self._gaussian_ids[:count]
