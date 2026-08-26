from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gala_sim.adapters.trace_capture import (
    LOSS_L1,
    LOSS_SSIM,
    LOSS_TV,
    FIELD_DENSITY,
    FIELD_POSITION,
    FIELD_SCALE,
    MOD_CLONE_CHILD,
    MOD_CLONE_PARENT,
    MOD_PRUNE,
    MOD_SPLIT_CHILD,
    MOD_SPLIT_PARENT,
    RASTER_TEMPLATE_ID,
    STATE_FIELD_MASK,
    VOXEL_TEMPLATE_ID,
    TraceSession,
    _QueryContext,
)
from gala_sim.clamp import (
    PrimitiveKind,
    ResourceClass,
    TraceBuilder,
    TraceEvent,
    UpdateBeginKind,
)
from gala_sim.trace import Trace, TraceValidationError, validate_trace


def _rows(trace, kind: PrimitiveKind) -> np.ndarray:
    return trace.events[trace.events["primitive_kind"] == int(kind)]


class _ArrayRef:
    def __init__(self, shape: tuple[int, ...], pointer: int) -> None:
        self.shape = shape
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer


def _queue_query(
    session: TraceSession, records: np.ndarray, *, binning_pointer: int,
    output_pointer: int, template_id: int,
) -> _ArrayRef:
    query_shape = (1, 1, 1) if template_id == VOXEL_TEMPLATE_ID else (1, 1)
    output = _ArrayRef(query_shape, output_pointer)
    gaussian_count = int(records[0, 2]) + 1
    session._capture_query(
        _ArrayRef((gaussian_count, 3), 0),
        _ArrayRef((records.shape[0],), binning_pointer),
        output, 1, query_shape, lambda: records,
        template_id=template_id, field_mask=STATE_FIELD_MASK,
    )
    return output


def _single_relation_records(gaussian_index: int = 0) -> np.ndarray:
    return np.asarray([
        [0, 0, gaussian_index, 0],
        [1, 0, 0, 0],
    ], dtype=np.int64)


def test_capture_expands_each_valid_mask_bit_into_a_query_relation(tmp_path: Path) -> None:
    session = TraceSession(tmp_path / "trace", chunk_events=16)
    session._ensure_gaussians(2)
    tile_one_key = np.int64(np.uint64(1) << np.uint64(32))
    records = np.asarray([
        [0, 0, 0, 0],
        [0, 1, 1, tile_one_key],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 16, 0],
    ], dtype=np.int64)
    query_count = 2 * 17
    session._audit.update({
        "official_raster_kernel_calls": 1,
        "captured_raster_kernel_calls": 1,
        "captured_query_kernel_calls": 1,
        "cuda_relation_candidates": 2,
        "captured_logical_queries": query_count,
    })
    session._emit_query_records(
        records, rendered=2, query_base=0, query_shape=(2, 17),
        binning_pointer=10, output_pointer=20,
        template_id=RASTER_TEMPLATE_ID, field_mask=STATE_FIELD_MASK,
    )
    context = session._contexts[10]
    context.loss_flags = LOSS_L1 | LOSS_SSIM
    context.ssim_radius = 1
    session._capture_backward(10, voxel=False)
    trace = session.finish()

    report = validate_trace(trace)
    assert report.counts[PrimitiveKind.RELATION.name] == 3
    assert report.counts[PrimitiveKind.QUERY_CLOSE.name] == query_count
    assert report.counts[PrimitiveKind.CONSUMER.name] == query_count
    relations = _rows(trace, PrimitiveKind.RELATION)
    assert relations["query_id"].tolist() == [0, 16, 33]
    candidates = _rows(trace, PrimitiveKind.RELATION_CANDIDATE)
    assert candidates["flags"].tolist() == [1, 1]
    candidate_event_ids = candidates["event_id"].tolist()
    assert trace.dependency_ids(relations[0]).tolist() == [candidate_event_ids[0]]
    assert trace.dependency_ids(relations[1]).tolist() == [candidate_event_ids[1]]

    consumer = _rows(trace, PrimitiveKind.CONSUMER)[16]
    dependency_queries = trace.events[trace.dependency_ids(consumer)]["query_id"].tolist()
    assert dependency_queries == [15, 16, 32, 33]


def test_pending_queries_share_one_host_copy_and_preserve_event_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = TraceSession(tmp_path / "trace", chunk_events=8)
    raster_output = _queue_query(
        session, _single_relation_records(), binning_pointer=10,
        output_pointer=20, template_id=RASTER_TEMPLATE_ID,
    )
    voxel_output = _queue_query(
        session, _single_relation_records(), binning_pointer=11,
        output_pointer=21, template_id=VOXEL_TEMPLATE_ID,
    )
    session._audit.update({
        "official_raster_kernel_calls": 1,
        "official_voxel_kernel_calls": 1,
        "captured_raster_kernel_calls": 1,
        "captured_voxel_kernel_calls": 1,
        "captured_query_kernel_calls": 2,
        "cuda_relation_candidates": 2,
    })
    l1_loss = session._wrap_loss(lambda _output: None, LOSS_L1)
    l1_loss(raster_output)
    l1_loss(voxel_output)
    copy_calls: list[int] = []
    copy_batches = session._copy_record_batches

    def counted_copy(record_batches: list[object]) -> np.ndarray:
        copy_calls.append(len(record_batches))
        return copy_batches(record_batches)

    monkeypatch.setattr(session, "_copy_record_batches", counted_copy)
    session._capture_backward(11, voxel=True)
    session._capture_backward(10, voxel=False)
    assert not session._pending_queries
    assert not session._pending_backwards
    trace = session.finish()

    report = validate_trace(trace)
    assert copy_calls == [2]
    assert session._audit["relation_record_device_batches"] == 2
    assert session._audit["relation_record_d2h_batches"] == 1
    assert report.counts[PrimitiveKind.RELATION.name] == 2
    assert report.counts[PrimitiveKind.CONSUMER.name] == 2
    assert _rows(trace, PrimitiveKind.RELATION_CANDIDATE)["template_id"].tolist() == [
        RASTER_TEMPLATE_ID, VOXEL_TEMPLATE_ID,
    ]
    assert _rows(trace, PrimitiveKind.CONSUMER)["template_id"].tolist() == [
        VOXEL_TEMPLATE_ID, RASTER_TEMPLATE_ID,
    ]
    incomplete_metadata = dict(trace.metadata)
    incomplete_audit = dict(incomplete_metadata["capture_audit"])
    incomplete_audit.pop("relation_record_d2h_batches")
    incomplete_metadata["capture_audit"] = incomplete_audit
    with pytest.raises(TraceValidationError, match="transfer audit totals are missing"):
        validate_trace(Trace(
            trace.events, trace.dependencies, trace.payload, incomplete_metadata,
        ))

    immediate = TraceSession(tmp_path / "immediate", chunk_events=8)
    immediate._ensure_gaussians(1)
    for query_base, buffer_pointer, output_pointer, template_id in (
        (0, 10, 20, RASTER_TEMPLATE_ID),
        (1, 11, 21, VOXEL_TEMPLATE_ID),
    ):
        query_shape = (1, 1, 1) if template_id == VOXEL_TEMPLATE_ID else (1, 1)
        immediate._emit_query_records(
            _single_relation_records(), rendered=1, query_base=query_base,
            query_shape=query_shape, binning_pointer=buffer_pointer,
            output_pointer=output_pointer, template_id=template_id,
            field_mask=STATE_FIELD_MASK,
        )
        immediate._contexts[buffer_pointer].loss_flags = LOSS_L1
    immediate._capture_backward(11, voxel=True)
    immediate._capture_backward(10, voxel=False)
    immediate_trace = immediate._builder.finish(metadata={"initial_gaussian_count": 1})
    for kind in (
        PrimitiveKind.RELATION, PrimitiveKind.QUERY_REDUCTION,
        PrimitiveKind.CONSUMER, PrimitiveKind.ADJOINT,
    ):
        assert _rows(trace, kind).size == _rows(immediate_trace, kind).size


def test_finish_rejects_pending_query_without_backward(tmp_path: Path) -> None:
    session = TraceSession(tmp_path / "trace")
    _queue_query(
        session, _single_relation_records(), binning_pointer=10,
        output_pointer=20, template_id=RASTER_TEMPLATE_ID,
    )
    with pytest.raises(RuntimeError, match="exactly one backward"):
        session.finish()


def test_pending_query_rejects_duplicate_backward(tmp_path: Path) -> None:
    session = TraceSession(tmp_path / "trace")
    output = _queue_query(
        session, _single_relation_records(), binning_pointer=10,
        output_pointer=20, template_id=RASTER_TEMPLATE_ID,
    )
    session._wrap_loss(lambda _output: None, LOSS_L1)(output)
    session._capture_backward(10, voxel=False)
    with pytest.raises(RuntimeError, match="duplicate backward"):
        session._capture_backward(10, voxel=False)


def test_pending_query_rejects_backward_without_loss_or_with_wrong_kind(
    tmp_path: Path,
) -> None:
    missing_loss = TraceSession(tmp_path / "missing-loss")
    _queue_query(
        missing_loss, _single_relation_records(), binning_pointer=10,
        output_pointer=20, template_id=RASTER_TEMPLATE_ID,
    )
    with pytest.raises(RuntimeError, match="without a captured loss consumer"):
        missing_loss._capture_backward(10, voxel=False)

    wrong_kind = TraceSession(tmp_path / "wrong-kind")
    output = _queue_query(
        wrong_kind, _single_relation_records(), binning_pointer=10,
        output_pointer=20, template_id=RASTER_TEMPLATE_ID,
    )
    wrong_kind._wrap_loss(lambda _output: None, LOSS_L1)(output)
    with pytest.raises(RuntimeError, match="does not match its forward context"):
        wrong_kind._capture_backward(10, voxel=True)


def test_densification_flushes_queries_before_stable_ids_change(tmp_path: Path) -> None:
    class Model:
        def __init__(self) -> None:
            self.get_xyz = np.empty((2, 3), dtype=np.float32)

    class Mask:
        def detach(self) -> "Mask":
            return self

        def cpu(self) -> "Mask":
            return self

        def numpy(self) -> np.ndarray:
            return np.asarray([True, False])

    session = TraceSession(tmp_path / "trace")
    output = _queue_query(
        session, _single_relation_records(1), binning_pointer=10,
        output_pointer=20, template_id=RASTER_TEMPLATE_ID,
    )
    session._wrap_loss(lambda _output: None, LOSS_L1)(output)
    session._capture_backward(10, voxel=False)

    def prune_original(model: Model, _mask: Mask) -> None:
        model.get_xyz = model.get_xyz[1:]

    prune = session._wrap_prune_points(prune_original)

    def densify_original(model: Model) -> None:
        assert not session._pending_queries
        assert session._audit["captured_backward_calls"] == 1
        prune(model, Mask())

    session._wrap_densify_and_prune(densify_original)(Model())
    trace = session._builder.finish(metadata={"initial_gaussian_count": 2})
    assert _rows(trace, PrimitiveKind.RELATION)["gaussian_id"].tolist() == [1]
    assert session._gaussian_ids == [1]


def test_next_iteration_flushes_queries_with_the_previous_iteration_id(
    tmp_path: Path,
) -> None:
    class Model:
        get_xyz = np.empty((1, 3), dtype=np.float32)

    session = TraceSession(tmp_path / "trace")
    session._iteration = 3
    output = _queue_query(
        session, _single_relation_records(), binning_pointer=10,
        output_pointer=20, template_id=RASTER_TEMPLATE_ID,
    )
    session._wrap_loss(lambda _output: None, LOSS_L1)(output)
    session._capture_backward(10, voxel=False)

    def update_learning_rate(_model: Model, iteration: int) -> None:
        assert iteration == 4
        assert not session._pending_queries

    session._wrap_learning_rate(update_learning_rate)(Model(), 4)
    trace = session._builder.finish(metadata={"initial_gaussian_count": 1})
    assert set(trace.events["iteration_id"].tolist()) == {3}


def test_voxel_query_offsets_match_official_x_y_z_layout() -> None:
    keys = np.zeros(8, dtype=np.uint64)
    local_queries = np.asarray([0, 1, 8, 9, 64, 65, 72, 73], dtype=np.int64)
    offsets = TraceSession._decode_query_offsets(
        keys, local_queries, (2, 2, 2), VOXEL_TEMPLATE_ID
    )
    assert offsets.tolist() == [0, 4, 2, 6, 1, 5, 3, 7]


def test_tv_consumer_uses_axis_adjacent_query_reductions() -> None:
    context = _QueryContext(
        query_base=0,
        query_shape=(2, 2, 2),
        relation_query_offsets=np.empty(0, dtype=np.int64),
        gaussian_ids=np.empty(0, dtype=np.int64),
        relation_ids=np.empty(0, dtype=np.int64),
        reduction_events=np.arange(8, dtype=np.int64),
        buffer_pointer=0,
        output_pointer=0,
        template_id=VOXEL_TEMPLATE_ID,
        field_mask=STATE_FIELD_MASK,
        loss_flags=LOSS_TV,
    )
    assert TraceSession._consumer_query_offsets(context, 0).tolist() == [0, 4, 2, 1]


def test_collection_modifications_preserve_lineage_and_active_ids(tmp_path: Path) -> None:
    session = TraceSession(tmp_path / "trace", chunk_events=8)
    session._ensure_gaussians(2)
    session._start_collection_transaction()
    clone_parent = session._emit_set_modification(
        0, flags=MOD_CLONE_PARENT, reduction_key=0
    )
    session._emit_set_modification(
        2, flags=MOD_CLONE_CHILD, reduction_key=0,
        dependencies=[clone_parent],
    )
    split_children = [
        session._emit_set_modification(
            child, flags=MOD_SPLIT_CHILD, reduction_key=1
        )
        for child in (3, 4)
    ]
    session._emit_set_modification(
        1, flags=MOD_SPLIT_PARENT, reduction_key=1,
        dependencies=split_children,
    )
    session._emit_set_modification(2, flags=MOD_PRUNE, reduction_key=2)
    session._finish_collection_transaction()
    relation = session._builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0,
        gaussian_id=3, relation_id=0, state_version=1,
        resource_class=int(ResourceClass.RELATION),
    ))
    session._builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0,
        state_version=1, resource_class=int(ResourceClass.RELATION),
    ), dependencies=[relation, session._state_ready_event])
    trace = session._builder.finish(metadata={"initial_gaussian_count": 2})
    assert validate_trace(trace).gaussian_count == 5
    assert session._state_version == 1

    broken = trace.events.copy()
    broken[-2]["gaussian_id"] = 1
    with pytest.raises(TraceValidationError, match="inactive Gaussian"):
        validate_trace(Trace(broken, trace.dependencies, trace.payload, trace.metadata))


def test_optimizer_noop_does_not_emit_commit_or_advance_version(tmp_path: Path) -> None:
    class Model:
        get_xyz = np.empty((2, 3), dtype=np.float32)

    session = TraceSession(tmp_path / "trace")
    session._ensure_gaussians(2)
    session._capture_update(Model(), field_mask=0)
    trace = session._builder.finish(metadata={"initial_gaussian_count": 2})
    assert _rows(trace, PrimitiveKind.UPDATE_COMMIT).size == 0
    assert _rows(trace, PrimitiveKind.UPDATE_BEGIN).size == 1
    assert _rows(trace, PrimitiveKind.UPDATE_END).size == 1
    assert validate_trace(trace).event_count == 2
    assert session._state_version == 0
    assert session._audit["optimizer_noop_steps"] == 1


def test_optimizer_commit_depends_on_begin_and_updates_only_real_fields(tmp_path: Path) -> None:
    class Model:
        get_xyz = np.empty((2, 3), dtype=np.float32)

    session = TraceSession(tmp_path / "trace")
    session._ensure_gaussians(2)
    session._capture_update(Model(), field_mask=FIELD_DENSITY)
    trace = session._builder.finish(metadata={"initial_gaussian_count": 2})
    report = validate_trace(trace)
    assert report.counts[PrimitiveKind.UPDATE_BEGIN.name] == 1
    commits = _rows(trace, PrimitiveKind.UPDATE_COMMIT)
    assert commits["field_mask"].tolist() == [FIELD_DENSITY, FIELD_DENSITY]
    begin = int(_rows(trace, PrimitiveKind.UPDATE_BEGIN)[0]["event_id"])
    assert all(begin in trace.dependency_ids(row) for row in commits)
    assert session._state_version == 1


def test_split_lineage_matches_official_tensor_repeat_order() -> None:
    assert TraceSession._split_child_lineage((10, 20), 2) == (10, 20, 10, 20)


def test_validator_rejects_update_commit_without_begin() -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_COMMIT), gaussian_id=0,
        state_version=0, resource_class=int(ResourceClass.UPDATE),
    ))
    with pytest.raises(TraceValidationError, match="optimizer begin"):
        validate_trace(builder.finish(metadata={"initial_gaussian_count": 1}))


def test_stable_gaussian_count_cannot_drift_silently(tmp_path: Path) -> None:
    session = TraceSession(tmp_path / "trace")
    session._ensure_gaussians(2)
    with pytest.raises(RuntimeError, match="stable ID count"):
        session._ensure_gaussians(3)


def test_next_state_query_waits_for_previous_update_end() -> None:
    builder = TraceBuilder()
    candidate_v0 = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE), gaussian_id=0,
        state_version=0, resource_class=int(ResourceClass.RELATION),
    ))
    relation_v0 = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=0, gaussian_id=0,
        relation_id=0, state_version=0, resource_class=int(ResourceClass.RELATION),
    ), dependencies=[candidate_v0])
    close_v0 = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=0,
        state_version=0, resource_class=int(ResourceClass.RELATION),
    ), dependencies=[relation_v0])
    begin = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_BEGIN), state_version=0,
        resource_class=int(ResourceClass.UPDATE), flags=int(UpdateBeginKind.OPTIMIZER),
    ), dependencies=[close_v0])
    commit = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_COMMIT), gaussian_id=0,
        state_version=0, resource_class=int(ResourceClass.UPDATE), field_mask=FIELD_DENSITY,
    ), dependencies=[begin])
    update_end = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_END), state_version=0,
        reduction_key=begin, resource_class=int(ResourceClass.UPDATE),
        flags=int(UpdateBeginKind.OPTIMIZER), field_mask=FIELD_DENSITY,
    ), dependencies=[commit])
    candidate_v1 = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE), gaussian_id=0,
        state_version=1, resource_class=int(ResourceClass.RELATION),
    ), dependencies=[update_end])
    relation_v1 = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.RELATION), query_id=1, gaussian_id=0,
        relation_id=1, state_version=1, resource_class=int(ResourceClass.RELATION),
    ), dependencies=[candidate_v1])
    close_v1 = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.QUERY_CLOSE), query_id=1,
        state_version=1, resource_class=int(ResourceClass.RELATION),
    ), dependencies=[relation_v1, update_end])
    trace = builder.finish(metadata={"initial_gaussian_count": 1})
    assert validate_trace(trace).event_count == 9

    broken = trace.events.copy()
    broken[candidate_v1]["dependency_count"] = 0
    with pytest.raises(TraceValidationError, match="prior update end"):
        validate_trace(Trace(broken, trace.dependencies, trace.payload, trace.metadata))


def test_state_changing_update_rejects_old_version_access() -> None:
    builder = TraceBuilder()
    begin = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_BEGIN), state_version=0,
        resource_class=int(ResourceClass.UPDATE), flags=int(UpdateBeginKind.OPTIMIZER),
    ))
    commit = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_COMMIT), gaussian_id=0,
        state_version=0, resource_class=int(ResourceClass.UPDATE), field_mask=FIELD_DENSITY,
    ), dependencies=[begin])
    update_end = builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.UPDATE_END), state_version=0,
        reduction_key=begin, resource_class=int(ResourceClass.UPDATE),
        flags=int(UpdateBeginKind.OPTIMIZER), field_mask=FIELD_DENSITY,
    ), dependencies=[commit])
    builder.emit(TraceEvent(
        primitive_kind=int(PrimitiveKind.CACHE_REQUEST), query_id=0, gaussian_id=0,
        state_version=0, address_token=0, data_bytes=64,
        resource_class=int(ResourceClass.CACHE),
    ), dependencies=[update_end])
    with pytest.raises(TraceValidationError, match="closed state version"):
        validate_trace(builder.finish(metadata={"initial_gaussian_count": 1}))


def test_optimizer_field_mask_uses_only_parameter_groups_with_gradients() -> None:
    class Parameter:
        def __init__(self, grad: object | None) -> None:
            self.grad = grad

    class Optimizer:
        param_groups = [
            {"name": "xyz", "params": [Parameter(object())]},
            {"name": "density", "params": [Parameter(None)]},
            {"name": "scaling", "params": [Parameter(object())]},
            {"name": "rotation", "params": [Parameter(None)]},
        ]

    assert TraceSession._optimizer_field_mask(Optimizer()) == FIELD_POSITION | FIELD_SCALE


def test_clone_and_split_wrappers_follow_official_tensor_order(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def __init__(self) -> None:
            self.scaling = torch.tensor([[0.5], [2.0], [3.0], [0.5]])

        @property
        def get_xyz(self):
            return torch.empty((self.scaling.shape[0], 3))

        @property
        def get_scaling(self):
            return self.scaling

    session = TraceSession(tmp_path / "trace")
    model = Model()
    session._ensure_gaussians(4)

    def original_postfix(model: Model, new_xyz, _density, new_scaling, *_args) -> None:
        model.scaling = torch.cat((model.scaling, new_scaling), dim=0)

    postfix = session._wrap_densification_postfix(original_postfix)

    def original_prune(model: Model, mask) -> None:
        model.scaling = model.scaling[~mask]

    prune = session._wrap_prune_points(original_prune)

    def original_clone(model: Model, grads, threshold, scale_threshold) -> None:
        selected = torch.logical_and(
            torch.norm(grads, dim=-1) >= threshold,
            torch.max(model.get_scaling, dim=1).values <= scale_threshold,
        )
        count = int(selected.sum())
        postfix(
            model, torch.empty((count, 3)), torch.empty((count, 1)),
            model.get_scaling[selected], torch.empty((count, 4)), torch.empty(count),
        )

    def original_split(model: Model, grads, threshold, scale_threshold, count=2) -> None:
        padded = torch.zeros(model.get_xyz.shape[0])
        padded[:grads.shape[0]] = grads.squeeze()
        selected = torch.logical_and(
            padded >= threshold,
            torch.max(model.get_scaling, dim=1).values > scale_threshold,
        )
        child_scaling = model.get_scaling[selected].repeat(count, 1)
        child_count = int(child_scaling.shape[0])
        postfix(
            model, torch.empty((child_count, 3)), torch.empty((child_count, 1)),
            child_scaling, torch.empty((child_count, 4)), torch.empty(child_count),
        )
        prune(model, torch.cat((selected, torch.zeros(child_count, dtype=torch.bool))))

    clone = session._wrap_densify_clone(original_clone)
    split = session._wrap_densify_split(original_split)
    gradients = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    session._start_collection_transaction()
    clone(model, gradients, 0.5, 1.0)
    split(model, gradients, 0.5, 1.0, 2)
    session._finish_collection_transaction()

    trace = session._builder.finish(metadata={"initial_gaussian_count": 4})
    assert validate_trace(trace).event_count == 9
    assert session._gaussian_ids == [0, 3, 4, 5, 6, 7, 8]
    split_children = _rows(trace, PrimitiveKind.SET_MODIFICATION)
    split_children = split_children[split_children["flags"] == MOD_SPLIT_CHILD]
    assert split_children["gaussian_id"].tolist() == [5, 6, 7, 8]
    assert split_children["reduction_key"].tolist() == [1, 2, 1, 2]


def test_collection_failure_prevents_finishing_partial_trace(tmp_path: Path) -> None:
    class Model:
        get_xyz = np.empty((1, 3), dtype=np.float32)

    session = TraceSession(tmp_path / "trace")
    session._ensure_gaussians(1)

    def failing_original(_model: Model) -> None:
        raise RuntimeError("official collection failure")

    wrapped = session._wrap_densify_and_prune(failing_original)
    with pytest.raises(RuntimeError, match="official collection failure"):
        wrapped(Model())
    with pytest.raises(RuntimeError, match="cannot finish after a collection failure"):
        session.finish()


def test_trace_runner_does_not_finish_after_official_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from gala_sim.adapters import trace_runner

    calls: list[str] = []

    class Session:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def install(self) -> None:
            calls.append("install")

        def restore(self) -> None:
            calls.append("restore")

        def finish(self) -> None:
            calls.append("finish")

    def fail_run_path(*_args, **_kwargs) -> None:
        raise RuntimeError("official training failed")

    monkeypatch.setattr(trace_runner, "TraceSession", Session)
    monkeypatch.setattr(trace_runner.runpy, "run_path", fail_run_path)
    with pytest.raises(RuntimeError, match="official training failed"):
        trace_runner.main([
            "--trace-output", str(tmp_path / "trace"), str(tmp_path / "train.py")
        ])
    assert calls == ["install", "restore"]
