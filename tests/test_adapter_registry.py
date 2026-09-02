from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import numpy as np

from gala_sim.adapters import get_model_adapter, model_descriptors, prepare_campaign
from gala_sim.adapters.trace_capture import get_trace_hook_profile
from gala_sim.clamp import PrimitiveKind, TraceBuilder, TraceEvent
from gala_sim.config import load_config
from gala_sim.trace import TraceWriter
from gala_sim.workspace import WorkspacePaths


ROOT = Path(__file__).resolve().parents[1]


def test_model_registry_exposes_four_distinct_implementations(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    descriptors = model_descriptors()
    assert set(descriptors) == {"r2_gaussian", "fact_gs", "exact_gs", "gr_gaussian"}
    assert descriptors["gr_gaussian"].implementation == "independent_reimplementation"
    for model_id in descriptors:
        adapter = get_model_adapter(model_id, workspace=workspace)
        assert adapter.descriptor.id == model_id


def test_official_adapter_builds_command_without_repository_cwd(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    adapter = get_model_adapter("fact_gs", workspace=workspace)
    command = adapter.build_command(tmp_path / "dataset", tmp_path / "output")
    assert command[1] == "train_recon.py"
    assert any(value.startswith("model.data_source_path=") for value in command)
    assert any(value.startswith("model.model_path=") for value in command)
    assert "model.init_mode=gradient" in command
    assert "model.eval=false" in command
    assert "eval.eval_in_training=false" in command


def test_model_adapter_prefers_portable_workspace_python(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    workspace_python = workspace.build / "envs/fact_gs/bin/python"
    workspace_python.parent.mkdir(parents=True)
    workspace_python.write_text("", encoding="utf-8")

    adapter = get_model_adapter("fact_gs", workspace=workspace)

    assert adapter.python_executable == workspace_python


def test_explicit_model_python_overrides_workspace_environment(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    workspace_python = workspace.build / "envs/fact_gs/bin/python"
    workspace_python.parent.mkdir(parents=True)
    workspace_python.write_text("", encoding="utf-8")
    explicit = tmp_path / "explicit-python"

    adapter = get_model_adapter(
        "fact_gs", workspace=workspace, python_executable=explicit,
    )

    assert adapter.python_executable == explicit


def test_fact_adapter_uses_precomputed_initialization_when_available(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    adapter = get_model_adapter("fact_gs", workspace=workspace)
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    np.save(dataset / "init_prepared.npy", np.ones((2, 4), dtype=np.float32))

    command = adapter.build_command(dataset, tmp_path / "output")

    assert "model.init_mode=precomputed" in command
    assert "model.eval=false" in command


def test_exact_adapter_uses_the_upstream_method_and_output_contract(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    adapter = get_model_adapter("exact_gs", workspace=workspace)
    command = adapter.build_command(tmp_path / "dataset", tmp_path / "output")
    assert "--method=Exact_GS" in command
    assert any(value.startswith("--source_path=") for value in command)
    assert any(value.startswith("--ply_path=") for value in command)
    assert not any(value.startswith("--model_path=") for value in command)
    assert adapter.descriptor.output_strategy == "dataset_output_directory"
    profile = get_trace_hook_profile("exact_gs")
    assert profile.raster_extension == "exact_gaussian_rasterization"
    assert profile.voxel_extension == "xray_gaussian_rasterization_voxelization"
    assert profile.pointwise_loss == "l2_loss"


def test_model_descriptors_publish_trace_stage_boundaries() -> None:
    descriptors = model_descriptors()
    for descriptor in descriptors.values():
        assert "backward" in descriptor.stage_boundaries
        assert "optimizer_step" in descriptor.stage_boundaries


def _exported_dataset(root: Path, *, reference: bool) -> Path:
    projections = root / "projections"
    projections.mkdir(parents=True)
    np.save(projections / "000.npy", np.ones((2, 3), dtype=np.float32))
    np.save(projections / "001.npy", np.ones((2, 3), dtype=np.float32))
    if reference:
        reconstruction = root / "recon"
        reconstruction.mkdir()
        np.save(reconstruction / "volume.npy", np.ones((2, 2, 2), dtype=np.float32))
    (root / "metadata.json").write_text(json.dumps({
        "angles_degrees": [0.0, 1.0], "detector_shape": [2, 3],
        "volume_shape": [2, 2, 2], "DSO": 50.0, "DSD": 100.0,
    }), encoding="utf-8")
    return root


def _chest_dataset(root: Path) -> Path:
    (root / "proj_train").mkdir(parents=True)
    (root / "proj_test").mkdir()
    np.save(root / "proj_train/proj_train_0000.npy", np.ones((2, 3), dtype=np.float32))
    np.save(root / "proj_test/proj_test_0000.npy", np.ones((2, 3), dtype=np.float32))
    np.save(root / "vol_gt.npy", np.ones((2, 2, 2), dtype=np.float32))
    np.save(root / "init_chest.npy", np.ones((2, 4), dtype=np.float32))
    (root / "meta_data.json").write_text(json.dumps({
        "scanner": {"nDetector": [2, 3], "nVoxel": [2, 2, 2],
                    "DSO": 50.0, "DSD": 100.0},
        "proj_train": [{"file_path": "proj_train/proj_train_0000.npy", "angle": 0.0}],
        "proj_test": [{"file_path": "proj_test/proj_test_0000.npy", "angle": 1.0}],
        "vol": "vol_gt.npy",
    }), encoding="utf-8")
    return root


def test_campaign_builder_composes_all_registered_model_dataset_pairs(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    for model_id in ("r2_gaussian", "fact_gs", "exact_gs"):
        (workspace.upstream / model_id).mkdir(parents=True, exist_ok=True)
    datasets = {
        "chest": _chest_dataset(tmp_path / "chest"),
        "walnut": _exported_dataset(tmp_path / "walnut", reference=False),
        "hdtomo_usb": _exported_dataset(tmp_path / "hdtomo", reference=True),
    }
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    for model_id in model_descriptors():
        for dataset_id, dataset_root in datasets.items():
            run = prepare_campaign(
                model_id, dataset_id, dataset_root, config, workspace=workspace,
            )
            assert run.model_name == model_descriptors()[model_id].display_name
            assert run.dataset_name == dataset_id
            assert run.dataset_root.is_dir()
            assert (run.dataset_root / "meta_data.json").is_file()
            assert run.output_root == (workspace.results / model_id / dataset_id).resolve()


def test_campaign_output_roots_are_isolated_by_model_and_dataset(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    for model_id in ("r2_gaussian", "fact_gs", "exact_gs"):
        (workspace.upstream / model_id).mkdir(parents=True, exist_ok=True)
    dataset_roots = {
        "chest": _chest_dataset(tmp_path / "chest"),
        "walnut": _exported_dataset(tmp_path / "walnut", reference=False),
        "hdtomo_usb": _exported_dataset(tmp_path / "hdtomo", reference=True),
    }
    config = load_config(ROOT / "configs/architecture/gala.yaml")

    observed: set[Path] = set()
    for model_id in model_descriptors():
        for dataset_id, dataset_root in dataset_roots.items():
            run = prepare_campaign(
                model_id, dataset_id, dataset_root, config, workspace=workspace,
            )
            expected = (workspace.results / model_id / dataset_id).resolve()
            assert expected not in observed
            observed.add(expected)
            assert run.output_root == expected

    assert len(observed) == 12


def test_exact_capture_runs_official_entry_with_model_specific_hooks(
    tmp_path: Path, monkeypatch,
) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    source = workspace.upstream / "exact_gs"
    source.mkdir(parents=True)
    (source / "train.py").write_text("", encoding="utf-8")
    dataset_root = _chest_dataset(tmp_path / "chest")
    output_root = workspace.results / "exact_gs" / "chest"
    adapter = get_model_adapter(
        "exact_gs", workspace=workspace, output_root=output_root,
    )
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    run = adapter.prepare(SimpleNamespace(root=dataset_root, id="chest"), config)
    source_binding = next(
        value for value in run.official_command if value.startswith("--source_path=")
    )
    assert source_binding == f"--source_path={output_root / 'input'}"
    captured: dict[str, object] = {}

    def fake_trace_process(command, **kwargs):
        captured["command"] = tuple(command)
        captured.update(kwargs)
        kwargs["prepare_fn"]()
        trace_root = Path(kwargs["trace_root"])
        builder = TraceBuilder()
        builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE)))
        TraceWriter().write(builder.finish(metadata={}), trace_root)
        volume = output_root / "model-output/point_cloud/iteration_1/vol_pred.npy"
        volume.parent.mkdir(parents=True)
        np.save(volume, np.ones((2, 2, 2), dtype=np.float32))
        return {"status": "passed"}

    class Sink:
        chunk_events = 8

        def __init__(self) -> None:
            self.event_count = 0
            self.closed = False

        def push(self, events, dependencies, payload) -> None:
            self.event_count += len(events)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("gala_sim.adapters.registry.run_trace_process", fake_trace_process)
    sink = Sink()
    artifact = adapter.capture_trace(run, sink)

    command = captured["command"]
    assert "--model-id" in command
    assert command[command.index("--model-id") + 1] == "exact_gs"
    assert "--dataset-id" in command
    assert command[command.index("--dataset-id") + 1] == "chest"
    assert "--capture-iteration-range" in command
    assert command[command.index("--capture-iteration-range") + 1] == "1:1"
    assert "--stop-after-capture-range" in command
    assert "--test_iterations" in command
    assert command[command.index("--test_iterations") + 1] == "2"
    assert f"--ply_path={dataset_root / 'init_chest.npy'}" in command
    assert captured["cwd"] == source.resolve()
    assert callable(captured["preflight_fn"])
    assert not (dataset_root / "output").exists()
    assert (output_root / "input/output").resolve() == (output_root / "model-output").resolve()
    assert artifact.trace.metadata["model_commit"] == model_descriptors()["exact_gs"].commit
    assert artifact.trace.metadata["model"] == "Exact-GS"
    assert artifact.trace.metadata["dataset"] == "chest"
    assert artifact.reference.volume_path.is_file()
    assert sink.event_count == 1
    assert sink.closed


def test_exact_output_views_isolate_repeated_runs_for_one_dataset(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    (workspace.upstream / "exact_gs").mkdir(parents=True)
    dataset_root = _chest_dataset(tmp_path / "chest")
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    outputs = (workspace.results / "first", workspace.results / "second")

    for output in outputs:
        adapter = get_model_adapter("exact_gs", workspace=workspace, output_root=output)
        run = adapter.prepare(SimpleNamespace(root=dataset_root, id="chest"), config)
        assert adapter._prepare_output(run.dataset_root) == output / "model-output"
        assert (output / "input/output").resolve() == (output / "model-output").resolve()

    assert not (dataset_root / "output").exists()
    assert (outputs[0] / "input/output").resolve() != (outputs[1] / "input/output").resolve()


def test_fact_capture_runs_official_entry_with_split_pipeline_hooks(
    tmp_path: Path, monkeypatch,
) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    source = workspace.upstream / "fact_gs"
    source.mkdir(parents=True)
    (source / "train_recon.py").write_text("", encoding="utf-8")
    dataset_root = _chest_dataset(tmp_path / "chest")
    output_root = workspace.results / "fact_gs" / "chest"
    adapter = get_model_adapter(
        "fact_gs", workspace=workspace, output_root=output_root,
    )
    config = load_config(ROOT / "configs/architecture/gala.yaml")
    run = adapter.prepare(SimpleNamespace(root=dataset_root, id="chest"), config)
    assert "model.eval=true" in run.official_command
    assert "eval.eval_start=true" in run.official_command
    captured: dict[str, object] = {}

    def fake_trace_process(command, **kwargs):
        captured["command"] = tuple(command)
        captured.update(kwargs)
        kwargs["prepare_fn"]()
        trace_root = Path(kwargs["trace_root"])
        builder = TraceBuilder()
        builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE)))
        TraceWriter().write(builder.finish(metadata={}), trace_root)
        volume = output_root / "point_cloud/step_1/vol_pred.npy"
        volume.parent.mkdir(parents=True)
        np.save(volume, np.ones((2, 2, 2), dtype=np.float32))
        return {"status": "passed"}

    class Sink:
        chunk_events = 8

        def __init__(self) -> None:
            self.event_count = 0
            self.closed = False

        def push(self, events, dependencies, payload) -> None:
            self.event_count += len(events)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("gala_sim.adapters.registry.run_trace_process", fake_trace_process)
    sink = Sink()
    artifact = adapter.capture_trace(run, sink)

    command = captured["command"]
    assert "--model-id" in command
    assert command[command.index("--model-id") + 1] == "fact_gs"
    assert "--dataset-id" in command
    assert command[command.index("--dataset-id") + 1] == "chest"
    assert "--capture-iteration-range" in command
    assert command[command.index("--capture-iteration-range") + 1] == "1:1"
    assert "--stop-after-capture-range" in command
    assert "model.init_mode=precomputed" in command
    assert "model.eval=false" in command
    assert "eval.eval_in_training=false" in command
    assert "eval.eval_start=false" in command
    assert "eval.eval_end=false" in command
    assert "model.eval=true" not in command
    assert captured["cwd"] == source.resolve()
    assert callable(captured["preflight_fn"])
    assert artifact.trace.metadata["model_commit"] == model_descriptors()["fact_gs"].commit
    assert artifact.trace.metadata["model"] == "FaCT-GS"
    assert artifact.trace.metadata["dataset"] == "chest"
    assert artifact.reference.volume_path.is_file()
    assert sink.event_count == 1
    assert sink.closed
