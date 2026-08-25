from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gala_sim.adapters import load_chest_manifest
from gala_sim.adapters.r2_gaussian import R2GaussianChestAdapter, _bind_trace_identity
from gala_sim.config import load_config
from gala_sim.clamp import PrimitiveKind, TraceBuilder, TraceEvent
from gala_sim.identity import sha256_tree
from gala_sim.trace import TraceWriter


def test_chest_manifest_preserves_official_projection_references(tmp_path: Path) -> None:
    (tmp_path / "proj_train").mkdir()
    (tmp_path / "proj_test").mkdir()
    np.save(tmp_path / "proj_train/proj_train_0000.npy", np.ones((2, 2), dtype=np.float32))
    np.save(tmp_path / "proj_test/proj_test_0000.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(tmp_path / "vol_gt.npy", np.zeros((2, 2, 2), dtype=np.float32))
    np.save(tmp_path / "init_fixture.npy", np.zeros((3, 4), dtype=np.float64))
    metadata = {
        "scanner": {"nDetector": [2, 2], "nVoxel": [2, 2, 2]},
        "proj_train": [{"file_path": "proj_train/proj_train_0000.npy", "angle": 0.0}],
        "proj_test": [{"file_path": "proj_test/proj_test_0000.npy", "angle": 1.0}],
        "vol": "vol_gt.npy",
    }
    (tmp_path / "meta_data.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = load_chest_manifest(tmp_path)
    assert len(manifest.train) == 1
    assert manifest.train[0].angle_radians == 0.0
    assert manifest.volume_shape == (2, 2, 2)


def test_r2_adapter_preserves_frozen_python_interpreter(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    adapter = R2GaussianChestAdapter(
        source, tmp_path, tmp_path / "output", "a" * 40, Path("/usr/bin/python3"),
    )
    config = load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml")
    run = adapter.prepare(None, config)
    assert run.official_command[0] == str(Path("/usr/bin/python3").resolve())


def test_trace_identity_is_bound_before_cycle_handoff(tmp_path: Path) -> None:
    builder = TraceBuilder()
    builder.emit(TraceEvent(primitive_kind=int(PrimitiveKind.RELATION_CANDIDATE)))
    trace = builder.finish(metadata={"model": "R²-Gaussian", "dataset": "Chest"})
    source = tmp_path / "source"
    source.mkdir()
    run_dataset = tmp_path / "dataset"
    run_dataset.mkdir()
    adapter = R2GaussianChestAdapter(
        source, run_dataset, tmp_path / "output", "b" * 40, Path("/usr/bin/python3"),
    )
    run = adapter.prepare(None, load_config(Path(__file__).parents[1] / "configs/architecture/gala.yaml"))
    bound = _bind_trace_identity(trace, run, adapter.model_commit, "c" * 40)
    root = tmp_path / "trace"
    TraceWriter().write(bound, root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config_sha256"] == run.config_sha256
    assert metadata["model_commit"] == "b" * 40
    assert metadata["dataset_manifest_sha256"] == sha256_tree(run_dataset)
    assert metadata["repository_commit"] == "c" * 40
