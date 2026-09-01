from __future__ import annotations

from pathlib import Path

import json

import numpy as np

from gala_sim.adapters import get_model_adapter, model_descriptors, prepare_campaign
from gala_sim.config import load_config
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


def test_exact_adapter_uses_the_upstream_method_and_output_contract(tmp_path: Path) -> None:
    workspace = WorkspacePaths.discover(repository=ROOT, workspace=tmp_path / "workspace")
    adapter = get_model_adapter("exact_gs", workspace=workspace)
    command = adapter.build_command(tmp_path / "dataset", tmp_path / "output")
    assert "--method=Exact_GS" in command
    assert any(value.startswith("--source_path=") for value in command)
    assert not any(value.startswith("--model_path=") for value in command)
    assert adapter.descriptor.output_strategy == "dataset_output_directory"


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
