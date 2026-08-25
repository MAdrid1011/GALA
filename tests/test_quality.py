from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gala_sim.adapters.protocol import PreparedRun
from gala_sim.adapters.r2_gaussian import _read_metrics
from gala_sim.config import GalaConfig, load_config
from gala_sim.metrics import QualityConfig, QualityMetrics


def _parameter(value, unit: str = "scalar") -> dict[str, object]:
    return {
        "value": value,
        "unit": unit,
        "source": "fixture",
        "scope": "quality",
        "status": "frozen",
    }


def test_quality_config_rejects_pending_registered_values() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/architecture/gala.yaml")
    with pytest.raises(ValueError, match="quality configuration is not frozen"):
        QualityConfig.from_gala(config)


def test_quality_config_loads_frozen_orthogonal_slice_lists() -> None:
    config = GalaConfig(
        Path("quality.yaml"),
        {"quality": {
            "data_min": _parameter(0.0),
            "data_max": _parameter(1.0),
            "ssim_window": _parameter(11, "voxel"),
            "ssim_sigma": _parameter(1.5, "voxel"),
            "lpips_slices": _parameter(((4, 8), (5,), (6, 9)), "index_list_per_axis"),
            "lpips_network": _parameter("fixture-net", "model_name"),
        }},
        "a" * 64,
        True,
    )
    quality = QualityConfig.from_gala(config)
    assert quality.lpips_slices == ((4, 8), (5,), (6, 9))
    assert quality.lpips_network == "fixture-net"


def test_r2_reference_metrics_include_unified_and_official_values(
    tmp_path: Path, monkeypatch
) -> None:
    reference_path = tmp_path / "vol_gt.npy"
    candidate_path = tmp_path / "vol_pred.npy"
    np.save(reference_path, np.zeros((2, 2, 2), dtype=np.float32))
    np.save(candidate_path, np.ones((2, 2, 2), dtype=np.float32))
    quality_config = QualityConfig(0.0, 1.0, 3, 1.0, ((0,), (0,), (0,)), "fixture")
    run = PreparedRun(
        "R2-Gaussian", "Chest", tmp_path, tmp_path, "a" * 64,
        quality_config, 0, ("python", "train.py"),
    )
    monkeypatch.setattr(
        "gala_sim.adapters.r2_gaussian.load_chest_manifest",
        lambda _root: SimpleNamespace(volume_path=reference_path),
    )
    monkeypatch.setattr(
        "gala_sim.adapters.r2_gaussian.measure_quality",
        lambda _reference, _candidate, _config: QualityMetrics(20.0, 0.8, 0.1),
    )
    monkeypatch.setattr(
        "gala_sim.adapters.r2_gaussian._read_latest_metrics",
        lambda _root: {"psnr_3d": 19.0, "ssim_3d": 0.7},
    )
    assert _read_metrics(run, candidate_path, tmp_path) == {
        "psnr": 20.0,
        "ssim": 0.8,
        "lpips": 0.1,
        "psnr_3d": 19.0,
        "ssim_3d": 0.7,
    }
