from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gala_sim.adapters.protocol import PreparedRun
from gala_sim.adapters.r2_gaussian import _read_metrics
from gala_sim.config import GalaConfig, load_config
from gala_sim.metrics import QualityConfig, QualityMetrics
from gala_sim.metrics.quality import _check_volumes, _normalize_lpips_slice


def _parameter(value, unit: str = "scalar") -> dict[str, object]:
    return {
        "value": value,
        "unit": unit,
        "source": "fixture",
        "scope": "quality",
        "status": "frozen",
    }


def test_quality_config_loads_frozen_chest_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/architecture/gala.yaml")
    quality = QualityConfig.from_gala(config)
    assert quality.data_min == 0.0
    assert quality.data_max == 1.0
    assert quality.ssim_window == 11
    assert quality.ssim_sigma == 1.5
    assert quality.ssim_boundary == "reflect"
    assert quality.lpips_slices == ((42, 85, 128, 170, 213),) * 3
    assert quality.lpips_network == "alex"
    assert quality.lpips_version == "0.1"
    assert quality.lpips_backbone_sha256.startswith("7be5be79")
    assert quality.lpips_calibration_sha256.startswith("df73285e")
    assert not config.ready


def test_quality_config_loads_frozen_orthogonal_slice_lists() -> None:
    config = GalaConfig(
        Path("quality.yaml"),
        {"quality": {
            "data_min": _parameter(0.0),
            "data_max": _parameter(1.0),
            "ssim_window": _parameter(11, "voxel"),
            "ssim_sigma": _parameter(1.5, "voxel"),
            "ssim_boundary": _parameter("reflect", "mode"),
            "lpips_slices": _parameter(((4, 8), (5,), (6, 9)), "index_list_per_axis"),
            "lpips_network": _parameter("alex", "model_name"),
            "lpips_version": _parameter("fixture-version", "version"),
            "lpips_backbone_sha256": _parameter("a" * 64, "SHA-256"),
            "lpips_calibration_sha256": _parameter("b" * 64, "SHA-256"),
        }},
        "a" * 64,
        True,
    )
    quality = QualityConfig.from_gala(config)
    assert quality.lpips_slices == ((4, 8), (5,), (6, 9))
    assert quality.lpips_network == "alex"
    assert quality.lpips_version == "fixture-version"


def test_quality_config_rejects_implicit_boundary_and_unordered_slices() -> None:
    values = {
        "data_min": 0.0,
        "data_max": 1.0,
        "ssim_window": 11,
        "ssim_sigma": 1.5,
        "ssim_boundary": "constant",
        "lpips_slices": ((4, 8), (5,), (6, 9)),
        "lpips_network": "alex",
        "lpips_version": "0.1",
        "lpips_backbone_sha256": "a" * 64,
        "lpips_calibration_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="SSIM boundary"):
        QualityConfig(**values)
    values["ssim_boundary"] = "reflect"
    values["lpips_slices"] = ((8, 4), (5,), (6, 9))
    with pytest.raises(ValueError, match="strictly increasing"):
        QualityConfig(**values)


def test_lpips_slice_normalization_clips_to_registered_data_range() -> None:
    config = QualityConfig(
        0.0, 1.0, 3, 1.0, "reflect", ((0,), (0,), (0,)), "alex", "0.1",
        "a" * 64, "b" * 64,
    )
    volume = np.asarray([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    normalized = _normalize_lpips_slice(volume, axis=0, index=0, config=config)
    assert normalized.dtype == np.float32
    assert normalized.tolist() == [[-1.0, 0.0, 1.0]]

    candidate = np.zeros((2, 2, 2), dtype=np.float32)
    reference = np.full((2, 2, 2), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="reference range"):
        _check_volumes(reference, candidate, config)


def test_r2_reference_metrics_include_unified_and_official_values(
    tmp_path: Path, monkeypatch
) -> None:
    reference_path = tmp_path / "vol_gt.npy"
    candidate_path = tmp_path / "vol_pred.npy"
    np.save(reference_path, np.zeros((2, 2, 2), dtype=np.float32))
    np.save(candidate_path, np.ones((2, 2, 2), dtype=np.float32))
    quality_config = QualityConfig(
        0.0, 1.0, 3, 1.0, "reflect", ((0,), (0,), (0,)), "alex", "0.1",
        "a" * 64, "b" * 64,
    )
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
