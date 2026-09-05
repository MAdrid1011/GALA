from __future__ import annotations

import os
from pathlib import Path

import pytest

from gala_sim.tools.r2_gaussian_training_probe import (
    COMPILER_VARIANT_FLAGS,
    OVERLAY_TRANSFORMS,
    TrainingProbeError,
    _compiler_variant_environment,
    _gpu_summary,
    _load_training,
    render_training_overlay,
)


def _source() -> str:
    return "".join(item[1] for item in OVERLAY_TRANSFORMS)


def test_overlay_removes_only_observation_stalls() -> None:
    source = _source() + "loss['total'].backward()\ngaussians.optimizer.step()\n"
    transformed, manifest = render_training_overlay(source)
    assert "torch.cuda.synchronize()" not in transformed
    assert ".item()" not in transformed
    assert "loss['total'].backward()" in transformed
    assert "gaussians.optimizer.step()" in transformed
    assert [item["id"] for item in manifest] == [item[0] for item in OVERLAY_TRANSFORMS]


def test_overlay_rejects_source_drift_without_using_a_hash_gate() -> None:
    source = _source().replace("        iter_start.record()\n", "", 1)
    with pytest.raises(TrainingProbeError, match="expected one source fragment"):
        render_training_overlay(source)


def test_gpu_summary_preserves_raw_samples() -> None:
    samples = [
        {"utilization_percent": 25, "memory_used_mib": 100, "elapsed_seconds": 1.0},
        {"utilization_percent": 75, "memory_used_mib": 200, "elapsed_seconds": 2.0},
    ]
    summary = _gpu_summary(samples)
    assert summary["mean_utilization_percent"] == 50.0
    assert summary["maximum_utilization_percent"] == 75
    assert summary["maximum_memory_used_mib"] == 200
    assert summary["samples"] == samples


@pytest.mark.parametrize(
    ("variant", "query", "semantic"),
    [
        ("gpu_base", "0", "0"),
        ("1000", "1", "0"),
        ("0100", "0", "1"),
        ("1100", "1", "1"),
    ],
)
def test_compiler_variant_environment_sets_independent_mechanisms(
    monkeypatch: pytest.MonkeyPatch, variant: str, query: str, semantic: str,
) -> None:
    monkeypatch.setenv("GALA_QUERY_WARP_REDUCE", "previous-query")
    monkeypatch.setenv("GALA_SEMANTIC_WARP_REDUCE", "previous-semantic")
    with _compiler_variant_environment(variant):
        assert os.environ["GALA_QUERY_WARP_REDUCE"] == query
        assert os.environ["GALA_SEMANTIC_WARP_REDUCE"] == semantic
    assert os.environ["GALA_QUERY_WARP_REDUCE"] == "previous-query"
    assert os.environ["GALA_SEMANTIC_WARP_REDUCE"] == "previous-semantic"


def test_compiler_variants_all_use_gpu_base_comparison() -> None:
    assert set(COMPILER_VARIANT_FLAGS) == {"gpu_base", "1000", "0100", "1100"}


def test_load_training_discovers_built_simple_knn_submodule(tmp_path: Path) -> None:
    source_root = tmp_path / "r2"
    package_root = source_root / "r2_gaussian/submodules/simple-knn/build/lib.linux"
    package = package_root / "simple_knn"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("marker = 'submodule'\n", encoding="utf-8")
    training = source_root / "train.py"
    training.parent.mkdir(parents=True, exist_ok=True)
    training.write_text("import simple_knn\nmarker = simple_knn.marker\n", encoding="utf-8")

    module = _load_training(training, source_root, None)

    assert module.marker == "submodule"
