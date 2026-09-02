from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import json

from gala_sim.adapters import exact_runtime


def _modules(monkeypatch, calls: list[tuple[bool, int | None]]) -> tuple[ModuleType, ModuleType, ModuleType]:
    package = ModuleType("exact_gs.gaussian")
    initialization = ModuleType("exact_gs.gaussian.initialize")
    scene = ModuleType("exact_gs.dataset")

    def initializer(_gaussians, _args, load_gt=True, loaded_iter=None):
        calls.append((load_gt, loaded_iter))
        return "delegated"

    class Scene:
        def creatVol_gt(self, _query):
            return "created"

        def loadVol_gt(self):
            return "loaded"

    package.initialize_gaussian = initializer
    initialization.initialize_gaussian = initializer
    scene.Scene = Scene
    modules = {
        "exact_gs.gaussian": package,
        "exact_gs.gaussian.initialize": initialization,
        "exact_gs.dataset": scene,
    }
    monkeypatch.setattr(exact_runtime.importlib, "import_module", modules.__getitem__)
    return package, initialization, scene


def _metadata(root: Path, available: bool) -> None:
    root.mkdir()
    (root / "meta_data.json").write_text(json.dumps({
        "reference_volume_available": available,
    }), encoding="utf-8")


def test_exact_overlay_skips_redundant_ground_truth_initialization(tmp_path, monkeypatch) -> None:
    calls: list[tuple[bool, int | None]] = []
    package, _, _ = _modules(monkeypatch, calls)
    _metadata(tmp_path / "dataset", True)
    (tmp_path / "dataset/vol_gt.npy").write_bytes(b"prepared")

    overlay = exact_runtime.install_exact_runtime_overlay()
    result = package.initialize_gaussian(
        object(), SimpleNamespace(source_path=tmp_path / "dataset"), load_gt=True,
    )

    assert result is None
    assert calls == []
    overlay.restore()


def test_exact_overlay_preserves_training_initialization_and_checkpoint_loads(
    tmp_path, monkeypatch,
) -> None:
    calls: list[tuple[bool, int | None]] = []
    package, _, _ = _modules(monkeypatch, calls)
    _metadata(tmp_path / "dataset", True)
    (tmp_path / "dataset/vol_gt.npy").write_bytes(b"prepared")
    overlay = exact_runtime.install_exact_runtime_overlay()
    args = SimpleNamespace(source_path=tmp_path / "dataset")

    assert package.initialize_gaussian(object(), args, load_gt=False) == "delegated"
    assert package.initialize_gaussian(object(), args, loaded_iter=7) == "delegated"
    assert calls == [(False, None), (True, 7)]
    overlay.restore()


def test_exact_overlay_suppresses_reference_only_calls_for_no_ground_truth(
    tmp_path, monkeypatch,
) -> None:
    calls: list[tuple[bool, int | None]] = []
    package, _, scene_module = _modules(monkeypatch, calls)
    _metadata(tmp_path / "dataset", False)
    overlay = exact_runtime.install_exact_runtime_overlay()
    args = SimpleNamespace(source_path=tmp_path / "dataset")
    scene = scene_module.Scene()
    scene.source_path = args.source_path

    assert package.initialize_gaussian(object(), args, load_gt=True) is None
    assert scene.creatVol_gt(object()) is None
    assert scene.loadVol_gt() is None
    assert calls == []
    overlay.restore()


def test_exact_overlay_delegates_and_restores_original_callables(tmp_path, monkeypatch) -> None:
    calls: list[tuple[bool, int | None]] = []
    package, initialization, scene_module = _modules(monkeypatch, calls)
    _metadata(tmp_path / "dataset", True)
    originals = (
        package.initialize_gaussian,
        initialization.initialize_gaussian,
        scene_module.Scene.creatVol_gt,
        scene_module.Scene.loadVol_gt,
    )
    overlay = exact_runtime.install_exact_runtime_overlay()
    args = SimpleNamespace(source_path=tmp_path / "dataset")

    assert package.initialize_gaussian(object(), args, load_gt=True) == "delegated"
    assert calls == [(True, None)]
    overlay.restore()

    assert package.initialize_gaussian is originals[0]
    assert initialization.initialize_gaussian is originals[1]
    assert scene_module.Scene.creatVol_gt is originals[2]
    assert scene_module.Scene.loadVol_gt is originals[3]
