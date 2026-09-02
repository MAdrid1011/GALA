"""Restorable compatibility hooks for official Exact-GS trace capture.

The pinned Exact-GS training entrypoint initializes a Gaussian model from a
publisher-only pickle solely to create ``vol_gt.npy``.  Prepared GALA datasets
already contain that volume, or explicitly declare that no reference volume is
available.  This process-local overlay skips only that redundant initialization
and preserves the official training initialization, rendering, loss, and
optimization paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _reference_state(source_path: str | Path) -> str:
    source = Path(source_path)
    metadata_path = source / "meta_data.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Exact-GS dataset metadata: {metadata_path}") from error
        available = metadata.get("reference_volume_available")
        if available is False:
            return "unavailable"
    if (source / "vol_gt.npy").is_file():
        return "prepared"
    return "upstream_required"


@dataclass
class ExactRuntimeOverlay:
    """Original Exact-GS callables needed to restore the imported modules."""

    gaussian_package: ModuleType
    initialization_module: ModuleType
    scene_module: ModuleType
    original_package_initializer: Callable[..., Any]
    original_module_initializer: Callable[..., Any]
    original_create_volume: Callable[..., Any]
    original_load_volume: Callable[..., Any]

    def restore(self) -> None:
        self.gaussian_package.initialize_gaussian = self.original_package_initializer
        self.initialization_module.initialize_gaussian = self.original_module_initializer
        self.scene_module.Scene.creatVol_gt = self.original_create_volume
        self.scene_module.Scene.loadVol_gt = self.original_load_volume


def install_exact_runtime_overlay() -> ExactRuntimeOverlay:
    """Install a trace-only Exact-GS prepared-dataset compatibility layer."""

    gaussian_package = importlib.import_module("exact_gs.gaussian")
    initialization_module = importlib.import_module("exact_gs.gaussian.initialize")
    scene_module = importlib.import_module("exact_gs.dataset")
    original_package_initializer = gaussian_package.initialize_gaussian
    original_module_initializer = initialization_module.initialize_gaussian
    original_create_volume = scene_module.Scene.creatVol_gt
    original_load_volume = scene_module.Scene.loadVol_gt

    def initialize_gaussian(
        gaussians: Any,
        args: Any,
        load_gt: bool = True,
        loaded_iter: int | None = None,
    ) -> Any:
        if load_gt and loaded_iter is None:
            state = _reference_state(args.source_path)
            if state in {"prepared", "unavailable"}:
                return loaded_iter
        return original_module_initializer(
            gaussians, args, load_gt=load_gt, loaded_iter=loaded_iter,
        )

    def create_volume(scene: Any, query_function: Callable[..., Any]) -> Any:
        if _reference_state(scene.source_path) == "unavailable":
            return None
        return original_create_volume(scene, query_function)

    def load_volume(scene: Any) -> Any:
        if _reference_state(scene.source_path) == "unavailable":
            return None
        return original_load_volume(scene)

    gaussian_package.initialize_gaussian = initialize_gaussian
    initialization_module.initialize_gaussian = initialize_gaussian
    scene_module.Scene.creatVol_gt = create_volume
    scene_module.Scene.loadVol_gt = load_volume
    return ExactRuntimeOverlay(
        gaussian_package,
        initialization_module,
        scene_module,
        original_package_initializer,
        original_module_initializer,
        original_create_volume,
        original_load_volume,
    )
