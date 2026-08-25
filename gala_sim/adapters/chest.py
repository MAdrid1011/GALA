"""Official R²-Gaussian Chest dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectionRecord:
    path: Path
    angle_radians: float
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ChestDatasetManifest:
    root: Path
    metadata: dict[str, Any]
    train: tuple[ProjectionRecord, ...]
    test: tuple[ProjectionRecord, ...]
    volume_path: Path
    initialization_path: Path
    volume_shape: tuple[int, ...]
    detector_shape: tuple[int, ...]

    @property
    def geometry(self) -> dict[str, Any]:
        scanner = self.metadata.get("scanner")
        if not isinstance(scanner, dict):
            raise ValueError("Chest metadata has no scanner object")
        return scanner


def load_chest_manifest(root: Path) -> ChestDatasetManifest:
    root = Path(root).resolve()
    metadata_path = root / "meta_data.json"
    if not root.is_dir() or not metadata_path.is_file():
        raise ValueError("Chest root or meta_data.json is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("scanner"), dict):
        raise ValueError("Chest metadata root/scanner must be objects")
    scanner = metadata["scanner"]
    detector_shape = tuple(int(value) for value in scanner.get("nDetector", ()))
    volume_shape = tuple(int(value) for value in scanner.get("nVoxel", ()))
    if len(detector_shape) != 2 or len(volume_shape) != 3:
        raise ValueError("Chest geometry has invalid detector or volume shape")
    train = _load_projections(root, metadata, "proj_train", detector_shape)
    test = _load_projections(root, metadata, "proj_test", detector_shape)
    volume_path = root / str(metadata.get("vol", "vol_gt.npy"))
    if not volume_path.is_file():
        volume_path = root / "vol_gt.npy"
    if not volume_path.is_file():
        raise ValueError("Chest reference volume is missing")
    volume = np.load(volume_path, mmap_mode="r")
    if tuple(volume.shape) != volume_shape:
        raise ValueError("Chest reference volume shape does not match scanner geometry")
    init_path = root / ("init_" + root.name + ".npy")
    if not init_path.is_file():
        candidates = sorted(root.glob("init_*.npy"))
        if not candidates:
            raise ValueError("Chest initialization array is missing")
        init_path = candidates[0]
    init = np.load(init_path, mmap_mode="r")
    if init.ndim != 2 or init.shape[1] < 4 or init.shape[0] == 0:
        raise ValueError("Chest initialization array has invalid shape")
    return ChestDatasetManifest(root, metadata, train, test, volume_path, init_path,
                                volume_shape, detector_shape)


def _load_projections(root: Path, metadata: dict[str, Any], key: str,
                      detector_shape: tuple[int, ...]) -> tuple[ProjectionRecord, ...]:
    entries = metadata.get(key)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Chest metadata has no {key} entries")
    records: list[ProjectionRecord] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file_path"), str):
            raise ValueError(f"Chest {key} has an invalid projection entry")
        path = root / entry["file_path"]
        if not path.is_file():
            raise ValueError(f"Chest projection is missing: {entry['file_path']}")
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != detector_shape or array.ndim != 2 or array.dtype != np.float32:
            raise ValueError(f"Chest projection has invalid array format: {path}")
        angle = entry.get("angle")
        if not isinstance(angle, (int, float)) or not np.isfinite(angle):
            raise ValueError(f"Chest projection has an invalid angle: {path}")
        records.append(ProjectionRecord(path, float(angle), tuple(array.shape), str(array.dtype)))
    return tuple(records)
