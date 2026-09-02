from __future__ import annotations

from typing import NamedTuple

import numpy as np

from gala_sim.adapters.fact_low_memory import (
    LazyProjection,
    _lazy_read_cameras,
    install_exact_low_memory_overlay,
    install_r2_low_memory_overlay,
)


class _CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    angle: float
    FovY: float
    FovX: float
    image: object
    image_path: str
    image_name: str
    width: int
    height: int
    mode: int
    scanner_cfg: dict[str, object]


def _angle_to_pose(dso: float, angle: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = dso + angle
    return pose


def test_lazy_projection_loads_only_when_the_camera_image_is_accessed(tmp_path) -> None:
    path = tmp_path / "projection.npy"
    np.save(path, np.asarray([[1.0, 2.0]], dtype=np.float32))
    projection = LazyProjection(path, 0.25, (1, 2))

    values = projection.load()

    assert np.allclose(values, [[0.25, 0.5]])
    assert values.flags.c_contiguous


def test_lazy_camera_reader_retains_projection_records_and_geometry(tmp_path) -> None:
    np.save(tmp_path / "view.npy", np.ones((2, 3), dtype=np.float32))
    metadata = {
        "scanner": {
            "DSO": 10.0, "DSD": 20.0, "sDetector": [4.0, 6.0],
            "nDetector": [2, 3], "mode": "cone",
        },
        "proj_train": [{"file_path": "view.npy", "angle": 0.3}],
        "proj_test": [],
    }

    records = _lazy_read_cameras(
        metadata, tmp_path, False, 0.5,
        camera_info_type=_CameraInfo, angle_to_pose=_angle_to_pose,
        mode_ids={"cone": 1},
    )

    assert len(records["train"]) == 1
    record = records["train"][0]
    assert isinstance(record.image, LazyProjection)
    assert record.image.path == tmp_path / "view.npy"
    assert record.image.shape == (2, 3)
    assert record.image.scale == 0.5
    assert record.width == 3
    assert record.height == 2
    assert record.mode == 1


def test_exact_low_memory_overlay_uses_exact_module_boundaries(monkeypatch) -> None:
    requested: list[str] = []

    class Reader:
        readCTameras = staticmethod(lambda *_args, **_kwargs: "reader")
        CameraInfo = _CameraInfo
        angle2pose = staticmethod(_angle_to_pose)
        mode_id = {"cone": 1}

    class CameraUtils:
        loadCam = staticmethod(lambda *_args, **_kwargs: "loader")

    class Camera:
        pass

    class CameraModule:
        pass

    CameraModule.Camera = Camera
    modules = {
        "exact_gs.dataset.dataset_readers": Reader,
        "exact_gs.utils.camera_utils": CameraUtils,
        "exact_gs.dataset.cameras": CameraModule,
        "exact_gs.utils.graphics_utils": object(),
        "torch": object(),
    }

    def load(name: str):
        requested.append(name)
        return modules[name]

    monkeypatch.setattr("gala_sim.adapters.fact_low_memory.importlib.import_module", load)
    original_reader = Reader.readCTameras
    original_loader = CameraUtils.loadCam

    overlay = install_exact_low_memory_overlay()

    assert requested[:4] == [
        "exact_gs.dataset.dataset_readers",
        "exact_gs.utils.camera_utils",
        "exact_gs.dataset.cameras",
        "exact_gs.utils.graphics_utils",
    ]
    assert Reader.readCTameras is not original_reader
    assert CameraUtils.loadCam is not original_loader
    overlay.restore()
    assert Reader.readCTameras is original_reader
    assert CameraUtils.loadCam is original_loader


def test_r2_low_memory_overlay_uses_r2_module_boundaries(monkeypatch) -> None:
    requested: list[str] = []

    class Reader:
        readCTameras = staticmethod(lambda *_args, **_kwargs: "reader")
        CameraInfo = _CameraInfo
        angle2pose = staticmethod(_angle_to_pose)
        mode_id = {"cone": 1}

    class CameraUtils:
        loadCam = staticmethod(lambda *_args, **_kwargs: "loader")

    class CameraModule:
        pass

    CameraModule.Camera = type("Camera", (), {})
    modules = {
        "r2_gaussian.dataset.dataset_readers": Reader,
        "r2_gaussian.utils.camera_utils": CameraUtils,
        "r2_gaussian.dataset.cameras": CameraModule,
        "r2_gaussian.utils.graphics_utils": object(),
        "torch": object(),
    }

    def load(name: str):
        requested.append(name)
        return modules[name]

    monkeypatch.setattr("gala_sim.adapters.fact_low_memory.importlib.import_module", load)
    overlay = install_r2_low_memory_overlay()

    assert requested[:4] == [
        "r2_gaussian.dataset.dataset_readers",
        "r2_gaussian.utils.camera_utils",
        "r2_gaussian.dataset.cameras",
        "r2_gaussian.utils.graphics_utils",
    ]
    overlay.restore()
