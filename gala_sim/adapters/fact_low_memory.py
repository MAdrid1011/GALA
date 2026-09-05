"""Memory-bounded projection hooks for official upstream trace capture.

The supported upstream loaders transfer every projection to CUDA while
constructing a scene.  High-resolution cone-beam data therefore exhausts a
commodity GPU before the first training step.  This module provides opt-in,
process-local overlays that preserve camera geometry and selected view order,
while holding each projection as a path until its camera is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import mmap
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class LazyProjection:
    """One scaled projection retained as source metadata rather than an array."""

    path: Path
    scale: float
    shape: tuple[int, int]

    def load(self) -> np.ndarray:
        values = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if tuple(values.shape) != self.shape:
            raise ValueError(
                f"projection shape changed after preparation: {self.path}"
            )
        # This matches ``np.load(path) * scene_scale`` in the upstream reader
        # while keeping the allocation limited to the selected training view.
        result = np.ascontiguousarray(
            np.asarray(values, dtype=np.float32) * self.scale
        )
        # The result owns its storage.  Advise the kernel that the read-only
        # mapping can be reclaimed before the next view is loaded; this keeps
        # multi-iteration probes from accumulating projection page cache.
        mapped = getattr(values, "_mmap", None)
        advice = getattr(mmap, "MADV_DONTNEED", None)
        if mapped is not None and advice is not None and hasattr(mapped, "madvise"):
            try:
                mapped.madvise(advice)
            except (OSError, ValueError):
                pass
        del values
        return result


def _lazy_read_cameras(
    meta_data: dict[str, Any], source_path: str | Path, eval: bool = False,
    scene_scale: float = 1.0,
    *,
    camera_info_type: type[Any], angle_to_pose: Callable[[float, float], np.ndarray],
    mode_ids: dict[str, int],
) -> dict[str, list[Any]]:
    """Reproduce FaCT's camera metadata reader without eager projection IO."""

    source = Path(source_path)
    scanner = meta_data["scanner"]
    splits = ("train", "test") if eval else ("train",)
    result: dict[str, list[Any]] = {"train": [], "test": []}
    for split in splits:
        frames = meta_data[f"proj_{split}"]
        uid_offset = len(meta_data["proj_train"]) if split == "test" else 0
        for index, frame in enumerate(frames):
            angle = float(frame["angle"])
            world_to_camera = np.linalg.inv(angle_to_pose(float(scanner["DSO"]), angle))
            rotation = np.transpose(world_to_camera[:3, :3])
            translation = world_to_camera[:3, 3]
            image_path = source / str(frame["file_path"])
            # Reading only the .npy header preserves upstream dimensions without
            # materialising image payloads in host memory.
            image_header = np.load(image_path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(value) for value in image_header.shape)
            del image_header
            if shape != (int(scanner["nDetector"][0]), int(scanner["nDetector"][1])):
                raise ValueError(f"projection shape does not match scanner metadata: {image_path}")
            fov_x = np.arctan2(float(scanner["sDetector"][1]) / 2, float(scanner["DSD"])) * 2
            fov_y = np.arctan2(float(scanner["sDetector"][0]) / 2, float(scanner["DSD"])) * 2
            result[split].append(camera_info_type(
                uid=index + uid_offset,
                R=rotation,
                T=translation,
                angle=angle,
                FovY=fov_y,
                FovX=fov_x,
                image=LazyProjection(image_path, float(scene_scale), shape),
                image_path=str(image_path),
                image_name=image_path.stem,
                width=shape[1],
                height=shape[0],
                mode=mode_ids[str(scanner["mode"])],
                scanner_cfg=scanner,
            ))
    return result


def _build_lazy_camera(
    camera_base: type[Any], graphics: ModuleType, torch_module: Any,
    *,
    colmap_id: int, scanner_cfg: dict[str, Any], R: np.ndarray, T: np.ndarray,
    angle: float, mode: int, FoVx: float, FoVy: float, projection: LazyProjection,
    image_name: str, uid: int,
) -> Any:
    """Construct an upstream-compatible camera whose image property loads on use."""

    class LazyCamera(camera_base):
        @property
        def original_image(self) -> Any:
            return torch_module.from_numpy(projection.load())[None]

    camera = LazyCamera.__new__(LazyCamera)
    torch_module.nn.Module.__init__(camera)
    camera.uid = uid
    camera.colmap_id = colmap_id
    camera.scanner_cfg = scanner_cfg
    camera.R = R
    camera.T = T
    camera.angle = angle
    camera.FoVx = FoVx
    camera.FoVy = FoVy
    camera.mode = mode
    camera.image_name = image_name
    camera.data_device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    camera.image_width = projection.shape[1]
    camera.image_height = projection.shape[0]
    camera.trans = np.array([0.0, 0.0, 0.0])
    camera.scale = 1.0
    camera.world_view_transform = torch_module.tensor(
        graphics.getWorld2View2(R, T, camera.trans, camera.scale)
    ).transpose(0, 1).cuda()
    camera.projection_matrix = graphics.getProjectionMatrix(
        fovX=FoVx, fovY=FoVy, mode=mode, scanner_cfg=scanner_cfg,
    ).transpose(0, 1).cuda()
    camera.full_proj_transform = camera.world_view_transform.unsqueeze(0).bmm(
        camera.projection_matrix.unsqueeze(0)
    ).squeeze(0)
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    return camera


@dataclass
class FactLowMemoryOverlay:
    """Restorable projection-loading patches for one official child process."""

    reader_module: ModuleType
    camera_utils: ModuleType
    original_reader: Callable[..., Any]
    original_loader: Callable[..., Any]

    def restore(self) -> None:
        self.reader_module.readCTameras = self.original_reader
        self.camera_utils.loadCam = self.original_loader


def _install_low_memory_overlay(
    *,
    reader_name: str,
    camera_utils_name: str,
    camera_module_name: str,
    graphics_name: str,
) -> FactLowMemoryOverlay:
    reader = importlib.import_module(reader_name)
    camera_utils = importlib.import_module(camera_utils_name)
    camera_module = importlib.import_module(camera_module_name)
    graphics = importlib.import_module(graphics_name)
    torch_module = importlib.import_module("torch")
    original_reader = reader.readCTameras
    original_loader = camera_utils.loadCam

    def lazy_reader(meta_data: dict[str, Any], source_path: str | Path,
                    eval: bool = False, scene_scale: float = 1.0) -> dict[str, list[Any]]:
        return _lazy_read_cameras(
            meta_data, source_path, eval, scene_scale,
            camera_info_type=reader.CameraInfo,
            angle_to_pose=reader.angle2pose,
            mode_ids=reader.mode_id,
        )

    def lazy_loader(args: Any, identifier: int, cam_info: Any) -> Any:
        image = cam_info.image
        if not isinstance(image, LazyProjection):
            return original_loader(args, identifier, cam_info)
        return _build_lazy_camera(
            camera_module.Camera, graphics, torch_module,
            colmap_id=cam_info.uid, scanner_cfg=cam_info.scanner_cfg,
            R=cam_info.R, T=cam_info.T, angle=cam_info.angle, mode=cam_info.mode,
            FoVx=cam_info.FovX, FoVy=cam_info.FovY, projection=image,
            image_name=cam_info.image_name, uid=identifier,
        )

    reader.readCTameras = lazy_reader
    camera_utils.loadCam = lazy_loader
    return FactLowMemoryOverlay(reader, camera_utils, original_reader, original_loader)


def install_fact_low_memory_overlay() -> FactLowMemoryOverlay:
    """Install lazy camera loading before the official FaCT-GS script is run."""

    return _install_low_memory_overlay(
        reader_name="fact_gs.r2_gaussian.dataset.dataset_readers",
        camera_utils_name="fact_gs.r2_gaussian.utils.camera_utils",
        camera_module_name="fact_gs.r2_gaussian.dataset.cameras",
        graphics_name="fact_gs.r2_gaussian.utils.graphics_utils",
    )


def install_exact_low_memory_overlay() -> FactLowMemoryOverlay:
    """Install the same demand-loaded projection contract for Exact-GS."""

    return _install_low_memory_overlay(
        reader_name="exact_gs.dataset.dataset_readers",
        camera_utils_name="exact_gs.utils.camera_utils",
        camera_module_name="exact_gs.dataset.cameras",
        graphics_name="exact_gs.utils.graphics_utils",
    )


def install_r2_low_memory_overlay() -> FactLowMemoryOverlay:
    """Install demand-loaded projections for the official R²-Gaussian path."""

    return _install_low_memory_overlay(
        reader_name="r2_gaussian.dataset.dataset_readers",
        camera_utils_name="r2_gaussian.utils.camera_utils",
        camera_module_name="r2_gaussian.dataset.cameras",
        graphics_name="r2_gaussian.utils.graphics_utils",
    )
