"""Common dataset contract and public tomography dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Protocol

import numpy as np

from .chest import load_chest_manifest


@dataclass(frozen=True)
class DatasetDescriptor:
    id: str
    display_name: str
    intensity_transform: str


@dataclass(frozen=True)
class ScannerGeometry:
    detector_shape: tuple[int, int]
    volume_shape: tuple[int, int, int]
    source_object_distance: float
    source_detector_distance: float


@dataclass(frozen=True)
class DatasetProjection:
    path: Path
    angle_radians: float
    shape: tuple[int, int]
    dtype: str
    frame_index: int | None = None


@dataclass(frozen=True)
class DatasetManifest:
    id: str
    root: Path
    projections: tuple[DatasetProjection, ...]
    geometry: ScannerGeometry
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    reference_volume: Path | None
    intensity_transform: str
    metadata: Mapping[str, Any]
    initialization: Path | None = None


class DatasetAdapter(Protocol):
    descriptor: DatasetDescriptor

    def load(self, root: Path) -> DatasetManifest: ...
    def validate(self, manifest: DatasetManifest) -> None: ...
    def convert(self, manifest: DatasetManifest, output: Path) -> DatasetManifest: ...


class _DatasetAdapterBase:
    descriptor: DatasetDescriptor

    def validate(self, manifest: DatasetManifest) -> None:
        if not manifest.projections:
            raise ValueError("dataset contains no projections")
        geometry = manifest.geometry
        if any(value <= 0 for value in (*geometry.detector_shape, *geometry.volume_shape)):
            raise ValueError("dataset geometry dimensions must be positive")
        if not 0 < geometry.source_object_distance < geometry.source_detector_distance:
            raise ValueError("dataset requires 0 < DSO < DSD")
        for projection in manifest.projections:
            if projection.shape != geometry.detector_shape:
                raise ValueError(f"projection shape mismatch: {projection.path}")
            if not math.isfinite(projection.angle_radians):
                raise ValueError(f"projection angle is not finite: {projection.path}")
        indices = set(manifest.train_indices) | set(manifest.test_indices)
        if indices != set(range(len(manifest.projections))):
            raise ValueError("train/test partitions must cover every projection")
        if manifest.reference_volume is not None:
            reference_shape = _reference_volume_shape(manifest.reference_volume)
            if reference_shape != geometry.volume_shape:
                raise ValueError("reference volume shape does not match scanner geometry")

    def convert(self, manifest: DatasetManifest, output: Path) -> DatasetManifest:
        self.validate(manifest)
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("dataset conversion output must be empty")
        converted: list[DatasetProjection] = []
        for index, projection in enumerate(manifest.projections):
            array = _apply_intensity_transform(_read_array(
                projection.path, frame_index=projection.frame_index,
            ), manifest.intensity_transform)
            if not np.isfinite(array).all():
                raise ValueError(f"projection contains non-finite values: {projection.path}")
            target = _projection_target(manifest, output, index)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, array, allow_pickle=False)
            converted.append(DatasetProjection(
                target, projection.angle_radians, tuple(array.shape), str(array.dtype),
            ))
        reference = _copy_reference_volume(manifest.reference_volume, output)
        initialization = _prepare_initialization(manifest, output, reference)
        staged = DatasetManifest(
            manifest.id, output, tuple(converted), manifest.geometry,
            manifest.train_indices, manifest.test_indices, reference,
            manifest.intensity_transform, manifest.metadata, initialization,
        )
        return _write_prepared_metadata(staged)


class ChestDatasetAdapter(_DatasetAdapterBase):
    descriptor = DatasetDescriptor("chest", "Chest", "identity")

    def load(self, root: Path) -> DatasetManifest:
        root = Path(root)
        if not (root / "meta_data.json").is_file():
            candidates = sorted(root.rglob("meta_data.json")) if root.is_dir() else []
            if len(candidates) != 1:
                raise ValueError("Chest asset must contain exactly one dataset root")
            root = candidates[0].parent
        chest = load_chest_manifest(root)
        scanner = chest.geometry
        dso = float(scanner.get("DSO", 1.0))
        dsd = float(scanner.get("DSD", max(2.0, dso + 1.0)))
        records = (*chest.train, *chest.test)
        projections = tuple(DatasetProjection(
            item.path, item.angle_radians, item.shape, item.dtype,
        ) for item in records)
        manifest = DatasetManifest(
            "chest", chest.root, projections,
            ScannerGeometry(chest.detector_shape, chest.volume_shape, dso, dsd),
            tuple(range(len(chest.train))),
            tuple(range(len(chest.train), len(records))),
            chest.volume_path, "identity", chest.metadata,
            next(iter(sorted(chest.root.glob("init_*.npy"))), None),
        )
        self.validate(manifest)
        return manifest


class WalnutDatasetAdapter(_DatasetAdapterBase):
    descriptor = DatasetDescriptor("walnut", "FIPS Walnut", "negative_log")

    def load(self, root: Path) -> DatasetManifest:
        return _load_exported_dataset(
            Path(root), self.descriptor, default_center_shift=-5,
            reference_patterns=("*reference*.npy", "*recon*.npy", "*volume*.npy"),
        )

    def convert(self, manifest: DatasetManifest, output: Path) -> DatasetManifest:
        self.validate(manifest)
        shift = int(manifest.metadata.get("center_shift_pixels", -5))
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("dataset conversion output must be empty")
        shifted = []
        for index, projection in enumerate(manifest.projections):
            array = _apply_intensity_transform(np.roll(
                _read_array(projection.path, frame_index=projection.frame_index),
                shift, axis=-1,
            ), manifest.intensity_transform)
            if not np.isfinite(array).all():
                raise ValueError(f"projection contains non-finite values: {projection.path}")
            target = _projection_target(manifest, output, index)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, array, allow_pickle=False)
            shifted.append(DatasetProjection(target, projection.angle_radians,
                                               tuple(array.shape), str(array.dtype)))
        reference = _copy_reference_volume(manifest.reference_volume, output)
        initialization = _prepare_initialization(manifest, output, reference)
        staged = DatasetManifest(
            manifest.id, output, tuple(shifted), manifest.geometry,
            manifest.train_indices, manifest.test_indices,
            reference, manifest.intensity_transform,
            {**manifest.metadata, "center_shift_pixels": shift},
            initialization,
        )
        return _write_prepared_metadata(staged)


class HDTomoUSBDatasetAdapter(_DatasetAdapterBase):
    descriptor = DatasetDescriptor("hdtomo_usb", "HDTomo-USB", "negative_log")

    def load(self, root: Path) -> DatasetManifest:
        return _load_exported_dataset(
            Path(root), self.descriptor, default_center_shift=0,
            reference_patterns=(
                "recon/**/*.npy", "recon/**/*.tif", "recon/**/*.tiff",
                "recon/**/*.txm", "*.txm",
            ),
        )


def _write_prepared_metadata(manifest: DatasetManifest) -> DatasetManifest:
    document = {
        "schema_version": "gala-prepared-dataset-v1", "id": manifest.id,
        "angles_radians": [item.angle_radians for item in manifest.projections],
        "detector_shape": list(manifest.geometry.detector_shape),
        "volume_shape": list(manifest.geometry.volume_shape),
        "DSO": manifest.geometry.source_object_distance,
        "DSD": manifest.geometry.source_detector_distance,
        "train_indices": list(manifest.train_indices),
        "test_indices": list(manifest.test_indices),
        "intensity_transform": manifest.intensity_transform,
        "projection_files": [
            item.path.relative_to(manifest.root).as_posix()
            for item in manifest.projections
        ],
        "reference_volume": (
            manifest.reference_volume.relative_to(manifest.root).as_posix()
            if manifest.reference_volume is not None else None
        ),
        "initialization": (
            manifest.initialization.relative_to(manifest.root).as_posix()
            if manifest.initialization is not None else None
        ),
        **_portable_geometry_metadata(manifest.metadata, manifest.geometry),
    }
    (manifest.root / "metadata.json").write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    _write_r2_metadata(manifest)
    return DatasetManifest(
        manifest.id, manifest.root, manifest.projections, manifest.geometry,
        manifest.train_indices, manifest.test_indices, manifest.reference_volume,
        manifest.intensity_transform, document, manifest.initialization,
    )


def _load_exported_dataset(
    root: Path, descriptor: DatasetDescriptor, *, default_center_shift: int,
    reference_patterns: tuple[str, ...],
) -> DatasetManifest:
    root = root.resolve()
    metadata = _load_metadata(root)
    projection_paths = _projection_paths(root, metadata)
    if not projection_paths:
        raise ValueError(f"{descriptor.display_name} contains no projection arrays")
    txrm_info = (
        _xradia_info(projection_paths[0])
        if projection_paths[0].suffix.lower() == ".txrm" else None
    )
    first = (
        _read_array(projection_paths[0], frame_index=0)
        if txrm_info is not None else _read_array(projection_paths[0], mmap=True)
    )
    if first.ndim != 2:
        raise ValueError("projection arrays must be two-dimensional")
    detector_shape = tuple(int(value) for value in metadata.get("detector_shape", first.shape))
    reference_value = metadata.get("reference_volume")
    reference = (
        root / str(reference_value) if isinstance(reference_value, str)
        else _find_reference_volume(root, reference_patterns)
    )
    default_volume_shape = _reference_volume_shape(reference) if reference is not None else ()
    volume_shape = tuple(int(value) for value in metadata.get(
        "volume_shape", default_volume_shape,
    ))
    if len(detector_shape) != 2 or len(volume_shape) != 3:
        raise ValueError("dataset metadata lacks detector or volume shape")
    if (
        "voxel_size" not in metadata
        and "detector_pixel_size" in metadata
        and "DSO" in metadata
        and "DSD" in metadata
    ):
        object_pixel = (
            float(metadata["detector_pixel_size"][0])
            * float(metadata["DSO"])
            / float(metadata["DSD"])
        )
        object_extent = min(detector_shape) * object_pixel
        metadata["voxel_size"] = [
            object_extent / dimension for dimension in volume_shape
        ]
    projections: list[DatasetProjection] = []
    if txrm_info is not None:
        if len(projection_paths) != 1:
            raise ValueError("a projection directory must contain one TXRM stack")
        count, image_shape, dtype, embedded_angles = txrm_info
        angles = _angles(metadata, count, fallback=embedded_angles)
        projections.extend(
            DatasetProjection(projection_paths[0], angle, image_shape, dtype, index)
            for index, angle in enumerate(angles)
        )
    else:
        angles = _angles(metadata, len(projection_paths))
        for path, angle in zip(projection_paths, angles, strict=True):
            array = _read_array(path, mmap=True)
            projections.append(DatasetProjection(path, angle, tuple(array.shape), str(array.dtype)))
    expected_count = metadata.get("projection_count")
    if expected_count is not None and int(expected_count) != len(projections):
        raise ValueError("projection count does not match publisher metadata")
    initialization_value = metadata.get("initialization")
    initialization = (
        root / str(initialization_value) if isinstance(initialization_value, str) else None
    )
    if isinstance(metadata.get("train_indices"), list) and isinstance(
        metadata.get("test_indices"), list,
    ):
        train_indices = tuple(int(value) for value in metadata["train_indices"])
        test_indices = tuple(int(value) for value in metadata["test_indices"])
    else:
        test_stride = int(metadata.get("test_view_stride", 0))
        if test_stride > 0:
            test_indices = tuple(range(0, len(projections), test_stride))
            test_set = set(test_indices)
            train_indices = tuple(
                index for index in range(len(projections)) if index not in test_set
            )
        else:
            split = min(int(metadata.get("train_count", max(1, len(projections) - 1))),
                        len(projections))
            train_indices = tuple(range(split))
            test_indices = tuple(range(split, len(projections)))
    manifest = DatasetManifest(
        descriptor.id, root, tuple(projections), ScannerGeometry(
            detector_shape, volume_shape,
            float(_metadata_value(metadata, "DSO", "source_object_distance")),
            float(_metadata_value(metadata, "DSD", "source_detector_distance")),
        ), train_indices, test_indices, reference,
        descriptor.intensity_transform,
        {**metadata, "center_shift_pixels": int(metadata.get(
            "center_shift_pixels", default_center_shift,
        ))}, initialization,
    )
    _DatasetAdapterBase().validate(manifest)
    return manifest


def _load_metadata(root: Path) -> dict[str, Any]:
    for name in ("metadata.json", "scan.json", "geometry.json"):
        path = root / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("dataset metadata must be an object")
            return data
    text_path = next(iter(sorted(root.rglob("*.txt"))), None)
    if text_path is None:
        recipe_path = next(iter(sorted(root.rglob("*.rcp"))), None)
        if recipe_path is None:
            raise ValueError("dataset metadata file is missing")
        return _load_xradia_recipe(recipe_path)
    text = text_path.read_text(encoding="utf-8", errors="replace")
    data: dict[str, Any] = {}
    aliases = {
        "DSO": ("DSO", "source object distance", "source-to-object distance"),
        "DSD": ("DSD", "source detector distance", "source-to-detector distance"),
    }
    for key, names in aliases.items():
        for name in names:
            match = re.search(rf"{re.escape(name)}\s*[:=]\s*([-+0-9.eE]+)", text, re.I)
            if match:
                data[key] = float(match.group(1))
                break
    for key, length in (("detector_shape", 2), ("volume_shape", 3)):
        match = re.search(rf"{key}\s*[:=]\s*([0-9 x,]+)", text, re.I)
        if match:
            values = [int(value) for value in re.findall(r"\d+", match.group(1))]
            if len(values) == length:
                data[key] = values
    publisher_fields = {
        "DSD": ("DistanceSourceDetector", float),
        "DSO": ("DistanceSourceOrigin", float),
        "projection_count": ("NumberImages", int),
        "angle_first_degrees": ("AngleFirst", float),
        "angle_interval_degrees": ("AngleInterval", float),
        "detector_pixel_size_scalar": ("PixelSize", float),
        "mode": ("GeometryType", str),
    }
    for key, (name, parser) in publisher_fields.items():
        match = re.search(rf"^{re.escape(name)}\s*=\s*([^\r\n]+)", text, re.I | re.M)
        if match:
            value = match.group(1).strip()
            data[key] = parser(value)
    if "detector_pixel_size_scalar" in data:
        pixel_size = float(data.pop("detector_pixel_size_scalar"))
        data["detector_pixel_size"] = [pixel_size, pixel_size]
    if all(key in data for key in ("DSO", "DSD", "detector_pixel_size")):
        data.setdefault("volume_shape", [512, 512, 512])
        data.setdefault("test_view_stride", 8)
        detector_shape = data.get("detector_shape")
        if detector_shape is not None:
            object_pixel = float(data["detector_pixel_size"][0]) * data["DSO"] / data["DSD"]
            extent = min(int(value) for value in detector_shape) * object_pixel
            data["voxel_size"] = [extent / 512.0] * 3
    return data


def _load_xradia_recipe(path: Path) -> dict[str, Any]:
    try:
        import olefile
    except ImportError as error:
        raise RuntimeError("HDTomo recipe metadata requires the assets extra") from error
    with olefile.OleFileIO(path) as stream:
        return _xradia_recipe_metadata(stream)


def _xradia_recipe_metadata(stream: Any) -> dict[str, Any]:
    source_sample = float(_ole_scalar(stream, "Recipe/SSDistance", "<d"))
    detector_sample = float(_ole_scalar(stream, "Recipe/DSDistance", "<d"))
    count = int(_ole_integer(stream, "Recipe/NoOfImages"))
    start = float(_ole_scalar(stream, "Recipe/StartAngle", "<d"))
    end = float(_ole_scalar(stream, "Recipe/EndAngle", "<d"))
    if count < 2:
        raise ValueError("HDTomo recipe must describe at least two projections")
    dso = abs(source_sample)
    return {
        "DSO": dso,
        "DSD": dso + abs(detector_sample),
        "projection_count": count,
        "angle_first_degrees": start,
        "angle_interval_degrees": (end - start) / (count - 1),
        "mode": "cone",
    }


def _projection_paths(root: Path, metadata: Mapping[str, Any]) -> list[Path]:
    listed = metadata.get("projection_files")
    if isinstance(listed, list) and all(isinstance(value, str) for value in listed):
        paths = [root / value for value in listed]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"prepared projection file is missing: {missing[0]}")
        return paths
    roots = [
        path for path in root.rglob("*")
        if path.is_dir() and "projection" in path.name.lower()
    ]
    search_roots = roots or [root]
    return sorted({
        path
        for search_root in search_roots
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".npy", ".tif", ".tiff", ".txrm"}
    })


def _read_array(
    path: Path, *, mmap: bool = False, frame_index: int | None = None,
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if path.suffix.lower() in {".txrm", ".txm"}:
        return _read_xradia(path, frame_index=frame_index)
    try:
        import tifffile
    except ImportError as error:
        raise RuntimeError("TIFF datasets require the assets extra") from error
    return np.asarray(tifffile.imread(path))


def _apply_intensity_transform(array: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if transform == "identity":
        return values
    if transform != "negative_log":
        raise ValueError(f"unknown intensity transform: {transform}")
    incident = float(np.max(values))
    if not math.isfinite(incident) or incident <= 0:
        raise ValueError("negative-log input must contain a positive finite intensity")
    floor = max(np.finfo(np.float32).tiny, incident * 1e-7)
    return -np.log(np.clip(values, floor, incident) / incident).astype(np.float32)


def _copy_reference_volume(source: Path | None, output: Path) -> Path | None:
    if source is None:
        return None
    target = output / "vol_gt.npy"
    if source.is_dir():
        slices = _tiff_stack_paths(source)
        if not slices:
            raise ValueError("reference reconstruction directory contains no TIFF slices")
        first = _read_array(slices[0])
        if first.ndim != 2:
            raise ValueError("reference reconstruction slices must be two-dimensional")
        volume = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.float32, shape=(len(slices), *first.shape),
        )
        for index, path in enumerate(slices):
            array = _read_array(path)
            if array.shape != first.shape:
                raise ValueError("reference reconstruction slice shapes do not match")
            volume[index] = array
        volume.flush()
        del volume
    else:
        volume = _read_array(source).astype(np.float32, copy=False)
        np.save(target, volume, allow_pickle=False)
    return target


def _projection_target(manifest: DatasetManifest, output: Path, index: int) -> Path:
    if index in set(manifest.train_indices):
        split = "train"
        ordinal = manifest.train_indices.index(index)
    else:
        split = "test"
        ordinal = manifest.test_indices.index(index)
    return output / f"proj_{split}" / f"proj_{split}_{ordinal:04d}.npy"


def _prepare_initialization(
    manifest: DatasetManifest, output: Path, reference: Path | None,
) -> Path | None:
    target = output / f"init_{output.name}.npy"
    if manifest.initialization is not None:
        values = np.load(manifest.initialization, allow_pickle=False)
        np.save(target, values, allow_pickle=False)
        return target
    if reference is None:
        return None
    from .gr_gaussian import denoised_point_cloud_initialization

    volume = np.load(reference, mmap_mode="r", allow_pickle=False)
    stride = tuple(max(1, math.ceil(size / 128)) for size in volume.shape)
    sample = np.asarray(volume[tuple(slice(None, None, step) for step in stride)])
    finite = sample[np.isfinite(sample)]
    positive = finite[finite > 0]
    if not positive.size:
        return None
    threshold = float(np.quantile(positive, 0.05))
    state = denoised_point_cloud_initialization(
        sample, min(50_000, sample.size), density_threshold=threshold,
    )
    np.save(target, np.column_stack((state.means, state.densities)), allow_pickle=False)
    return target


def _write_r2_metadata(manifest: DatasetManifest) -> None:
    metadata = manifest.metadata
    geometry = manifest.geometry
    scanner_source = metadata.get("scanner", {})
    if not isinstance(scanner_source, Mapping):
        scanner_source = {}
    voxel_size = _physical_spacing(
        metadata, scanner_source, "voxel_size", "dVoxel", "sVoxel",
        geometry.volume_shape,
    )
    detector_pixel_size = _physical_spacing(
        metadata, scanner_source, "detector_pixel_size", "dDetector", "sDetector",
        geometry.detector_shape,
    )
    if voxel_size.shape != (3,) or detector_pixel_size.shape != (2,):
        raise ValueError("voxel and detector pixel sizes have invalid shapes")
    scanner = {
        "mode": str(metadata.get("mode", scanner_source.get("mode", "cone"))),
        "DSD": geometry.source_detector_distance,
        "DSO": geometry.source_object_distance,
        "nDetector": list(geometry.detector_shape),
        "sDetector": _physical_extent(
            scanner_source, "sDetector", detector_pixel_size, geometry.detector_shape,
        ),
        "nVoxel": list(geometry.volume_shape),
        "sVoxel": _physical_extent(
            scanner_source, "sVoxel", voxel_size, geometry.volume_shape,
        ),
        "offOrigin": list(metadata.get(
            "offOrigin", scanner_source.get("offOrigin", (0.0, 0.0, 0.0)),
        )),
        "offDetector": list(metadata.get(
            "offDetector", scanner_source.get("offDetector", (0.0, 0.0)),
        )),
    }
    records = [
        {"file_path": item.path.relative_to(manifest.root).as_posix(),
         "angle": item.angle_radians}
        for item in manifest.projections
    ]
    document = {
        "scanner": scanner,
        "bbox": metadata.get("bbox", [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        "vol": "vol_gt.npy",
        "proj_train": [records[index] for index in manifest.train_indices],
        "proj_test": [records[index] for index in manifest.test_indices],
    }
    (manifest.root / "meta_data.json").write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )


def _angles(
    metadata: Mapping[str, Any], count: int,
    *, fallback: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    if "angles_radians" in metadata:
        values = tuple(float(value) for value in metadata["angles_radians"])
    elif "angles_degrees" in metadata:
        values = tuple(math.radians(float(value)) for value in metadata["angles_degrees"])
    elif "angle_first_degrees" in metadata and "angle_interval_degrees" in metadata:
        first = float(metadata["angle_first_degrees"])
        interval = float(metadata["angle_interval_degrees"])
        values = tuple(math.radians(first + interval * index) for index in range(count))
    elif fallback is not None:
        values = fallback
    else:
        values = tuple(2.0 * math.pi * index / count for index in range(count))
    if len(values) != count:
        raise ValueError("projection count does not match angle count")
    return values


def _xradia_info(path: Path) -> tuple[int, tuple[int, int], str, tuple[float, ...] | None]:
    try:
        import olefile
    except ImportError as error:
        raise RuntimeError("TXRM/TXM datasets require the assets extra") from error
    with olefile.OleFileIO(path) as stream:
        count = int(_ole_scalar(stream, "ImageInfo/NoOfImages", "<I"))
        width = int(_ole_scalar(stream, "ImageInfo/ImageWidth", "<I"))
        height = int(_ole_scalar(stream, "ImageInfo/ImageHeight", "<I"))
        code = int(_ole_scalar(stream, "ImageInfo/DataType", "<I"))
        dtype = {5: np.dtype("<u2"), 10: np.dtype("<f4")}.get(code)
        if dtype is None:
            raise ValueError(f"unsupported Xradia data type: {code}")
        angles = None
        if stream.exists("ImageInfo/Angles"):
            raw = stream.openstream("ImageInfo/Angles").read()
            degrees = np.frombuffer(raw, dtype="<f4", count=count)
            angles = tuple(float(math.radians(value)) for value in degrees)
    return count, (height, width), dtype.str, angles


def _read_xradia(path: Path, *, frame_index: int | None) -> np.ndarray:
    try:
        import olefile
    except ImportError as error:
        raise RuntimeError("TXRM/TXM datasets require the assets extra") from error
    count, shape, dtype, _ = _xradia_info(path)
    with olefile.OleFileIO(path) as stream:
        if frame_index is not None:
            if frame_index < 0 or frame_index >= count:
                raise IndexError("Xradia frame index is outside the image stack")
            return _ole_image(stream, frame_index, shape, np.dtype(dtype))
        volume = np.empty((count, *shape), dtype=np.dtype(dtype))
        for index in range(count):
            volume[index] = _ole_image(stream, index, shape, np.dtype(dtype))
        return volume


def _ole_scalar(stream: Any, name: str, format_string: str) -> Any:
    if not stream.exists(name):
        raise ValueError(f"Xradia stream is missing: {name}")
    size = struct.calcsize(format_string)
    raw = stream.openstream(name).read()
    if len(raw) < size and format_string == "<d" and len(raw) >= 4:
        return struct.unpack("<f", raw[:4])[0]
    if len(raw) < size:
        raise ValueError(f"Xradia scalar stream has an invalid size: {name}")
    return struct.unpack(format_string, raw[:size])[0]


def _ole_integer(stream: Any, name: str) -> int:
    if not stream.exists(name):
        raise ValueError(f"Xradia stream is missing: {name}")
    raw = stream.openstream(name).read()
    if len(raw) >= 8:
        as_float = struct.unpack("<d", raw[:8])[0]
        if math.isfinite(as_float) and as_float.is_integer() and as_float > 0:
            return int(as_float)
    if len(raw) >= 4:
        return int(struct.unpack("<I", raw[:4])[0])
    raise ValueError(f"Xradia integer stream has an invalid size: {name}")


def _ole_image(
    stream: Any, index: int, shape: tuple[int, int], dtype: np.dtype[Any],
) -> np.ndarray:
    name = f"ImageData{math.ceil((index + 1) / 100)}/Image{index + 1}"
    if not stream.exists(name):
        raise ValueError(f"Xradia image stream is missing: {name}")
    array = np.frombuffer(stream.openstream(name).read(), dtype=dtype)
    if array.size != shape[0] * shape[1]:
        raise ValueError(f"Xradia image stream has an invalid size: {name}")
    return array.reshape(shape)


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    raise ValueError(f"dataset metadata lacks {' or '.join(keys)}")


def _first_match(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        match = next(iter(sorted(root.glob(pattern))), None)
        if match is not None:
            return match
    return None


def _find_reference_volume(root: Path, patterns: tuple[str, ...]) -> Path | None:
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        if directory.name.lower() == "recon" and _tiff_stack_paths(directory):
            return directory
    match = _first_match(root, patterns)
    if match is not None:
        return match
    return next(iter(sorted(root.rglob("*.txm"))), None)


def _tiff_stack_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def _reference_volume_shape(reference: Path) -> tuple[int, ...]:
    if reference.is_dir():
        slices = _tiff_stack_paths(reference)
        if not slices:
            return ()
        first = _read_array(slices[0], mmap=True)
        return (len(slices), *tuple(int(value) for value in first.shape))
    if reference.suffix.lower() == ".txm":
        count, shape, _, _ = _xradia_info(reference)
        return (count, *shape)
    return tuple(int(value) for value in _read_array(reference, mmap=True).shape)


def _physical_spacing(
    metadata: Mapping[str, Any], scanner: Mapping[str, Any], normalized: str,
    direct: str, extent: str, shape: tuple[int, ...],
) -> np.ndarray:
    value = metadata.get(normalized, scanner.get(direct))
    if value is not None:
        result = np.asarray(value, dtype=float)
    elif scanner.get(extent) is not None:
        result = np.asarray(scanner[extent], dtype=float) / np.asarray(shape, dtype=float)
    else:
        result = np.ones(len(shape), dtype=float)
    if result.ndim == 0:
        result = np.repeat(result, len(shape))
    return result


def _physical_extent(
    scanner: Mapping[str, Any], key: str, spacing: np.ndarray,
    shape: tuple[int, ...],
) -> list[float]:
    if scanner.get(key) is not None:
        extent = np.asarray(scanner[key], dtype=float)
    else:
        extent = spacing * np.asarray(shape, dtype=float)
    return extent.tolist()


def _portable_geometry_metadata(
    metadata: Mapping[str, Any], geometry: ScannerGeometry,
) -> dict[str, Any]:
    scanner = metadata.get("scanner", {})
    if not isinstance(scanner, Mapping):
        scanner = {}
    voxel = _physical_spacing(
        metadata, scanner, "voxel_size", "dVoxel", "sVoxel", geometry.volume_shape,
    )
    detector = _physical_spacing(
        metadata, scanner, "detector_pixel_size", "dDetector", "sDetector",
        geometry.detector_shape,
    )
    return {
        "voxel_size": voxel.tolist(),
        "detector_pixel_size": detector.tolist(),
        "mode": str(metadata.get("mode", scanner.get("mode", "cone"))),
        "offOrigin": list(metadata.get(
            "offOrigin", scanner.get("offOrigin", (0.0, 0.0, 0.0)),
        )),
        "offDetector": list(metadata.get(
            "offDetector", scanner.get("offDetector", (0.0, 0.0)),
        )),
        "center_shift_pixels": int(metadata.get("center_shift_pixels", 0)),
        "bbox": metadata.get("bbox", [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
    }


_DATASET_ADAPTERS: Mapping[str, type[_DatasetAdapterBase]] = {
    "chest": ChestDatasetAdapter,
    "walnut": WalnutDatasetAdapter,
    "hdtomo_usb": HDTomoUSBDatasetAdapter,
}


def dataset_descriptors() -> Mapping[str, DatasetDescriptor]:
    return {key: adapter.descriptor for key, adapter in _DATASET_ADAPTERS.items()}


def get_dataset_adapter(dataset_id: str) -> DatasetAdapter:
    try:
        return _DATASET_ADAPTERS[dataset_id]()
    except KeyError as error:
        raise KeyError(f"unknown dataset adapter: {dataset_id}") from error
