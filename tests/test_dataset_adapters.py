from __future__ import annotations

import json
import io
from pathlib import Path
import struct

import numpy as np

from gala_sim.adapters import get_dataset_adapter
from gala_sim.adapters import datasets as dataset_module


class _FakeOLE:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values

    def exists(self, name: str) -> bool:
        return name in self.values

    def openstream(self, name: str) -> io.BytesIO:
        return io.BytesIO(self.values[name])


def test_walnut_adapter_loads_projection_geometry(tmp_path: Path) -> None:
    projections = tmp_path / "projections"
    projections.mkdir()
    np.save(projections / "projection_000.npy", np.ones((3, 4), dtype=np.float32))
    np.save(projections / "projection_001.npy", np.ones((3, 4), dtype=np.float32) * 2)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "angles_degrees": [0.0, 0.5],
        "detector_shape": [3, 4],
        "volume_shape": [4, 4, 4],
        "DSO": 100.0,
        "DSD": 200.0,
    }), encoding="utf-8")
    manifest = get_dataset_adapter("walnut").load(tmp_path)
    assert manifest.geometry.detector_shape == (3, 4)
    assert np.isclose(manifest.projections[1].angle_radians, np.deg2rad(0.5))
    assert manifest.intensity_transform == "negative_log"


def test_walnut_adapter_parses_publisher_ini_metadata(tmp_path: Path) -> None:
    tifffile = __import__("tifffile")
    projections = tmp_path / "20201111_walnut_projections"
    projections.mkdir()
    tifffile.imwrite(projections / "0000.tiff", np.ones((3, 4), dtype=np.uint16))
    tifffile.imwrite(projections / "0001.tiff", np.ones((3, 4), dtype=np.uint16))
    (tmp_path / "20201111_walnut_.txt").write_text(
        "DistanceSourceDetector=553.74\n"
        "DistanceSourceOrigin=210.66\n"
        "NumberImages=2\n"
        "AngleFirst=0\n"
        "AngleInterval=0.5\n"
        "PixelSize=0.050\n"
        "GeometryType=Cone\n",
        encoding="utf-8",
    )
    manifest = get_dataset_adapter("walnut").load(tmp_path)
    assert manifest.geometry.volume_shape == (512, 512, 512)
    assert manifest.geometry.detector_shape == (3, 4)
    assert np.isclose(manifest.projections[1].angle_radians, np.deg2rad(0.5))
    assert manifest.test_indices == (0,)
    expected = 3 * 0.050 * 210.66 / 553.74 / 512
    assert np.allclose(manifest.metadata["voxel_size"], [expected] * 3)


def test_hdtomo_adapter_loads_exported_volume_and_views(tmp_path: Path) -> None:
    projections = tmp_path / "projections"
    reconstruction = tmp_path / "recon"
    projections.mkdir()
    reconstruction.mkdir()
    np.save(projections / "000.npy", np.ones((2, 3), dtype=np.float32))
    np.save(projections / "001.npy", np.ones((2, 3), dtype=np.float32))
    np.save(reconstruction / "volume.npy", np.ones((2, 2, 2), dtype=np.float32))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "angles_degrees": [0.0, 1.0],
        "detector_shape": [2, 3],
        "volume_shape": [2, 2, 2],
        "DSO": 50.0,
        "DSD": 120.0,
    }), encoding="utf-8")
    manifest = get_dataset_adapter("hdtomo_usb").load(tmp_path)
    assert manifest.reference_volume == reconstruction / "volume.npy"
    assert len(manifest.projections) == 2
    assert manifest.geometry.source_detector_distance == 120.0


def test_xradia_recipe_metadata_derives_geometry_and_angles() -> None:
    stream = _FakeOLE({
        "Recipe/SSDistance": struct.pack("<f", -95.0478515625),
        "Recipe/DSDistance": struct.pack("<f", 55.082183837890625),
        "Recipe/NoOfImages": struct.pack("<I", 801),
        "Recipe/StartAngle": struct.pack("<f", -180.0),
        "Recipe/EndAngle": struct.pack("<f", 180.0),
    })
    metadata = dataset_module._xradia_recipe_metadata(stream)
    assert np.isclose(metadata["DSO"], 95.0478515625)
    assert np.isclose(metadata["DSD"], 150.13003540039062)
    assert metadata["projection_count"] == 801
    assert np.isclose(metadata["angle_interval_degrees"], 0.45)


def test_hdtomo_adapter_discovers_recipe_and_streams_reconstruction_stack(
    monkeypatch, tmp_path: Path,
) -> None:
    tifffile = __import__("tifffile")
    scan = tmp_path / "gruppe 4" / "tomo-A"
    projections = scan / "projections"
    reconstruction = scan / "recon"
    projections.mkdir(parents=True)
    reconstruction.mkdir()
    (tmp_path / "gruppe 4" / "gruppe 4.rcp").write_bytes(b"fixture")
    for index in range(2):
        tifffile.imwrite(
            projections / f"projection_{index:04d}.tiff",
            np.full((2, 3), index + 1, dtype=np.uint16),
        )
    for index in range(3):
        tifffile.imwrite(
            reconstruction / f"recon_{index:04d}.tiff",
            np.full((2, 2), index, dtype=np.uint16),
        )
    monkeypatch.setattr(dataset_module, "_load_xradia_recipe", lambda path: {
        "DSO": 95.0, "DSD": 150.0, "projection_count": 2,
        "angle_first_degrees": -180.0, "angle_interval_degrees": 180.0,
    })
    adapter = get_dataset_adapter("hdtomo_usb")
    manifest = adapter.load(tmp_path)
    assert manifest.reference_volume == reconstruction
    assert manifest.geometry.volume_shape == (3, 2, 2)
    converted = adapter.convert(manifest, tmp_path / "converted")
    volume = np.load(converted.reference_volume, mmap_mode="r")
    assert volume.shape == (3, 2, 2)
    assert np.array_equal(volume[:, 0, 0], [0.0, 1.0, 2.0])


def test_walnut_conversion_applies_shift_and_negative_log(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    projections = raw / "projections"
    projections.mkdir(parents=True)
    np.save(projections / "000.npy", np.asarray([[1.0, 0.5, 0.25]], dtype=np.float32))
    np.save(projections / "001.npy", np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32))
    (raw / "metadata.json").write_text(json.dumps({
        "angles_degrees": [0.0, 1.0], "detector_shape": [1, 3],
        "volume_shape": [2, 2, 2], "DSO": 50.0, "DSD": 120.0,
        "center_shift_pixels": 0,
    }), encoding="utf-8")
    adapter = get_dataset_adapter("walnut")
    converted = adapter.convert(adapter.load(raw), tmp_path / "converted")
    values = np.load(converted.projections[0].path)
    assert np.allclose(values, [[0.0, np.log(2.0), np.log(4.0)]])
    metadata = json.loads((converted.root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["projection_files"] == [
        "proj_train/proj_train_0000.npy", "proj_test/proj_test_0000.npy",
    ]
    r2_metadata = json.loads((converted.root / "meta_data.json").read_text(encoding="utf-8"))
    assert r2_metadata["scanner"]["nDetector"] == [1, 3]
    assert r2_metadata["proj_train"][0]["file_path"] == "proj_train/proj_train_0000.npy"


def test_hdtomo_txrm_stack_expands_into_projection_records(
    monkeypatch, tmp_path: Path,
) -> None:
    projections = tmp_path / "projections"
    projections.mkdir()
    stack = projections / "usb.txrm"
    stack.write_bytes(b"fixture")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "detector_shape": [2, 3], "volume_shape": [2, 2, 2],
        "DSO": 50.0, "DSD": 120.0,
    }), encoding="utf-8")
    angles = (0.0, 0.5)
    frames = np.asarray([
        [[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]],
        [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
    ], dtype=np.float32)
    monkeypatch.setattr(
        dataset_module, "_xradia_info",
        lambda path: (2, (2, 3), "<f4", angles),
    )
    monkeypatch.setattr(
        dataset_module, "_read_xradia",
        lambda path, *, frame_index: frames if frame_index is None else frames[frame_index],
    )
    adapter = get_dataset_adapter("hdtomo_usb")
    manifest = adapter.load(tmp_path)
    assert [item.frame_index for item in manifest.projections] == [0, 1]
    assert [item.angle_radians for item in manifest.projections] == [0.0, 0.5]
    converted = adapter.convert(manifest, tmp_path / "converted")
    assert len(converted.projections) == 2
    assert np.all(np.load(converted.projections[0].path) >= 0)


def test_prepared_dataset_manifest_reloads_without_discovering_auxiliary_arrays(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    projections = raw / "projections"
    projections.mkdir(parents=True)
    np.save(projections / "000.npy", np.ones((2, 3), dtype=np.float32))
    np.save(projections / "001.npy", np.ones((2, 3), dtype=np.float32))
    (raw / "metadata.json").write_text(json.dumps({
        "angles_degrees": [0.0, 1.0], "detector_shape": [2, 3],
        "volume_shape": [2, 2, 2], "DSO": 50.0, "DSD": 120.0,
    }), encoding="utf-8")
    adapter = get_dataset_adapter("walnut")
    converted = adapter.convert(adapter.load(raw), tmp_path / "converted")
    reloaded = adapter.load(converted.root)
    assert len(reloaded.projections) == 2
    assert reloaded.train_indices == converted.train_indices
    assert reloaded.test_indices == converted.test_indices
    upstream = json.loads(
        (converted.root / "meta_data.json").read_text(encoding="utf-8")
    )
    assert upstream["scanner"]["mode"] == "cone"
    assert upstream["reference_volume_available"] is False
    sentinel = converted.root / upstream["vol"]
    assert sentinel.name == ".gala_no_ground_truth.npy"
    assert np.load(sentinel).shape == (1, 1, 1)
    assert converted.initialization is not None
    initialization = np.load(converted.initialization, mmap_mode="r")
    assert initialization.shape == (50_000, 4)
    assert np.isfinite(initialization).all()
    provenance = json.loads(
        converted.initialization.with_suffix(".provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["method"] == "deterministic_projection_sample_lattice"
    assert provenance["projection_count"] == 1
    assert reloaded.initialization == converted.initialization


def test_chest_conversion_preserves_physical_scanner_geometry(tmp_path: Path) -> None:
    raw = tmp_path / "chest"
    (raw / "proj_train").mkdir(parents=True)
    (raw / "proj_test").mkdir()
    np.save(raw / "proj_train/000.npy", np.ones((2, 3), dtype=np.float32))
    np.save(raw / "proj_test/000.npy", np.ones((2, 3), dtype=np.float32))
    np.save(raw / "vol_gt.npy", np.ones((2, 2, 2), dtype=np.float32))
    np.save(raw / "init_chest.npy", np.ones((2, 4), dtype=np.float32))
    (raw / "meta_data.json").write_text(json.dumps({
        "scanner": {
            "mode": "cone", "DSO": 50.0, "DSD": 120.0,
            "nDetector": [2, 3], "sDetector": [0.4, 0.9],
            "nVoxel": [2, 2, 2], "dVoxel": [0.1, 0.2, 0.3],
            "offOrigin": [1.0, 2.0, 3.0], "offDetector": [4.0, 5.0],
        },
        "bbox": [[-2.0, -1.0, 0.0], [2.0, 1.0, 3.0]],
        "vol": "vol_gt.npy",
        "proj_train": [{"file_path": "proj_train/000.npy", "angle": 0.0}],
        "proj_test": [{"file_path": "proj_test/000.npy", "angle": 1.0}],
    }), encoding="utf-8")
    adapter = get_dataset_adapter("chest")
    converted = adapter.convert(adapter.load(raw), tmp_path / "prepared")
    metadata = json.loads((converted.root / "meta_data.json").read_text(encoding="utf-8"))
    assert metadata["scanner"]["sVoxel"] == [0.2, 0.4, 0.6]
    assert metadata["scanner"]["sDetector"] == [0.4, 0.9]
    assert metadata["scanner"]["offOrigin"] == [1.0, 2.0, 3.0]
    assert metadata["scanner"]["offDetector"] == [4.0, 5.0]
    assert metadata["bbox"] == [[-2.0, -1.0, 0.0], [2.0, 1.0, 3.0]]


def test_chest_adapter_discovers_publisher_archive_directory(tmp_path: Path) -> None:
    nested = tmp_path / "cone_ntrain_50_angle_360" / "0_chest_cone"
    (nested / "proj_train").mkdir(parents=True)
    (nested / "proj_test").mkdir()
    np.save(nested / "proj_train/000.npy", np.ones((2, 2), dtype=np.float32))
    np.save(nested / "proj_test/000.npy", np.ones((2, 2), dtype=np.float32))
    np.save(nested / "vol_gt.npy", np.ones((2, 2, 2), dtype=np.float32))
    np.save(nested / "init_nested.npy", np.ones((2, 4), dtype=np.float32))
    (nested / "meta_data.json").write_text(json.dumps({
        "scanner": {"nDetector": [2, 2], "nVoxel": [2, 2, 2],
                    "DSO": 50.0, "DSD": 100.0},
        "proj_train": [{"file_path": "proj_train/000.npy", "angle": 0.0}],
        "proj_test": [{"file_path": "proj_test/000.npy", "angle": 1.0}],
        "vol": "vol_gt.npy",
    }), encoding="utf-8")
    manifest = get_dataset_adapter("chest").load(tmp_path)
    assert manifest.root == nested.resolve()
