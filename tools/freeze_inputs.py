#!/usr/bin/env python3
"""Create the auditable input-freeze record for the first campaign.

This command only freezes identities.  It does not download data, train a
model, or infer missing hardware parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

# Running a tool by path puts ``tools/`` first on ``sys.path``.  Add the
# repository root explicitly so the command works without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gala_sim.config import ConfigError, load_config
from gala_sim.manifest import (build_freeze_record, dataset_record, source_record,
                               training_record, write_freeze_record)


MODEL_URL = "https://github.com/Ruyi-Zha/r2_gaussian.git"
MODEL_COMMIT = "f2579bfddd9aac009cb797c8503bef8119bbd022"
DATA_URL = "https://drive.google.com/drive/folders/1YZ3w87XrCNyjDRos6gkY8zgT5hESl-PN?usp=sharing"
DATA_LICENSE_URL = "https://www.cancerimagingarchive.net/collection/lidc-idri/"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path,
                        help="clean checkout of the locked R²-Gaussian source")
    parser.add_argument("--config", default="configs/architecture/gala.yaml", type=Path)
    parser.add_argument("--dataset-root", type=Path,
                        help="prepared Chest root; omit only with --dataset-reason")
    parser.add_argument("--dataset-reason",
                        help="machine-readable reason when the dataset cannot be used")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--training-profile",
                        default="configs/campaigns/r2_gaussian_chest.yaml", type=Path)
    parser.add_argument("--model-output", type=Path,
                        help="frozen output directory for the official training command")
    parser.add_argument("--repository", default=".", type=Path)
    parser.add_argument("--output", default="records/input_freeze/r2_gaussian_chest.json", type=Path)
    return parser


def _check_dataset_root(root: Path) -> None:
    required = ("meta_data.json", "vol_gt.npy", "proj_train", "proj_test")
    missing = [item for item in required if not (root / item).exists()]
    if not any(root.glob("init_*.npy")):
        missing.append("init_*.npy")
    if missing:
        raise ValueError("Chest dataset is incomplete: " + ", ".join(missing))
    try:
        metadata = json.loads((root / "meta_data.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Chest meta_data.json is not valid JSON") from error
    scanner = metadata.get("scanner")
    # R2-Gaussian stores total detector/volume sizes as sDetector/sVoxel;
    # dDetector/dVoxel are derived by the upstream loader.
    required_scanner = {"nVoxel", "sVoxel", "nDetector", "sDetector", "DSO", "DSD"}
    if not isinstance(scanner, dict) or not required_scanner.issubset(scanner):
        raise ValueError("Chest meta_data.json lacks the required scanner geometry")
    for field, expected_length in (("nVoxel", 3), ("sVoxel", 3),
                                   ("nDetector", 2), ("sDetector", 2)):
        value = scanner[field]
        if not isinstance(value, list) or len(value) != expected_length:
            raise ValueError(f"Chest scanner field {field} has an invalid shape")
    try:
        if any(int(value) <= 0 for value in scanner["nVoxel"] + scanner["nDetector"]):
            raise ValueError("Chest scanner counts must be positive")
        if any(not math.isfinite(float(value)) or float(value) <= 0
               for value in scanner["sVoxel"] + scanner["sDetector"]):
            raise ValueError("Chest scanner sizes must be positive")
        if any(not math.isfinite(float(value)) or float(value) <= 0
               for value in (scanner["DSO"], scanner["DSD"])):
            raise ValueError("Chest scanner distances must be positive")
    except (TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Chest scanner"):
            raise
        raise ValueError("Chest scanner geometry contains non-numeric values") from error
    for split in ("proj_train", "proj_test"):
        entries = metadata.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Chest meta_data.json has no {split} entries")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("file_path"), str):
                raise ValueError(f"Chest {split} contains an invalid projection entry")
            if not (root / entry["file_path"]).is_file():
                raise ValueError(f"Chest projection file is missing: {entry['file_path']}")
            if not isinstance(entry.get("angle"), (int, float)):
                raise ValueError(f"Chest {split} contains a projection without an angle")
    for split in ("proj_train", "proj_test"):
        files = sorted((root / split).glob("*.npy"))
        if not files:
            raise ValueError(f"Chest {split} has no projection arrays")
        entries = metadata[split]
        listed = {entry["file_path"] for entry in entries}
        actual = {path.relative_to(root).as_posix() for path in files}
        if listed != actual:
            raise ValueError(f"Chest {split} metadata/files mismatch")
        sample = np.load(files[0], mmap_mode="r")
        expected = tuple(int(value) for value in scanner["nDetector"])
        if sample.shape != expected or sample.ndim != 2:
            raise ValueError(f"Chest {split} projection shape {sample.shape} != {expected}")
        if not np.isfinite(sample).all() or float(sample.min()) < 0:
            raise ValueError(f"Chest {split} projection contains invalid values")
    volume = np.load(root / "vol_gt.npy", mmap_mode="r")
    expected_volume = tuple(int(value) for value in scanner["nVoxel"])
    if volume.shape != expected_volume or volume.ndim != 3:
        raise ValueError(f"Chest volume shape {volume.shape} != {expected_volume}")
    if not np.isfinite(volume).all():
        raise ValueError("Chest volume contains non-finite values")
    init_files = sorted(root.glob("init_*.npy"))
    init = np.load(init_files[0], mmap_mode="r")
    if init.ndim != 2 or init.shape[1] < 4 or init.shape[0] == 0:
        raise ValueError(f"Chest initialization array has an invalid shape: {init.shape}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        source = source_record(args.source_root, "R2-Gaussian", MODEL_URL, MODEL_COMMIT)
        if args.dataset_root is not None:
            _check_dataset_root(args.dataset_root.resolve())
        elif not args.dataset_reason:
            raise ValueError("--dataset-root or --dataset-reason is required")
        dataset = dataset_record(args.dataset_root, "Chest", DATA_URL, DATA_LICENSE_URL,
                                 args.dataset_reason)
        training = training_record(args.training_profile, source, args.dataset_root,
                                   args.model_output)
        record = build_freeze_record(config, source, dataset, training, args.seed,
                                     args.repository)
        write_freeze_record(record, args.output)
    except (ConfigError, OSError, ValueError) as error:
        print(f"freeze_inputs: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
