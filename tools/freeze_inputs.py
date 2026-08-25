#!/usr/bin/env python3
"""Create the auditable input-freeze record for the first campaign.

This command only freezes identities.  It does not download data, train a
model, or infer missing hardware parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Running a tool by path puts ``tools/`` first on ``sys.path``.  Add the
# repository root explicitly so the command works without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gala_sim.config import ConfigError, load_config
from gala_sim.manifest import (build_freeze_record, dataset_record, source_record,
                               write_freeze_record)


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
    required_scanner = {"nVoxel", "dVoxel", "nDetector", "dDetector", "DSO", "DSD"}
    if not isinstance(scanner, dict) or not required_scanner.issubset(scanner):
        raise ValueError("Chest meta_data.json lacks the required scanner geometry")
    for split in ("proj_train", "proj_test"):
        if not any((root / split).glob("*.npy")):
            raise ValueError(f"Chest {split} has no projection arrays")


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
        record = build_freeze_record(config, source, dataset, args.seed, args.repository)
        write_freeze_record(record, args.output)
    except (ConfigError, OSError, ValueError) as error:
        print(f"freeze_inputs: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
