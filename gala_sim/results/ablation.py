"""Ablation output rows and consistency checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from gala_sim.ablation import AblationVariant, all_variants, validate_matrix


@dataclass(frozen=True)
class AblationRow:
    model: str
    dataset: str
    bits: str
    cycles: int
    speedup_vs_base_asic: float
    local_gpu_seconds: float | None
    orin_seconds: float | None
    speedup_vs_orin: float | None
    psnr_delta_db: float | None
    ssim_delta: float | None
    lpips_delta: float | None
    config_sha256: str
    status: str


def validate_full_variant(base_cycles: int, full_cycles: int, entry_bits: str) -> None:
    if entry_bits != "1111":
        raise ValueError("full GALA consistency check requires the 1111 row")
    if base_cycles <= 0 or full_cycles <= 0:
        raise ValueError("cycle counts must be positive")


def write_ablation_csv(rows: Iterable[AblationRow], path: Path) -> None:
    rows = list(rows)
    validate_matrix([row.bits for row in rows])
    if rows[0].model != rows[-1].model or rows[0].dataset != rows[-1].dataset:
        raise ValueError("ablation rows do not describe one model/dataset pair")
    config_hashes = {row.config_sha256 for row in rows}
    if len(config_hashes) != 1:
        raise ValueError("ablation rows use different configuration hashes")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
