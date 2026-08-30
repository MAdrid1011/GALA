"""Ablation output rows and consistency checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from gala_sim.ablation.matrix import (
    ASIC_BASE_VARIANTS, GPU_COMPILER_VARIANTS, AblationVariant, all_variants,
    comparison_baseline, validate_matrix,
)


@dataclass(frozen=True)
class AblationRow:
    model: str
    dataset: str
    bits: str
    cycles: int
    speedup_vs_base_asic: float | None
    local_gpu_seconds: float | None
    orin_seconds: float | None
    speedup_vs_orin: float | None
    psnr_delta_db: float | None
    ssim_delta: float | None
    lpips_delta: float | None
    config_sha256: str
    status: str
    module_breakdown_path: str | None = None
    run_id: str | None = None
    comparison_baseline: str = ""
    gpu_base_seconds: float | None = None
    speedup_vs_gpu_base: float | None = None


def validate_full_variant(base_cycles: int, full_cycles: int, entry_bits: str,
                          *, complete_cycles: int | None = None) -> None:
    if entry_bits != "1111":
        raise ValueError("full GALA consistency check requires the 1111 row")
    if base_cycles <= 0 or full_cycles <= 0:
        raise ValueError("cycle counts must be positive")
    if complete_cycles is not None and full_cycles != complete_cycles:
        raise ValueError("1111 cycles differ from the complete GALA entry")


def write_ablation_csv(rows: Iterable[AblationRow], path: Path) -> None:
    rows = list(rows)
    validate_matrix([row.bits for row in rows])
    if rows[0].model != rows[-1].model or rows[0].dataset != rows[-1].dataset:
        raise ValueError("ablation rows do not describe one model/dataset pair")
    for row in rows:
        expected_baseline = comparison_baseline(row.bits)
        if row.comparison_baseline != expected_baseline:
            raise ValueError(
                f"variant {row.bits} must use {expected_baseline} as its baseline"
            )
        if (
            row.bits in GPU_COMPILER_VARIANTS or row.bits == "0000"
        ) and row.speedup_vs_base_asic is not None:
            raise ValueError(
                f"variant {row.bits} cannot report speedup versus Base ASIC"
            )
        if row.bits in ASIC_BASE_VARIANTS and row.speedup_vs_gpu_base is not None:
            raise ValueError(
                f"variant {row.bits} cannot report compiler speedup versus GPU Base"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
