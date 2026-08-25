from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gala_sim.ablation import AblationVariant, all_variants, validate_matrix
from gala_sim.results import AblationRow, write_ablation_csv


def _rows(config_hash: str = "a" * 64) -> list[AblationRow]:
    return [AblationRow(
        model="fixture", dataset="fixture", bits=variant.bits, cycles=100 - index,
        speedup_vs_base_asic=100 / (100 - index), local_gpu_seconds=None,
        orin_seconds=None, speedup_vs_orin=None, psnr_delta_db=0.0,
        ssim_delta=0.0, lpips_delta=0.0, config_sha256=config_hash, status="passed",
    ) for index, variant in enumerate(all_variants())]


def test_ablation_matrix_has_canonical_sixteen_rows(tmp_path: Path) -> None:
    variants = validate_matrix([variant.bits for variant in all_variants()])
    assert variants[0] == AblationVariant("0000")
    assert variants[-1] == AblationVariant("1111")
    output = tmp_path / "ablation.csv"
    write_ablation_csv(_rows(), output)
    with output.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 16


def test_ablation_matrix_rejects_missing_variant() -> None:
    with pytest.raises(ValueError, match="sixteen"):
        validate_matrix([variant.bits for variant in all_variants()][:-1])
