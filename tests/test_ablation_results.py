from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gala_sim.ablation import AblationVariant, all_variants, validate_matrix
from gala_sim.results import (
    AblationRow, asic_speedup, comparison_baseline, gpu_speedup,
    write_ablation_csv,
)


def _rows(config_hash: str = "a" * 64) -> list[AblationRow]:
    return [AblationRow(
        model="fixture", dataset="fixture", bits=variant.bits, cycles=100 - index,
        speedup_vs_base_asic=asic_speedup(
            variant.bits, base_cycles=100, cycles=100 - index,
        ),
        local_gpu_seconds=None,
        orin_seconds=None, speedup_vs_orin=None, psnr_delta_db=0.0,
        ssim_delta=0.0, lpips_delta=0.0, config_sha256=config_hash, status="passed",
        comparison_baseline=comparison_baseline(variant.bits),
        gpu_base_seconds=None, speedup_vs_gpu_base=None,
    ) for index, variant in enumerate(all_variants())]


def test_ablation_matrix_has_canonical_seven_rows(tmp_path: Path) -> None:
    variants = validate_matrix([variant.bits for variant in all_variants()])
    assert [variant.bits for variant in variants] == [
        "0000", "1000", "1010", "0100", "0101", "1100", "1111",
    ]
    output = tmp_path / "ablation.csv"
    write_ablation_csv(_rows(), output)
    with output.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 7


def test_ablation_matrix_rejects_missing_variant() -> None:
    with pytest.raises(ValueError, match="seven"):
        validate_matrix([variant.bits for variant in all_variants()][:-1])


@pytest.mark.parametrize("bits", ["0010", "0001", "0111"])
def test_ablation_variant_rejects_unsupported_mechanism_combinations(
    bits: str,
) -> None:
    with pytest.raises(ValueError):
        AblationVariant(bits)


def test_ablation_output_does_not_gate_on_recorded_configuration_hashes(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows[1] = AblationRow(**{
        **rows[1].__dict__, "config_sha256": "different-recorded-hash",
    })
    write_ablation_csv(rows, tmp_path / "ablation.csv")


def test_ablation_output_keeps_module_breakdown_and_run_identity(tmp_path: Path) -> None:
    rows = _rows()
    rows[0] = AblationRow(**{
        **rows[0].__dict__,
        "module_breakdown_path": "ablation.csv.modules/0000.json",
        "run_id": "fixture-0000",
    })
    output = tmp_path / "ablation.csv"
    write_ablation_csv(rows, output)
    with output.open(newline="", encoding="utf-8") as stream:
        first = next(csv.DictReader(stream))
    assert first["module_breakdown_path"] == "ablation.csv.modules/0000.json"
    assert first["run_id"] == "fixture-0000"


@pytest.mark.parametrize("bits", ["0000", "1000", "0100", "1100"])
def test_gpu_baseline_variants_reject_asic_speedup(
    tmp_path: Path, bits: str,
) -> None:
    rows = _rows()
    index = next(index for index, row in enumerate(rows) if row.bits == bits)
    rows[index] = AblationRow(**{
        **rows[index].__dict__, "speedup_vs_base_asic": 1.25,
    })
    with pytest.raises(ValueError, match="cannot report speedup versus Base ASIC"):
        write_ablation_csv(rows, tmp_path / "ablation.csv")


@pytest.mark.parametrize("bits", ["1010", "0101", "1111"])
def test_asic_baseline_variants_reject_gpu_compiler_speedup(
    tmp_path: Path, bits: str,
) -> None:
    rows = _rows()
    index = next(index for index, row in enumerate(rows) if row.bits == bits)
    rows[index] = AblationRow(**{
        **rows[index].__dict__, "speedup_vs_gpu_base": 1.25,
    })
    with pytest.raises(ValueError, match="cannot report compiler speedup versus GPU Base"):
        write_ablation_csv(rows, tmp_path / "ablation.csv")


def test_gpu_compiler_speedup_uses_gpu_time_only() -> None:
    assert gpu_speedup(
        "1000", gpu_base_seconds=10.0, gpu_variant_seconds=8.0,
    ) == pytest.approx(1.25)
    assert gpu_speedup(
        "0100", gpu_base_seconds=None, gpu_variant_seconds=8.0,
    ) is None
    assert gpu_speedup(
        "1010", gpu_base_seconds=10.0, gpu_variant_seconds=8.0,
    ) is None


def test_base_asic_has_distinct_platform_baseline() -> None:
    assert comparison_baseline("0000") == "agx_orin_gpu_base_estimate"
    assert comparison_baseline("1000") == "gpu_base"
    assert comparison_baseline("1010") == "base_asic"
