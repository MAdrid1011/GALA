"""The seven canonical compiler/architecture mechanism evaluations."""

from __future__ import annotations

from dataclasses import dataclass

from gala_sim.mechanisms import CANONICAL_VARIANT_BITS, validate_variant_bits


GPU_COMPILER_VARIANTS = frozenset({"1000", "0100", "1100"})
ASIC_BASE_VARIANTS = frozenset({"1010", "0101", "1111"})


def comparison_baseline(bits: str) -> str:
    validate_variant_bits(bits)
    if bits == "0000":
        return "agx_orin_gpu_base_estimate"
    if bits in GPU_COMPILER_VARIANTS:
        return "gpu_base"
    return "base_asic"


def asic_speedup(bits: str, *, base_cycles: int, cycles: int) -> float | None:
    validate_variant_bits(bits)
    if base_cycles <= 0 or cycles <= 0:
        raise ValueError("cycle counts must be positive")
    return base_cycles / cycles if bits in ASIC_BASE_VARIANTS else None


def gpu_speedup(
    bits: str, *, gpu_base_seconds: float | None,
    gpu_variant_seconds: float | None,
) -> float | None:
    validate_variant_bits(bits)
    if bits not in GPU_COMPILER_VARIANTS:
        return None
    if gpu_base_seconds is None or gpu_variant_seconds is None:
        return None
    if gpu_base_seconds <= 0 or gpu_variant_seconds <= 0:
        raise ValueError("GPU times must be positive")
    return gpu_base_seconds / gpu_variant_seconds


@dataclass(frozen=True, order=True)
class AblationVariant:
    bits: str

    def __post_init__(self) -> None:
        validate_variant_bits(self.bits)

    @property
    def compiler_query_load_rules(self) -> bool:
        return self.bits[0] == "1"

    @property
    def compiler_semantic_worksets(self) -> bool:
        return self.bits[1] == "1"

    @property
    def overlap_guided_issue(self) -> bool:
        return self.bits[2] == "1"

    @property
    def semantic_residency(self) -> bool:
        return self.bits[3] == "1"


def parse_variant(value: str) -> AblationVariant:
    return AblationVariant(value)


def all_variants() -> tuple[AblationVariant, ...]:
    return tuple(AblationVariant(bits) for bits in CANONICAL_VARIANT_BITS)


def validate_matrix(bits: list[str] | tuple[str, ...]) -> tuple[AblationVariant, ...]:
    variants = tuple(AblationVariant(value) for value in bits)
    expected = all_variants()
    if set(variants) != set(expected) or len(variants) != len(expected):
        raise ValueError(
            "evaluation matrix must contain each of the seven canonical variants exactly once"
        )
    if variants != expected:
        raise ValueError(
            "evaluation matrix must use canonical "
            "0000,1000,1010,0100,0101,1100,1111 order"
        )
    return variants
