"""The seven canonical compiler/architecture mechanism evaluations."""

from __future__ import annotations

from dataclasses import dataclass

from gala_sim.mechanisms import CANONICAL_VARIANT_BITS, validate_variant_bits


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
