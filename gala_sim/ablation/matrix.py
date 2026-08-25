"""The fixed sixteen-row ablation matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class AblationVariant:
    bits: str

    def __post_init__(self) -> None:
        if len(self.bits) != 4 or any(bit not in "01" for bit in self.bits):
            raise ValueError("ablation bits must be four binary characters in ABCD order")

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
    return tuple(AblationVariant(f"{number:04b}") for number in range(16))


def validate_matrix(bits: list[str] | tuple[str, ...]) -> tuple[AblationVariant, ...]:
    variants = tuple(AblationVariant(value) for value in bits)
    expected = all_variants()
    if set(variants) != set(expected) or len(variants) != len(expected):
        raise ValueError("ablation matrix must contain each of the sixteen variants exactly once")
    if variants != expected:
        raise ValueError("ablation matrix must use canonical 0000..1111 order")
    return variants
