"""Canonical compiler/architecture mechanism configurations."""

from __future__ import annotations


# ABCD: query compiler, semantic compiler, query architecture, residency architecture.
CANONICAL_VARIANT_BITS = (
    "0000",  # Base
    "1000",  # Query compiler
    "1010",  # Query compiler + query architecture
    "0100",  # Semantic compiler
    "0101",  # Semantic compiler + residency architecture
    "1100",  # Query compiler + semantic compiler
    "1111",  # Full GALA
)
CANONICAL_VARIANT_POLICIES = tuple(
    f"variant:{bits}" for bits in CANONICAL_VARIANT_BITS
)
CYCLE_POLICY_NAMES = (
    "base", "query", "residency", "full",
    "query_oracle", "residency_oracle",
    *CANONICAL_VARIANT_POLICIES,
)
ONLINE_POLICY_NAMES = (
    "base", "query", "residency", "full",
    *CANONICAL_VARIANT_POLICIES,
)


def validate_variant_bits(bits: str) -> str:
    if len(bits) != 4 or any(bit not in "01" for bit in bits):
        raise ValueError("mechanism bits must be four binary characters in ABCD order")
    if bits[2] == "1" and bits[0] != "1":
        raise ValueError(
            "overlap-guided issue requires compiler query load rules"
        )
    if bits[3] == "1" and bits[1] != "1":
        raise ValueError(
            "semantic residency requires compiler semantic worksets"
        )
    if bits not in CANONICAL_VARIANT_BITS:
        raise ValueError(
            "mechanism configuration is not one of the seven canonical evaluations"
        )
    return bits
