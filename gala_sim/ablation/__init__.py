"""A/B/C/D switch semantics and matrix validation."""

from .matrix import AblationVariant, all_variants, parse_variant, validate_matrix
from .runner import AblationRun, run_archive_matrix, run_matrix

__all__ = [
    "AblationVariant", "all_variants", "parse_variant", "validate_matrix",
    "AblationRun", "run_archive_matrix", "run_matrix",
]
