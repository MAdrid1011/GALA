"""A/B/C/D switch semantics and matrix validation."""

from .matrix import (
    AblationVariant, asic_speedup, all_variants, comparison_baseline, gpu_speedup,
    parse_variant, validate_matrix,
)
from .anchors import (
    AnchorGateResult, StaticAnchor, assess_upper_bound,
    compiler_target_speedups, hardware_target_speedups, static_anchors,
)
from .runner import (
    AblationRun, run_archive_matrix, run_archive_speedup_diagnostic, run_matrix,
)

__all__ = [
    "AnchorGateResult", "StaticAnchor", "assess_upper_bound",
    "compiler_target_speedups", "hardware_target_speedups", "static_anchors",
    "AblationVariant", "asic_speedup", "all_variants", "comparison_baseline",
    "gpu_speedup", "parse_variant", "validate_matrix",
    "AblationRun", "run_archive_matrix", "run_archive_speedup_diagnostic",
    "run_matrix",
]
