"""A/B/C/D switch semantics and matrix validation."""

from .matrix import (
    AblationVariant, asic_speedup, all_variants, comparison_baseline,
    composition_assessment, gpu_speedup,
    parse_variant, validate_matrix,
)
from .anchors import (
    ANCHOR_ACCEPTABLE_MIN_FRACTION,
    AnchorGateResult, STATIC_ANCHOR_VERSION, StaticAnchor, assess_upper_bound,
    compiler_target_speedups, hardware_target_speedups, static_anchor_matrix,
    static_anchors, speedup_within_anchor_tolerance,
)
from .runner import (
    AblationRun, run_archive_matrix, run_archive_speedup_diagnostic, run_matrix,
)

__all__ = [
    "ANCHOR_ACCEPTABLE_MIN_FRACTION",
    "AnchorGateResult", "STATIC_ANCHOR_VERSION", "StaticAnchor",
    "assess_upper_bound", "compiler_target_speedups", "hardware_target_speedups",
    "static_anchor_matrix", "static_anchors", "speedup_within_anchor_tolerance",
    "AblationVariant", "asic_speedup", "all_variants", "comparison_baseline",
    "composition_assessment", "gpu_speedup", "parse_variant", "validate_matrix",
    "AblationRun", "run_archive_matrix", "run_archive_speedup_diagnostic",
    "run_matrix",
]
