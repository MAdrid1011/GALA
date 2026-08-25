"""Run the fixed sixteen variants over one validated trace."""

from __future__ import annotations

from dataclasses import dataclass, replace

from gala_sim.trace import Trace, validate_trace
from gala_sim.timing import CycleConfig, CycleEngine, CycleResult

from .matrix import AblationVariant, all_variants, validate_matrix


@dataclass(frozen=True)
class AblationRun:
    variant: AblationVariant
    result: CycleResult


def _policy(variant: AblationVariant) -> str:
    return f"variant:{variant.bits}"


def run_matrix(trace: Trace, config: CycleConfig) -> tuple[AblationRun, ...]:
    validate_trace(trace)
    runs = tuple(
        AblationRun(variant, CycleEngine(_run_config(config), policy=_policy(variant)).run(trace))
                 for variant in all_variants())
    validate_matrix([run.variant.bits for run in runs])
    full = next(run for run in runs if run.variant.bits == "1111")
    if full.result.total_cycles != CycleEngine(
            _run_config(config), policy="variant:1111").run(trace).total_cycles:
        raise AssertionError("full GALA entry and 1111 ablation cycles differ")
    event_counts = {tuple(sorted(run.result.event_counts.items())) for run in runs}
    if len(event_counts) != 1:
        raise AssertionError("ablation variants changed the dynamic event set")
    return runs


def _run_config(config: CycleConfig) -> CycleConfig:
    """Give each variant an independent cursor for replayed memory completions."""

    clone = getattr(config.memory, "clone", None)
    if clone is None:
        return config
    return replace(config, memory=clone())
