"""Run the seven canonical mechanism configurations over one validated trace."""

from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from typing import Callable

from gala_sim.trace import Trace, validate_trace
from gala_sim.timing import CycleConfig, CycleEngine, CycleResult

from .matrix import AblationVariant, all_variants, validate_matrix


@dataclass(frozen=True)
class AblationRun:
    variant: AblationVariant
    result: CycleResult


_WORKER_TRACE: Trace | None = None
_WORKER_CONFIG: CycleConfig | None = None


def _policy(variant: AblationVariant) -> str:
    return f"variant:{variant.bits}"


def run_matrix(
    trace: Trace,
    config: CycleConfig,
    *,
    progress: Callable[[AblationRun], None] | None = None,
    parallel_workers: int = 1,
) -> tuple[AblationRun, ...]:
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    validate_trace(trace)
    variants = all_variants()
    if parallel_workers == 1:
        completed: list[AblationRun] = []
        for variant in variants:
            run = _run_variant(trace, config, variant)
            completed.append(run)
            if progress is not None:
                progress(run)
    else:
        if "fork" not in mp.get_all_start_methods():
            raise ValueError("parallel ablation requires a fork-capable platform")
        global _WORKER_TRACE, _WORKER_CONFIG
        _WORKER_TRACE = trace
        _WORKER_CONFIG = config
        context = mp.get_context("fork")
        with context.Pool(processes=parallel_workers) as pool:
            completed = []
            for run in pool.imap(_run_variant_bits, (item.bits for item in variants)):
                completed.append(run)
                if progress is not None:
                    progress(run)
        _WORKER_TRACE = None
        _WORKER_CONFIG = None
    runs = tuple(completed)
    validate_matrix([run.variant.bits for run in runs])
    full = next(run for run in runs if run.variant.bits == "1111")
    # ``full`` is a policy alias for the 1111 variant.  Compare the frozen
    # selections directly instead of launching a redundant seventh replay
    # over the complete trace merely to re-check that alias.
    if (
        CycleEngine._selection_for_policy("full")
        != CycleEngine._selection_for_policy("variant:1111")
    ):
        raise AssertionError("full GALA entry and 1111 ablation policies differ")
    event_counts = {tuple(sorted(run.result.event_counts.items())) for run in runs}
    if len(event_counts) != 1:
        raise AssertionError("ablation variants changed the dynamic event set")
    return runs


def _run_variant(trace: Trace, config: CycleConfig, variant: AblationVariant) -> AblationRun:
    return AblationRun(
        variant,
        CycleEngine(_run_config(config), policy=_policy(variant)).run(
            trace, validate_input=False
        ),
    )


def _run_variant_bits(bits: str) -> AblationRun:
    if _WORKER_TRACE is None or _WORKER_CONFIG is None:
        raise RuntimeError("parallel ablation worker is not initialized")
    return _run_variant(_WORKER_TRACE, _WORKER_CONFIG, AblationVariant(bits))


def _run_config(config: CycleConfig) -> CycleConfig:
    """Give each variant an independent cursor for replayed memory completions."""

    clone = getattr(config.memory, "clone", None)
    if clone is None:
        return config
    return replace(config, memory=clone())
