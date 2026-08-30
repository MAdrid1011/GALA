"""Run the seven canonical mechanism configurations over one validated trace."""

from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from pathlib import Path
from typing import Callable

from gala_sim.trace import Trace, VirtualPacketArchiveReader, validate_trace
from gala_sim.timing import CycleConfig, CycleEngine, CycleProgress, CycleResult

from .matrix import AblationVariant, all_variants, validate_matrix


@dataclass(frozen=True)
class AblationRun:
    variant: AblationVariant
    result: CycleResult


_WORKER_TRACE: Trace | None = None
_WORKER_CONFIG: CycleConfig | None = None
_WORKER_ARCHIVE: Path | None = None
_WORKER_ARCHIVE_LIMITS: tuple[int, int | None, int | None, float | None] | None = None
_WORKER_ARCHIVE_PROGRESS: Callable[[AblationVariant, CycleProgress], None] | None = None


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


def run_archive_matrix(
    archive_root: Path,
    config: CycleConfig,
    *,
    max_events: int,
    max_frontier_events: int | None = None,
    max_atomic_packet_events: int | None = None,
    progress_interval_seconds: float | None = None,
    progress: Callable[[AblationRun], None] | None = None,
    cycle_progress: Callable[[AblationVariant, CycleProgress], None] | None = None,
    parallel_workers: int = 1,
) -> tuple[AblationRun, ...]:
    """Run the canonical matrix by independently replaying one compact archive."""
    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    archive_root = Path(archive_root)
    VirtualPacketArchiveReader(archive_root).validate()
    variants = all_variants()
    limits = (
        max_events, max_frontier_events, max_atomic_packet_events,
        progress_interval_seconds,
    )
    if parallel_workers == 1:
        completed = []
        for variant in variants:
            run = _run_archive_variant(
                archive_root, config, variant, limits, cycle_progress,
            )
            completed.append(run)
            if progress is not None:
                progress(run)
    else:
        if "fork" not in mp.get_all_start_methods():
            raise ValueError("parallel archive ablation requires a fork-capable platform")
        global _WORKER_CONFIG, _WORKER_ARCHIVE, _WORKER_ARCHIVE_LIMITS
        global _WORKER_ARCHIVE_PROGRESS
        _WORKER_CONFIG = config
        _WORKER_ARCHIVE = archive_root
        _WORKER_ARCHIVE_LIMITS = limits
        _WORKER_ARCHIVE_PROGRESS = cycle_progress
        context = mp.get_context("fork")
        with context.Pool(processes=parallel_workers) as pool:
            completed = []
            for run in pool.imap(_run_archive_variant_bits, (item.bits for item in variants)):
                completed.append(run)
                if progress is not None:
                    progress(run)
        _WORKER_CONFIG = None
        _WORKER_ARCHIVE = None
        _WORKER_ARCHIVE_LIMITS = None
        _WORKER_ARCHIVE_PROGRESS = None
    runs = tuple(completed)
    validate_matrix([run.variant.bits for run in runs])
    if (
        CycleEngine._selection_for_policy("full")
        != CycleEngine._selection_for_policy("variant:1111")
    ):
        raise AssertionError("full GALA entry and 1111 ablation policies differ")
    event_counts = {tuple(sorted(run.result.event_counts.items())) for run in runs}
    if len(event_counts) != 1:
        raise AssertionError("archive ablation variants changed the dynamic event set")
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


def _run_archive_variant(
    archive_root: Path,
    config: CycleConfig,
    variant: AblationVariant,
    limits: tuple[int, int | None, int | None, float | None],
    cycle_progress: Callable[[AblationVariant, CycleProgress], None] | None,
) -> AblationRun:
    (
        max_events, max_frontier_events, max_atomic_packet_events,
        progress_interval_seconds,
    ) = limits
    callback = (
        (lambda item: cycle_progress(variant, item))
        if cycle_progress is not None else None
    )
    result = VirtualPacketArchiveReader(archive_root).replay_session(
        CycleEngine(_run_config(config), policy=_policy(variant)),
        max_events=max_events,
        max_frontier_events=max_frontier_events,
        max_atomic_packet_events=max_atomic_packet_events,
        progress=callback,
        progress_interval_seconds=progress_interval_seconds,
    )
    return AblationRun(variant, result)


def _run_archive_variant_bits(bits: str) -> AblationRun:
    if (
        _WORKER_CONFIG is None
        or _WORKER_ARCHIVE is None
        or _WORKER_ARCHIVE_LIMITS is None
    ):
        raise RuntimeError("parallel archive ablation worker is not initialized")
    return _run_archive_variant(
        _WORKER_ARCHIVE,
        _WORKER_CONFIG,
        AblationVariant(bits),
        _WORKER_ARCHIVE_LIMITS,
        _WORKER_ARCHIVE_PROGRESS,
    )


def _run_config(config: CycleConfig) -> CycleConfig:
    """Give each variant an independent cursor for replayed memory completions."""

    clone = getattr(config.memory, "clone", None)
    if clone is None:
        return config
    return replace(config, memory=clone())
