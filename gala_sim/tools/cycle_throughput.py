"""Runtime throughput diagnostics for long cycle replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from gala_sim.config import GalaConfig
from gala_sim.timing.engine import CycleProgress


@dataclass(frozen=True)
class ThroughputDiagnosticConfig:
    report_interval_events: int
    report_interval_seconds: float
    warmup_samples: int
    stability_window_samples: int
    required_consecutive_stable_windows: int
    stability_relative_span: float
    minimum_completion_fraction: float

    def __post_init__(self) -> None:
        if min(
            self.report_interval_events,
            self.warmup_samples,
            self.stability_window_samples,
            self.required_consecutive_stable_windows,
        ) <= 0:
            raise ValueError("throughput diagnostic counts must be positive")
        if not math.isfinite(self.report_interval_seconds) or self.report_interval_seconds <= 0:
            raise ValueError("throughput diagnostic interval seconds must be positive")
        if not 0 < self.stability_relative_span < 1:
            raise ValueError("throughput stability relative span must be in (0, 1)")
        if not 0 < self.minimum_completion_fraction <= 1:
            raise ValueError("throughput minimum completion fraction must be in (0, 1]")

    @classmethod
    def from_gala(cls, config: GalaConfig) -> "ThroughputDiagnosticConfig":
        return cls(
            report_interval_events=int(
                config.value("diagnostic.throughput_report_interval_events")
            ),
            report_interval_seconds=float(
                config.value("diagnostic.throughput_report_interval_seconds")
            ),
            warmup_samples=int(config.value("diagnostic.throughput_warmup_samples")),
            stability_window_samples=int(
                config.value("diagnostic.throughput_stability_window_samples")
            ),
            required_consecutive_stable_windows=int(
                config.value("diagnostic.throughput_required_stable_windows")
            ),
            stability_relative_span=float(
                config.value("diagnostic.throughput_stability_relative_span")
            ),
            minimum_completion_fraction=float(
                config.value("diagnostic.throughput_minimum_completion_fraction")
            ),
        )


@dataclass(frozen=True)
class ThroughputSample:
    completed_events: int
    total_events: int
    completion_fraction: float
    completed_iterations: int
    total_iterations: int
    last_completed_iteration: int | None
    iteration_completion_fraction: float
    simulated_cycles: int
    elapsed_seconds: float
    cumulative_events_per_second: float
    interval_events_per_second: float | None
    interval_cycles_per_event: float | None
    projected_total_cycles: float
    projected_asic_seconds: float | None


@dataclass(frozen=True)
class RuntimeThroughputSample:
    completed_events: int
    total_events: int
    elapsed_seconds: float
    interval_events_per_second: float | None
    completion_fraction: float


@dataclass(frozen=True)
class ArchiveSpeedupSample:
    iteration_id: int
    completed_iterations: int
    total_iterations: int
    completion_fraction: float
    completed_events: int
    cycles_by_variant: dict[str, int]
    cumulative_speedup_vs_base: dict[str, float]
    interval_speedup_vs_base: dict[str, float]
    projected_total_cycles: dict[str, float]


class ThroughputConverged(RuntimeError):
    """Raised only when an explicitly enabled diagnostic may stop early."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__("cycle throughput projection converged")
        self.report = report


class ThroughputMonitor:
    def __init__(
        self, config: ThroughputDiagnosticConfig, *, stop_when_stable: bool = False,
        clock_frequency_hz: int | None = None,
    ) -> None:
        self.config = config
        self.stop_when_stable = stop_when_stable
        if clock_frequency_hz is not None and clock_frequency_hz <= 0:
            raise ValueError("clock frequency must be positive")
        self.clock_frequency_hz = clock_frequency_hz
        self.samples: list[ThroughputSample] = []
        self.runtime_samples: list[RuntimeThroughputSample] = []
        self._consecutive_stable_windows = 0
        self._last_stability_sample_count = 0
        self._last_runtime_events = -1
        self._last_runtime_elapsed = -1.0
        self._last_iteration_sample = -1
        self._last_simulated_cycles = -1
        self._total_iterations: int | None = None

    def observe(self, progress: CycleProgress) -> dict[str, Any]:
        if progress.phase != "replay":
            return self.report()
        if progress.completed_events <= 0 or progress.total_events <= 0:
            return self.report()
        if progress.completed_events > progress.total_events:
            raise ValueError("completed events exceed total events")
        if progress.completed_iterations < 0 or progress.completed_iterations > progress.total_iterations:
            raise ValueError("completed iterations exceed total iterations")
        if progress.simulated_cycles < self._last_simulated_cycles:
            raise ValueError("simulated cycles are not monotonic")
        if self._total_iterations is None:
            self._total_iterations = progress.total_iterations
        elif progress.total_iterations != self._total_iterations:
            raise ValueError("total iteration count changed during replay")
        if self.runtime_samples and progress.total_events != self.runtime_samples[0].total_events:
            raise ValueError("total event count changed during replay")
        elapsed = max(progress.elapsed_seconds, float.fromhex("0x1.0p-52"))
        if progress.completed_events < self._last_runtime_events or elapsed < self._last_runtime_elapsed:
            raise ValueError("progress is not monotonic")
        previous_runtime = self.runtime_samples[-1] if self.runtime_samples else None
        interval_rate: float | None = None
        completed_delta = progress.completed_events - self._last_runtime_events
        if previous_runtime is not None:
            elapsed_delta = elapsed - previous_runtime.elapsed_seconds
            if elapsed_delta > 0 and completed_delta > 0:
                interval_rate = completed_delta / elapsed_delta
        if progress.completed_events != self._last_runtime_events or elapsed != self._last_runtime_elapsed:
            self.runtime_samples.append(RuntimeThroughputSample(
                completed_events=progress.completed_events,
                total_events=progress.total_events,
                elapsed_seconds=progress.elapsed_seconds,
                interval_events_per_second=interval_rate,
                completion_fraction=progress.completed_events / progress.total_events,
            ))
            self._last_runtime_events = progress.completed_events
            self._last_runtime_elapsed = elapsed
            self._last_simulated_cycles = progress.simulated_cycles
        if progress.completed_iterations <= 0 or progress.total_iterations <= 0:
            return self.report()
        boundary_iteration = progress.last_completed_iteration
        if boundary_iteration is None or boundary_iteration <= self._last_iteration_sample:
            return self.report()
        boundary_events = progress.last_completed_iteration_events or progress.completed_events
        boundary_cycles = progress.last_completed_iteration_cycles or progress.simulated_cycles
        boundary_elapsed = (
            progress.last_completed_iteration_elapsed_seconds
            if progress.last_completed_iteration_elapsed_seconds is not None
            else progress.elapsed_seconds
        )
        previous = self.samples[-1] if self.samples else None
        interval_cycles_per_event: float | None = None
        iteration_interval_rate: float | None = None
        if previous is not None:
            cycle_delta = boundary_cycles - previous.simulated_cycles
            event_delta = boundary_events - previous.completed_events
            if cycle_delta > 0 and event_delta > 0:
                interval_cycles_per_event = cycle_delta / event_delta
            elapsed_delta = boundary_elapsed - previous.elapsed_seconds
            if elapsed_delta > 0 and event_delta > 0:
                iteration_interval_rate = event_delta / elapsed_delta
        completion_fraction = boundary_events / progress.total_events
        iteration_completion_fraction = (
            progress.completed_iterations / progress.total_iterations
        )
        projected_total_cycles = (
            boundary_cycles / (progress.completed_iterations / progress.total_iterations)
        )
        projected_asic_seconds = (
            projected_total_cycles / self.clock_frequency_hz
            if self.clock_frequency_hz is not None else None
        )
        sample = ThroughputSample(
            completed_events=boundary_events,
            total_events=progress.total_events,
            completion_fraction=completion_fraction,
            completed_iterations=progress.completed_iterations,
            total_iterations=progress.total_iterations,
            last_completed_iteration=progress.last_completed_iteration,
            iteration_completion_fraction=iteration_completion_fraction,
            simulated_cycles=boundary_cycles,
            elapsed_seconds=boundary_elapsed,
            cumulative_events_per_second=boundary_events / max(
                boundary_elapsed, float.fromhex("0x1.0p-52")
            ),
            interval_events_per_second=iteration_interval_rate,
            interval_cycles_per_event=interval_cycles_per_event,
            projected_total_cycles=projected_total_cycles,
            projected_asic_seconds=projected_asic_seconds,
        )
        self.samples.append(sample)
        self._last_iteration_sample = boundary_iteration
        report = self.report()
        if (
            self.stop_when_stable
            and report["stability"]["status"] == "stable"
            and progress.completed_events < progress.total_events
        ):
            raise ThroughputConverged(report)
        return report

    def report(self) -> dict[str, Any]:
        usable = self.samples[self.config.warmup_samples:]
        window = usable[-self.config.stability_window_samples:]
        interval_rates = [
            sample.interval_events_per_second for sample in window
            if sample.interval_events_per_second is not None
        ]
        cycle_rates = [
            sample.interval_cycles_per_event for sample in window
            if sample.interval_cycles_per_event is not None
        ]
        projections = [sample.projected_total_cycles for sample in window]
        enough_samples = (
            len(window) == self.config.stability_window_samples
            and len(interval_rates) == self.config.stability_window_samples
            and len(cycle_rates) == self.config.stability_window_samples
        )
        enough_work = bool(
            self.samples
            and self.samples[-1].iteration_completion_fraction
            >= self.config.minimum_completion_fraction
        )
        throughput_span = self._relative_span(interval_rates) if enough_samples else None
        cycle_rate_span = self._relative_span(cycle_rates) if enough_samples else None
        projection_span = self._relative_span(projections) if enough_samples else None
        stable = bool(
            enough_samples
            and enough_work
            and throughput_span is not None
            and cycle_rate_span is not None
            and projection_span is not None
            and throughput_span <= self.config.stability_relative_span
            and cycle_rate_span <= self.config.stability_relative_span
            and projection_span <= self.config.stability_relative_span
        )
        if len(self.samples) != self._last_stability_sample_count:
            if stable:
                self._consecutive_stable_windows += 1
            else:
                self._consecutive_stable_windows = 0
            self._last_stability_sample_count = len(self.samples)
        stable = (
            stable
            and self._consecutive_stable_windows
            >= self.config.required_consecutive_stable_windows
        )
        projection_interval = (
            {"low": min(projections), "high": max(projections)}
            if projections else None
        )
        return {
            "schema_version": "gala-cycle-throughput-diagnostic-v1",
            "result_scope": "development_throughput_projection",
            "formal_performance_eligible": False,
            "complete_trace_replay": bool(
                self.samples and self.samples[-1].completed_events == self.samples[-1].total_events
            ),
            "configuration": asdict(self.config),
            "runtime_samples": [asdict(sample) for sample in self.runtime_samples],
            "samples": [asdict(sample) for sample in self.samples],
            "stability": {
                "status": "stable" if stable else "collecting",
                "enough_samples": enough_samples,
                "enough_work": enough_work,
                "throughput_relative_span": throughput_span,
                "cycles_per_event_relative_span": cycle_rate_span,
                "projection_relative_span": projection_span,
                "consecutive_stable_windows": self._consecutive_stable_windows,
                "projection_interval_cycles": projection_interval,
            },
        }

    @staticmethod
    def _relative_span(values: list[float]) -> float:
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("throughput stability values must be finite and positive")
        mean = sum(values) / len(values)
        return (max(values) - min(values)) / mean


class ArchiveSpeedupMonitor:
    """Judge convergence only at common, quiescent iteration boundaries."""

    def __init__(
        self,
        config: ThroughputDiagnosticConfig,
        *,
        variants: tuple[str, ...],
        base_variant: str = "0000",
    ) -> None:
        if not variants or len(set(variants)) != len(variants):
            raise ValueError("speedup diagnostic variants must be unique")
        if base_variant not in variants:
            raise ValueError("speedup diagnostic base variant is absent")
        self.config = config
        self.variants = variants
        self.base_variant = base_variant
        self.samples: list[ArchiveSpeedupSample] = []
        self._consecutive_stable_windows = 0
        self._last_stability_sample_count = 0

    def observe(
        self,
        *,
        iteration_id: int,
        completed_iterations: int,
        total_iterations: int,
        completed_events: int,
        cycles_by_variant: dict[str, int],
    ) -> dict[str, Any]:
        if completed_iterations <= 0 or total_iterations <= 0:
            raise ValueError("speedup diagnostic iteration counts must be positive")
        if completed_iterations > total_iterations:
            raise ValueError("completed iterations exceed total iterations")
        if completed_events <= 0:
            raise ValueError("completed events must be positive")
        if set(cycles_by_variant) != set(self.variants):
            raise ValueError("speedup diagnostic variant set changed")
        normalized_cycles = {
            bits: int(cycles_by_variant[bits]) for bits in self.variants
        }
        if any(value <= 0 for value in normalized_cycles.values()):
            raise ValueError("speedup diagnostic cycles must be positive")
        previous = self.samples[-1] if self.samples else None
        if previous is not None:
            if iteration_id <= previous.iteration_id:
                raise ValueError("speedup diagnostic iterations are not monotonic")
            if completed_iterations <= previous.completed_iterations:
                raise ValueError("completed iteration count is not monotonic")
            if completed_events <= previous.completed_events:
                raise ValueError("completed event count is not monotonic")
            if total_iterations != previous.total_iterations:
                raise ValueError("total iteration count changed")
            if any(
                normalized_cycles[bits] <= previous.cycles_by_variant[bits]
                for bits in self.variants
            ):
                raise ValueError("variant cycles are not monotonic")
        base_cycles = normalized_cycles[self.base_variant]
        cumulative_speedups = {
            bits: base_cycles / normalized_cycles[bits]
            for bits in self.variants
        }
        interval_speedups: dict[str, float] = {}
        if previous is not None:
            base_delta = base_cycles - previous.cycles_by_variant[self.base_variant]
            if base_delta <= 0:
                raise ValueError("Base ASIC interval cycles must be positive")
            interval_speedups = {
                bits: base_delta / (
                    normalized_cycles[bits] - previous.cycles_by_variant[bits]
                )
                for bits in self.variants
            }
        completion_fraction = completed_iterations / total_iterations
        sample = ArchiveSpeedupSample(
            iteration_id=int(iteration_id),
            completed_iterations=int(completed_iterations),
            total_iterations=int(total_iterations),
            completion_fraction=completion_fraction,
            completed_events=int(completed_events),
            cycles_by_variant=normalized_cycles,
            cumulative_speedup_vs_base=cumulative_speedups,
            interval_speedup_vs_base=interval_speedups,
            projected_total_cycles={
                bits: cycles / completion_fraction
                for bits, cycles in normalized_cycles.items()
            },
        )
        self.samples.append(sample)
        return self.report()

    def report(self) -> dict[str, Any]:
        usable = self.samples[self.config.warmup_samples:]
        window = usable[-self.config.stability_window_samples:]
        enough_samples = (
            len(window) == self.config.stability_window_samples
            and all(sample.interval_speedup_vs_base for sample in window)
        )
        enough_work = bool(
            self.samples
            and self.samples[-1].completion_fraction
            >= self.config.minimum_completion_fraction
        )
        compared_variants = tuple(
            bits for bits in self.variants if bits != self.base_variant
        )
        cumulative_spans: dict[str, float] = {}
        interval_spans: dict[str, float] = {}
        projection_spans: dict[str, float] = {}
        if enough_samples:
            cumulative_spans = {
                bits: ThroughputMonitor._relative_span([
                    sample.cumulative_speedup_vs_base[bits] for sample in window
                ])
                for bits in compared_variants
            }
            interval_spans = {
                bits: ThroughputMonitor._relative_span([
                    sample.interval_speedup_vs_base[bits] for sample in window
                ])
                for bits in compared_variants
            }
            projection_spans = {
                bits: ThroughputMonitor._relative_span([
                    sample.projected_total_cycles[bits] for sample in window
                ])
                for bits in self.variants
            }
        span_limit = self.config.stability_relative_span
        stable_window = bool(
            enough_samples
            and enough_work
            and all(value <= span_limit for value in cumulative_spans.values())
            and all(value <= span_limit for value in interval_spans.values())
            and all(value <= span_limit for value in projection_spans.values())
        )
        if len(self.samples) != self._last_stability_sample_count:
            if stable_window:
                self._consecutive_stable_windows += 1
            else:
                self._consecutive_stable_windows = 0
            self._last_stability_sample_count = len(self.samples)
        stable = (
            stable_window
            and self._consecutive_stable_windows
            >= self.config.required_consecutive_stable_windows
        )
        return {
            "schema_version": "gala-archive-speedup-diagnostic-v1",
            "result_scope": "development_speedup_projection",
            "formal_performance_eligible": False,
            "base_variant": self.base_variant,
            "variants": list(self.variants),
            "complete_trace_replay": bool(
                self.samples and self.samples[-1].completed_iterations
                == self.samples[-1].total_iterations
            ),
            "configuration": asdict(self.config),
            "samples": [asdict(sample) for sample in self.samples],
            "stability": {
                "status": "stable" if stable else "collecting",
                "enough_samples": enough_samples,
                "enough_work": enough_work,
                "cumulative_speedup_relative_span": cumulative_spans,
                "interval_speedup_relative_span": interval_spans,
                "projected_cycles_relative_span": projection_spans,
                "consecutive_stable_windows": self._consecutive_stable_windows,
            },
        }


def require_empty_diagnostic_output(path: Path) -> None:
    """Prevent an early-stop diagnostic from mixing with an older run."""

    if path.exists() and (not path.is_dir() or next(path.iterdir(), None) is not None):
        raise ValueError("early-stop diagnostic output must be absent or empty")
