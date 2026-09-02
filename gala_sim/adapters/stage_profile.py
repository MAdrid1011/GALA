"""CUDA stage timing and NVTX boundaries for the official R2-Gaussian path."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable


STAGE_SCHEMA_VERSION = "gala-r2-gpu-stage-profile-v1"


@dataclass(frozen=True)
class IterationRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start <= 0 or self.end < self.start:
            raise ValueError("GPU profile iteration range is invalid")

    def contains(self, iteration: int) -> bool:
        return self.start <= iteration <= self.end


def parse_iteration_range(value: str) -> IterationRange:
    start, separator, end = value.partition(":")
    if not separator:
        raise ValueError("GPU profile iteration range must use START:END")
    try:
        return IterationRange(int(start), int(end))
    except ValueError as error:
        raise ValueError("GPU profile iteration range must contain integers") from error


@dataclass
class _PendingMeasurement:
    iteration: int
    stage: str
    call_index: int
    start: Any
    end: Any


@dataclass
class GpuStageProfileSession:
    output: Path
    iteration_ranges: tuple[IterationRange, ...]
    run_identity: dict[str, Any]
    control_cuda_profiler: bool = False
    _patches: list[tuple[Any, str, Any]] = field(default_factory=list, init=False)
    _pending: list[_PendingMeasurement] = field(default_factory=list, init=False)
    _records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _current_iteration: int = field(default=0, init=False)
    _call_index: int = field(default=0, init=False)
    _context: str | None = field(default=None, init=False)
    _torch: Any = field(default=None, init=False)
    _original_synchronize: Callable[..., Any] | None = field(default=None, init=False)
    _iteration_start: Any | None = field(default=None, init=False)
    _iteration_call_index: int = field(default=0, init=False)
    _overhead_start: Any | None = field(default=None, init=False)
    _overhead_call_index: int = field(default=0, init=False)
    _stage_depth: int = field(default=0, init=False)
    _cuda_profiler_active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.iteration_ranges:
            raise ValueError("GPU stage profiling requires at least one iteration range")
        ordered = sorted(self.iteration_ranges, key=lambda item: (item.start, item.end))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start <= previous.end:
                raise ValueError("GPU profile iteration ranges overlap")
        self.output = self.output.resolve()

    def install(self) -> None:
        import torch
        import r2_gaussian.gaussian as gaussian_module
        from r2_gaussian.gaussian.gaussian_model import GaussianModel
        from r2_gaussian.dataset import Scene
        from r2_gaussian.utils import loss_utils

        if not torch.cuda.is_available() or not hasattr(torch.cuda, "nvtx"):
            raise RuntimeError("GPU stage profiling requires CUDA and NVTX")
        self._torch = torch
        self._patch(
            gaussian_module, "render",
            lambda original: self._wrap_query(original, "projection"),
        )
        self._patch(
            gaussian_module, "query",
            lambda original: self._wrap_query(original, "volume"),
        )
        self._patch(
            loss_utils, "l1_loss",
            lambda original: self._wrap_loss(original, "projection_loss"),
        )
        self._patch(
            loss_utils, "ssim",
            lambda original: self._wrap_loss(original, "projection_loss"),
        )
        self._patch(
            loss_utils, "tv_3d_loss",
            lambda original: self._wrap_loss(original, "volume_loss"),
        )
        self._patch(GaussianModel, "update_learning_rate", self._wrap_learning_rate)
        self._patch(GaussianModel, "training_setup", self._wrap_training_setup)
        self._patch(
            GaussianModel, "add_densification_stats",
            lambda original: self._wrap_stage(original, "adaptive_control"),
        )
        self._patch(
            GaussianModel, "densify_and_prune",
            lambda original: self._wrap_stage(original, "collection"),
        )
        self._patch(Scene, "save", self._wrap_reconstruction)
        self._patch(torch.Tensor, "backward", lambda original: self._wrap_stage(original, "backward"))
        self._original_synchronize = torch.cuda.synchronize
        self._patch(torch.cuda, "synchronize", self._wrap_synchronize)

    def install_training_aliases(self, training: Any) -> None:
        """Instrument functions imported directly by the training entrypoint."""

        self._patch(
            training, "render",
            lambda original: self._wrap_query(original, "projection"),
        )
        self._patch(
            training, "query",
            lambda original: self._wrap_query(original, "volume"),
        )
        self._patch(
            training, "l1_loss",
            lambda original: self._wrap_loss(original, "projection_loss"),
        )
        self._patch(
            training, "ssim",
            lambda original: self._wrap_loss(original, "projection_loss"),
        )
        self._patch(
            training, "tv_3d_loss",
            lambda original: self._wrap_loss(original, "volume_loss"),
        )

    def restore(self) -> None:
        if self._cuda_profiler_active:
            self._stop_cuda_profiler()
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def finish(self) -> dict[str, Any]:
        if self._original_synchronize is None or self._torch is None:
            raise RuntimeError("GPU stage profile session was not installed")
        self._close_iteration()
        self._original_synchronize()
        self._drain()
        result = self._result()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    def _patch(self, owner: Any, name: str, wrapper_factory: Callable[[Any], Any]) -> None:
        original = getattr(owner, name)
        setattr(owner, name, wrapper_factory(original))
        self._patches.append((owner, name, original))

    def _selected(self) -> bool:
        return any(item.contains(self._current_iteration) for item in self.iteration_ranges)

    def _campaign_label_suffix(self) -> str:
        campaign = self.run_identity.get("profiling_campaign")
        if not isinstance(campaign, dict):
            return ""
        digest = campaign.get("campaign_sha256")
        return f":campaign={digest}" if isinstance(digest, str) else ""

    def _uses_ncu_global_ranges(self) -> bool:
        campaign = self.run_identity.get("profiling_campaign")
        return isinstance(campaign, dict) and campaign.get("profile_tool") == "ncu"

    def _measure(self, stage: str, call: Callable[[], Any]) -> Any:
        if not self._selected():
            return call()
        if self._stage_depth == 0:
            self._close_overhead()
        torch = self._torch
        self._call_index += 1
        call_index = self._call_index
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stable_label = f"gala_stage:{stage}"
        detailed_label = (
            f"gala_stage:{stage}:iteration={self._current_iteration}:call={call_index}"
            f"{self._campaign_label_suffix()}"
        )
        start.record()
        torch.cuda.nvtx.range_push(stable_label)
        torch.cuda.nvtx.range_push(detailed_label)
        global_range = (
            torch.cuda.nvtx.range_start(
                f"gala_ncu_stage:{stage}:iteration={self._current_iteration}:"
                f"call={call_index}{self._campaign_label_suffix()}"
            )
            if self._uses_ncu_global_ranges() else None
        )
        self._stage_depth += 1
        try:
            return call()
        finally:
            self._stage_depth -= 1
            if global_range is not None:
                torch.cuda.nvtx.range_end(global_range)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            end.record()
            self._pending.append(_PendingMeasurement(
                iteration=self._current_iteration,
                stage=stage,
                call_index=call_index,
                start=start,
                end=end,
            ))
            if self._stage_depth == 0:
                self._start_overhead()

    def _drain(self) -> None:
        for item in self._pending:
            self._records.append({
                "iteration": item.iteration,
                "stage": item.stage,
                "call_index": item.call_index,
                "milliseconds": float(item.start.elapsed_time(item.end)),
            })
        self._pending.clear()

    def _wrap_stage(self, original: Any, stage: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._measure(stage, lambda: original(*args, **kwargs))
        return wrapped

    def _wrap_query(self, original: Any, query_kind: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self._context == "reconstruction":
                stage = "reconstruction"
            else:
                suffix = "forward" if self._torch.is_grad_enabled() else "evaluation"
                stage = f"{query_kind}_{suffix}"
            return self._measure(stage, lambda: original(*args, **kwargs))
        return wrapped

    def _wrap_loss(self, original: Any, training_stage: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            stage = training_stage if self._torch.is_grad_enabled() else "evaluation_metric"
            return self._measure(stage, lambda: original(*args, **kwargs))
        return wrapped

    def _wrap_reconstruction(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            previous = self._context
            self._context = "reconstruction"
            try:
                return original(*args, **kwargs)
            finally:
                self._context = previous
        return wrapped

    def _wrap_learning_rate(self, original: Any) -> Any:
        def wrapped(model: Any, iteration: int, *args: Any, **kwargs: Any) -> Any:
            self._close_iteration()
            self._current_iteration = int(iteration)
            self._start_iteration()
            return original(model, iteration, *args, **kwargs)
        return wrapped

    def _start_iteration(self) -> None:
        if not self._selected():
            return
        if self.control_cuda_profiler:
            self._start_cuda_profiler()
        self._iteration_call_index += 1
        self._iteration_start = self._torch.cuda.Event(enable_timing=True)
        self._iteration_start.record()
        self._torch.cuda.nvtx.range_push("gala_iteration:training")
        self._torch.cuda.nvtx.range_push(
            f"gala_iteration:training:iteration={self._current_iteration}:"
            f"call={self._iteration_call_index}"
            f"{self._campaign_label_suffix()}"
        )
        self._start_overhead()

    def _close_iteration(self) -> None:
        if self._iteration_start is None:
            return
        self._close_overhead()
        self._torch.cuda.nvtx.range_pop()
        self._torch.cuda.nvtx.range_pop()
        end = self._torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append(_PendingMeasurement(
            iteration=self._current_iteration,
            stage="iteration_total",
            call_index=self._iteration_call_index,
            start=self._iteration_start,
            end=end,
        ))
        self._iteration_start = None
        if self.control_cuda_profiler:
            self._stop_cuda_profiler()

    def _start_cuda_profiler(self) -> None:
        if self._cuda_profiler_active:
            raise RuntimeError("CUDA profiler capture range is already active")
        status = int(self._torch.cuda.cudart().cudaProfilerStart())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStart failed with status {status}")
        self._cuda_profiler_active = True

    def _stop_cuda_profiler(self) -> None:
        if not self._cuda_profiler_active:
            return
        status = int(self._torch.cuda.cudart().cudaProfilerStop())
        if status != 0:
            raise RuntimeError(f"cudaProfilerStop failed with status {status}")
        self._cuda_profiler_active = False

    def _start_overhead(self) -> None:
        if self._iteration_start is None or self._overhead_start is not None:
            return
        self._overhead_call_index += 1
        self._overhead_start = self._torch.cuda.Event(enable_timing=True)
        self._overhead_start.record()
        self._torch.cuda.nvtx.range_push("gala_stage:iteration_overhead")
        self._torch.cuda.nvtx.range_push(
            f"gala_stage:iteration_overhead:iteration={self._current_iteration}:"
            f"call={self._overhead_call_index}"
            f"{self._campaign_label_suffix()}"
        )

    def _close_overhead(self) -> None:
        if self._overhead_start is None:
            return
        self._torch.cuda.nvtx.range_pop()
        self._torch.cuda.nvtx.range_pop()
        end = self._torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append(_PendingMeasurement(
            iteration=self._current_iteration,
            stage="iteration_overhead",
            call_index=self._overhead_call_index,
            start=self._overhead_start,
            end=end,
        ))
        self._overhead_start = None

    def _wrap_training_setup(self, original: Any) -> Any:
        def wrapped(model: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(model, *args, **kwargs)
            optimizer = model.optimizer
            self._patch(
                optimizer, "step",
                lambda step: self._wrap_stage(step, "optimizer"),
            )
            self._patch(
                optimizer, "zero_grad",
                lambda zero_grad: self._wrap_stage(zero_grad, "optimizer"),
            )
            return result
        return wrapped

    def _wrap_synchronize(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._drain()
            return result
        return wrapped

    def _result(self) -> dict[str, Any]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in self._records:
            grouped[str(record["stage"])].append(float(record["milliseconds"]))
        summaries = {
            stage: {
                "call_count": len(values),
                "total_ms": float(sum(values)),
                "median_ms": float(median(values)),
                "minimum_ms": float(min(values)),
                "maximum_ms": float(max(values)),
            }
            for stage, values in sorted(grouped.items())
        }
        iteration_totals = {
            int(record["iteration"]): float(record["milliseconds"])
            for record in self._records if record["stage"] == "iteration_total"
        }
        attributed_by_iteration: dict[int, float] = defaultdict(float)
        for record in self._records:
            if record["stage"] != "iteration_total":
                attributed_by_iteration[int(record["iteration"])] += float(
                    record["milliseconds"]
                )
        coverage_records = []
        for iteration, total_ms in sorted(iteration_totals.items()):
            attributed_ms = attributed_by_iteration[iteration]
            residual_ms = total_ms - attributed_ms
            coverage_records.append({
                "iteration": iteration,
                "iteration_total_ms": total_ms,
                "attributed_stage_ms": attributed_ms,
                "unattributed_ms": residual_ms,
                "attributed_fraction": attributed_ms / total_ms if total_ms > 0 else 0.0,
            })
        negative_residual_count = sum(
            record["unattributed_ms"] < 0 for record in coverage_records
        )
        total_iteration_ms = sum(record["iteration_total_ms"] for record in coverage_records)
        total_attributed_ms = sum(record["attributed_stage_ms"] for record in coverage_records)
        coverage = {
            "status": "failed_preflight",
            "iteration_count": len(coverage_records),
            "iteration_total_ms": total_iteration_ms,
            "attributed_stage_ms": total_attributed_ms,
            "unattributed_ms": total_iteration_ms - total_attributed_ms,
            "attributed_fraction": (
                total_attributed_ms / total_iteration_ms if total_iteration_ms > 0 else 0.0
            ),
            "negative_residual_count": negative_residual_count,
            "records": coverage_records,
        }
        if coverage_records and negative_residual_count == 0:
            coverage["status"] = (
                "complete"
                if math.isclose(
                    total_attributed_ms, total_iteration_ms,
                    rel_tol=1e-6, abs_tol=1e-3,
                ) else "partial"
            )
        coverage["complete"] = coverage["status"] == "complete"
        observed_iterations = set(iteration_totals)
        requested_iteration_count = sum(
            item.end - item.start + 1 for item in self.iteration_ranges
        )
        missing_iterations = [
            iteration
            for item in self.iteration_ranges
            for iteration in range(item.start, item.end + 1)
            if iteration not in observed_iterations
        ]
        requested_coverage = {
            "status": "complete" if not missing_iterations else "incomplete",
            "requested_iteration_count": requested_iteration_count,
            "observed_iteration_count": requested_iteration_count - len(missing_iterations),
            "missing_iterations": missing_iterations,
        }
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
            "status": (
                "passed" if requested_coverage["status"] == "complete"
                else "failed_preflight"
            ),
            "result_scope": "gpu_stage_characterization",
            "formal_performance_eligible": False,
            "quality_eligible": False,
            "iteration_ranges": [
                {"start": item.start, "end": item.end}
                for item in self.iteration_ranges
            ],
            "run_identity": self.run_identity,
            "stage_summaries": summaries,
            "stage_records": self._records,
            "stage_coverage": coverage,
            "requested_iteration_coverage": requested_coverage,
            "kernel_inventory": {
                "status": "pending_nsys_or_ncu",
                "required_fields": [
                    "kernel_names", "dram_read_bytes", "dram_write_bytes",
                    "fp32_operations", "transcendental_operations",
                    "atomic_operations", "launch_count",
                ],
            },
        }


def ranges_as_strings(ranges: Iterable[IterationRange]) -> list[str]:
    return [f"{item.start}:{item.end}" for item in ranges]
