"""R²-Gaussian official-entry adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from gala_sim.config import GalaConfig
from gala_sim.identity import sha256_tree
from gala_sim.metrics import QualityConfig, measure_quality
from gala_sim.trace import DeviceTraceSink, Trace, TraceReader, TraceWriter

from .chest import load_chest_manifest
from .protocol import PreparedRun, ReferenceArtifact, TraceArtifact


class TraceCaptureUnavailable(RuntimeError):
    """Raised until the locked CUDA extension exposes its real event buffers."""


def _repository_commit(root: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise TraceCaptureUnavailable("GALA repository commit is unavailable") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise TraceCaptureUnavailable("GALA repository commit identity is invalid")
    return commit


def _bind_trace_identity(
    trace: Trace,
    run: PreparedRun,
    model_commit: str,
    repository_commit: str,
) -> Trace:
    """Attach the frozen run identity before a trace reaches cycle replay."""

    return Trace(
        trace.events,
        trace.dependencies,
        trace.payload,
        {
            **trace.metadata,
            "config_sha256": run.config_sha256,
            "model_commit": model_commit,
            "model": run.model_name,
            "dataset": run.dataset_name,
            "dataset_manifest_sha256": sha256_tree(run.dataset_root),
            "repository_commit": repository_commit,
        },
    )


@dataclass(frozen=True)
class R2GaussianChestAdapter:
    source_root: Path
    dataset_root: Path
    output_root: Path
    model_commit: str
    python_executable: Path | None = None

    def prepare(self, dataset: Any, config: GalaConfig) -> PreparedRun:
        if not self.source_root.is_dir() or not self.dataset_root.is_dir():
            raise FileNotFoundError("R²-Gaussian source or Chest root is unavailable")
        return PreparedRun(
            model_name="R2-Gaussian",
            dataset_name="Chest",
            source_root=self.source_root.resolve(),
            dataset_root=self.dataset_root.resolve(),
            config_sha256=config.sha256,
            quality_config=QualityConfig.from_gala(config),
            seed=0,
            official_command=(
                str(self.python_executable.resolve()) if self.python_executable is not None else "python",
                "train.py", "-s", str(self.dataset_root), "-m", str(self.output_root),
            ),
        )

    def run_reference(self, run: PreparedRun) -> ReferenceArtifact:
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        completed = subprocess.run(
            list(run.official_command),
            cwd=run.source_root,
            check=False,
            text=True,
            capture_output=True,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(
                "official R²-Gaussian command failed: "
                + (completed.stderr[-2000:] or completed.stdout[-2000:])
            )
        volume_candidates = sorted(self.output_root.glob("point_cloud/iteration_*/vol_pred.npy"))
        if not volume_candidates:
            raise RuntimeError("official R²-Gaussian run produced no reconstructed volume")
        return ReferenceArtifact(
            output_root=self.output_root,
            volume_path=volume_candidates[-1],
            metrics=_read_metrics(run, volume_candidates[-1], self.output_root),
            gpu_reference={
                "wall_seconds": elapsed,
                "command": list(run.official_command),
                "stdout_sha256": _text_sha256(completed.stdout),
                "stderr_sha256": _text_sha256(completed.stderr),
            },
        )

    def capture_trace(self, run: PreparedRun, sink: DeviceTraceSink) -> TraceArtifact:
        trace_root = self.output_root / "trace"
        repository_root = Path(__file__).resolve().parents[2]
        repository_commit = _repository_commit(repository_root)
        command = (
            run.official_command[0], "-m", "gala_sim.adapters.trace_runner",
            "--trace-output", str(trace_root), str(run.source_root / "train.py"),
            *run.official_command[2:],
        )
        environment = os.environ.copy()
        python_path = str(repository_root)
        if environment.get("PYTHONPATH"):
            python_path += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = python_path
        started = time.monotonic()
        completed = subprocess.run(
            list(command), cwd=run.source_root, env=environment, check=False,
            text=True, capture_output=True,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise TraceCaptureUnavailable(
                "official trace-enabled run failed: "
                + (completed.stderr[-3000:] or completed.stdout[-3000:])
            )
        try:
            trace = TraceReader().read(trace_root, mmap_mode="r")
        except (OSError, ValueError, RuntimeError) as error:
            raise TraceCaptureUnavailable(f"captured trace failed validation: {error}") from error
        trace = _bind_trace_identity(trace, run, self.model_commit, repository_commit)
        TraceWriter().write(trace, trace_root, validate=True)
        try:
            _push_trace_chunks(trace, sink)
        except (BufferError, RuntimeError, ValueError) as error:
            raise TraceCaptureUnavailable(f"trace sink handoff failed: {error}") from error
        reference = self._reference_from_output(
            run,
            gpu_reference={
                "wall_seconds": elapsed,
                "command": list(command),
                "stdout_sha256": _text_sha256(completed.stdout),
                "stderr_sha256": _text_sha256(completed.stderr),
                "trace_event_count": trace.event_count,
            }
        )
        return TraceArtifact(trace_root=trace_root, trace=trace, reference=reference)

    def replay_reductions(self, run: PreparedRun, order: Any) -> ReferenceArtifact:
        raise TraceCaptureUnavailable("functional replay requires a validated trace and reduction order")

    def _reference_from_output(
        self, run: PreparedRun, *, gpu_reference: dict[str, Any]
    ) -> ReferenceArtifact:
        volume_candidates = sorted(self.output_root.glob("point_cloud/iteration_*/vol_pred.npy"))
        if not volume_candidates:
            raise TraceCaptureUnavailable("trace-enabled run produced no reconstructed volume")
        return ReferenceArtifact(
            output_root=self.output_root,
            volume_path=volume_candidates[-1],
            metrics=_read_metrics(run, volume_candidates[-1], self.output_root),
            gpu_reference=gpu_reference,
        )


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _push_trace_chunks(trace: Any, sink: DeviceTraceSink) -> None:
    """Transfer validated trace chunks with offsets local to each sink submission."""
    chunk_events = int(getattr(sink, "chunk_events", trace.event_count or 1))
    if chunk_events <= 0:
        raise ValueError("trace sink chunk capacity must be positive")
    try:
        for start in range(0, trace.event_count, chunk_events):
            end = min(start + chunk_events, trace.event_count)
            events = trace.events[start:end].copy()
            dependencies: list[int] = []
            payload: list[float] = []
            for row in events:
                dep_begin = int(row["dependency_begin"])
                dep_end = dep_begin + int(row["dependency_count"])
                payload_begin = int(row["payload_offset"])
                payload_end = payload_begin + int(row["payload_length"])
                row["dependency_begin"] = len(dependencies)
                row["payload_offset"] = len(payload)
                dependencies.extend(int(value) for value in trace.dependencies[dep_begin:dep_end])
                payload.extend(float(value) for value in trace.payload[payload_begin:payload_end])
            sink.push(
                events,
                np.asarray(dependencies, dtype=np.dtype("<u8")),
                np.asarray(payload, dtype=np.dtype("<f4")),
            )
        if trace.event_count == 0:
            sink.push(
                np.empty(0, dtype=trace.events.dtype),
                np.empty(0, dtype=np.dtype("<u8")),
                np.empty(0, dtype=np.dtype("<f4")),
            )
    finally:
        sink.close()


def _read_latest_metrics(root: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    candidates = sorted(root.glob("eval/iter_*/eval3d.yml"))
    if not candidates:
        return metrics
    try:
        import yaml
    except ImportError:
        return metrics
    try:
        data = yaml.safe_load(candidates[-1].read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return metrics
    if isinstance(data, dict):
        for key in ("psnr_3d", "ssim_3d"):
            if isinstance(data.get(key), (int, float)):
                metrics[key] = float(data[key])
    return metrics


def _read_metrics(run: PreparedRun, volume_path: Path, output_root: Path) -> dict[str, float]:
    dataset = load_chest_manifest(run.dataset_root)
    reference = np.load(dataset.volume_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(volume_path, mmap_mode="r", allow_pickle=False)
    quality = measure_quality(reference, candidate, run.quality_config)
    return {
        "psnr": quality.psnr,
        "ssim": quality.ssim,
        "lpips": quality.lpips,
        **_read_latest_metrics(output_root),
    }
