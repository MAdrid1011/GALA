"""R²-Gaussian official-entry adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from gala_sim.config import GalaConfig
from gala_sim.identity import sha256_tree
from gala_sim.metrics import QualityConfig, measure_quality
from gala_sim.trace import DeviceTraceSink, Trace, TraceReader, TraceWriter

from .chest import load_chest_manifest
from .protocol import PreparedRun, ReferenceArtifact, TraceArtifact
from .trace_process import run_trace_process


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
class _R2GaussianAdapterBase:
    source_root: Path
    dataset_root: Path
    output_root: Path
    model_commit: str
    python_executable: Path | None = None
    descriptor: Any | None = None

    def prepare(self, dataset: Any, config: GalaConfig) -> PreparedRun:
        dataset_root = Path(getattr(dataset, "root", self.dataset_root)).resolve()
        dataset_name = str(getattr(dataset, "id", "chest"))
        if not self.source_root.is_dir() or not dataset_root.is_dir():
            raise FileNotFoundError(f"R²-Gaussian dataset root is unavailable: {dataset_root}")
        return PreparedRun(
            model_name="R2-Gaussian",
            dataset_name=dataset_name,
            source_root=self.source_root.resolve(),
            dataset_root=dataset_root,
            config_sha256=config.sha256,
            quality_config=QualityConfig.from_gala(config),
            seed=0,
            official_command=(
                str(self.python_executable.resolve()) if self.python_executable is not None else "python",
                "train.py", "-s", str(dataset_root), "-m", str(self.output_root),
            ),
            output_root=self.output_root.resolve(),
        )

    def run_reference(self, run: PreparedRun) -> ReferenceArtifact:
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        repository_root = Path(__file__).resolve().parents[2]
        command = (
            run.official_command[0], "-m", "gala_sim.adapters.reference_runner",
            "--model-id", "r2_gaussian", str(run.source_root / "train.py"),
            *run.official_command[2:],
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(repository_root), environment.get("PYTHONPATH")) if item
        )
        completed = subprocess.run(
            list(command),
            cwd=run.source_root,
            env=environment,
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
                "execution_command": list(command),
                "stdout_sha256": _text_sha256(completed.stdout),
                "stderr_sha256": _text_sha256(completed.stderr),
            },
        )

    def preflight_environment(self, environment: dict[str, str] | None = None) -> None:
        """Verify R² imports with CUDA hidden before launching official training."""

        imports = (
            "torch",
            "r2_gaussian.gaussian.gaussian_model",
            "xray_gaussian_rasterization_voxelization",
        )
        check_environment = dict(environment or os.environ)
        check_environment["CUDA_VISIBLE_DEVICES"] = ""
        script = (
            "import importlib;"
            + ";".join(f"importlib.import_module({name!r})" for name in imports)
        )
        executable = str(self.python_executable or Path(sys.executable))
        try:
            completed = subprocess.run(
                (executable, "-c", script), cwd=self.source_root,
                env=check_environment, check=False, text=True,
                capture_output=True, timeout=60.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TraceCaptureUnavailable("R²-Gaussian environment preflight failed") from error
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise TraceCaptureUnavailable(
                "R²-Gaussian environment preflight failed" + suffix
            )


class R2GaussianAdapter(_R2GaussianAdapterBase):
    """Official R²-Gaussian adapter for every prepared GALA dataset.

    The historical ``R2GaussianChestAdapter`` name remains as a compatibility
    alias.  Preparation is dataset-driven, so Walnut and HDTomo-USB use the
    same command, trace identity, and output discovery contract as Chest.
    """

    def capture_trace(self, run: PreparedRun, sink: DeviceTraceSink) -> TraceArtifact:
        trace_root = self.output_root / "trace"
        repository_root = Path(__file__).resolve().parents[2]
        repository_commit = _repository_commit(repository_root)
        command = (
            run.official_command[0], "-m", "gala_sim.adapters.trace_runner",
            "--trace-output", str(trace_root), "--stream-only",
            str(run.source_root / "train.py"),
            *run.official_command[2:],
        )
        environment = os.environ.copy()
        # Keep the allocator from reserving a fragmented tail during the
        # bounded capture window.  This is especially important for Walnut,
        # whose dense projections leave less than 1 GiB of headroom.
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:128",
        )
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
        manifest_path = trace_root / "chunk_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise TraceCaptureUnavailable("captured trace chunk manifest is malformed")
            manifest["metadata"] = dict(trace.metadata)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
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

    def capture_virtual_archive(
        self, run: PreparedRun, *, archive_root: Path, capture_config: Path,
        capture_iteration_range: tuple[int, int] = (1, 1),
    ) -> dict[str, Any]:
        """Capture compact CUDA packets without materializing a full trace."""
        repository_root = Path(__file__).resolve().parents[2]
        capture_root = self.output_root / "virtual-capture"
        command = (
            run.official_command[0], "-m", "gala_sim.adapters.trace_runner",
            "--trace-output", str(capture_root),
            "--model-id", "r2_gaussian", "--dataset-id", run.dataset_name,
            "--virtual-capture", "--virtual-capture-audit-only",
            "--packet-archive-root", str(Path(archive_root)),
            "--capture-config", str(Path(capture_config)),
            "--capture-iteration-range",
            f"{capture_iteration_range[0]}:{capture_iteration_range[1]}",
            "--stop-after-capture-range", str(run.source_root / "train.py"),
            *run.official_command[2:],
        )
        environment = os.environ.copy()
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:128",
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(repository_root), environment.get("PYTHONPATH")) if item
        )
        return run_trace_process(
            command, cwd=run.source_root, environment=environment,
            trace_root=capture_root, inactivity_timeout_seconds=300.0,
            preflight_fn=lambda: self.preflight_environment(environment),
            prepare_fn=lambda: self.output_root.mkdir(parents=True, exist_ok=True),
        )

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


# Compatibility name retained for callers that imported the Chest-specific
# adapter before R² support was generalized to all prepared datasets.
R2GaussianChestAdapter = R2GaussianAdapter


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
    # Resolve the reference through the shared dataset contract.  This keeps
    # the R² adapter usable for Walnut and HDTomo-USB, whose source formats are
    # different from the original Chest metadata layout.
    from .datasets import get_dataset_adapter

    dataset_id = {
        "chest": "chest",
        "walnut": "walnut",
        "fips walnut": "walnut",
        "hdtomo_usb": "hdtomo_usb",
        "hdtomo-usb": "hdtomo_usb",
    }.get(run.dataset_name.lower(), run.dataset_name.lower().replace("-", "_"))
    try:
        manifest = get_dataset_adapter(dataset_id).load(run.dataset_root)
    except (KeyError, ValueError, OSError):
        manifest = load_chest_manifest(run.dataset_root)
    reference_path = getattr(
        manifest, "reference_volume", getattr(manifest, "volume_path", None)
    )
    if reference_path is None:
        return _read_latest_metrics(output_root)
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    candidate = np.load(volume_path, mmap_mode="r", allow_pickle=False)
    quality = measure_quality(reference, candidate, run.quality_config)
    return {
        "psnr": quality.psnr,
        "ssim": quality.ssim,
        "lpips": quality.lpips,
        **_read_latest_metrics(output_root),
    }
