"""R²-Gaussian official-entry adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Any

from gala_sim.config import GalaConfig
from gala_sim.trace import DeviceTraceSink

from .protocol import PreparedRun, ReferenceArtifact, TraceArtifact


class TraceCaptureUnavailable(RuntimeError):
    """Raised until the locked CUDA extension exposes its real event buffers."""


@dataclass(frozen=True)
class R2GaussianChestAdapter:
    source_root: Path
    dataset_root: Path
    output_root: Path
    model_commit: str

    def prepare(self, dataset: Any, config: GalaConfig) -> PreparedRun:
        if not self.source_root.is_dir() or not self.dataset_root.is_dir():
            raise FileNotFoundError("R²-Gaussian source or Chest root is unavailable")
        return PreparedRun(
            model_name="R2-Gaussian",
            dataset_name="Chest",
            source_root=self.source_root.resolve(),
            dataset_root=self.dataset_root.resolve(),
            config_sha256=config.sha256,
            seed=0,
            official_command=("python", "train.py", "-s", str(self.dataset_root), "-m", str(self.output_root)),
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
            metrics=_read_latest_metrics(self.output_root),
            gpu_reference={
                "wall_seconds": elapsed,
                "command": list(run.official_command),
                "stdout_sha256": _text_sha256(completed.stdout),
                "stderr_sha256": _text_sha256(completed.stderr),
            },
        )

    def capture_trace(self, run: PreparedRun, sink: DeviceTraceSink) -> TraceArtifact:
        raise TraceCaptureUnavailable(
            "locked R²-Gaussian CUDA extension does not expose complete CLAMP relation/task buffers; "
            "formal trace capture is refused rather than synthesized"
        )

    def replay_reductions(self, run: PreparedRun, order: Any) -> ReferenceArtifact:
        raise TraceCaptureUnavailable("functional replay requires a validated trace and reduction order")


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
