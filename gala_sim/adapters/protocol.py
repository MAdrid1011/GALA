"""Typed boundary between an official model and the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gala_sim.metrics import QualityConfig
from gala_sim.trace import Trace


@dataclass(frozen=True)
class PreparedRun:
    model_name: str
    dataset_name: str
    source_root: Path
    dataset_root: Path
    config_sha256: str
    quality_config: QualityConfig
    seed: int
    official_command: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceArtifact:
    output_root: Path
    volume_path: Path
    metrics: dict[str, float]
    gpu_reference: dict[str, Any]


@dataclass(frozen=True)
class TraceArtifact:
    trace_root: Path
    trace: Trace
    reference: ReferenceArtifact


class ModelAdapter(Protocol):
    def prepare(self, dataset: Any, config: Any) -> PreparedRun: ...

    def run_reference(self, run: PreparedRun) -> ReferenceArtifact: ...

    def capture_trace(self, run: PreparedRun, sink: Any) -> TraceArtifact: ...

    def replay_reductions(self, run: PreparedRun, order: Any) -> ReferenceArtifact: ...
