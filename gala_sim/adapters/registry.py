"""Registry and command adapters for supported reconstruction models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from gala_sim.assets import AssetCatalog
from gala_sim.identity import sha256_tree
from gala_sim.metrics import QualityConfig
from gala_sim.workspace import WorkspacePaths

from .gr_gaussian import GRGaussianAdapter
from .protocol import PreparedRun, ReferenceArtifact, TraceArtifact


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    display_name: str
    implementation: str
    commit: str | None
    paper_doi: str | None
    commands: Mapping[str, tuple[str, ...]]
    stage_boundaries: tuple[str, ...]
    output_strategy: str


def model_descriptors(repository: Path | None = None) -> Mapping[str, ModelDescriptor]:
    paths = WorkspacePaths.discover(repository=repository)
    catalog = AssetCatalog.load(paths.repository / "configs")
    return {
        key: ModelDescriptor(
            item.id, item.name, item.implementation, item.commit, item.paper_doi,
            item.official_commands, item.stage_boundaries, item.output_strategy,
        )
        for key, item in catalog.models.items()
    }


@dataclass(frozen=True)
class CommandModelAdapter:
    descriptor: ModelDescriptor
    source_root: Path
    output_root: Path
    python_executable: Path = Path(sys.executable)

    def build_command(self, dataset_root: Path, output_root: Path) -> tuple[str, ...]:
        bindings = {
            "python_executable": str(self.python_executable),
            "dataset_root": str(Path(dataset_root)),
            "output_root": str(Path(output_root)),
        }
        try:
            return tuple(argument.format(**bindings) for argument in self.descriptor.commands["train"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid training command for {self.descriptor.id}") from error

    def prepare(self, dataset: Any, config: Any) -> PreparedRun:
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"model source is missing: {self.source_root}")
        dataset_root = Path(getattr(dataset, "root", dataset)).resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root is missing: {dataset_root}")
        return PreparedRun(
            self.descriptor.display_name, getattr(dataset, "id", "dataset"),
            self.source_root.resolve(), dataset_root, config.sha256,
            QualityConfig.from_gala(config), 0,
            self.build_command(dataset_root, self.output_root),
        )

    def run_reference(self, run: PreparedRun) -> ReferenceArtifact:
        execution_output = self._prepare_output(run.dataset_root)
        started = time.monotonic()
        completed = subprocess.run(
            run.official_command, cwd=run.source_root, check=False,
            text=True, capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{self.descriptor.display_name} command failed: "
                + (completed.stderr[-2000:] or completed.stdout[-2000:])
            )
        volume = _find_volume(execution_output)
        return ReferenceArtifact(
            execution_output, volume, {}, {
                "wall_seconds": time.monotonic() - started,
                "command": list(run.official_command),
            },
        )

    def capture_trace(self, run: PreparedRun, sink: Any) -> TraceArtifact:
        raise RuntimeError(
            f"{self.descriptor.display_name} trace capture requires its installed adapter hooks"
        )

    def replay_reductions(self, run: PreparedRun, order: Any) -> ReferenceArtifact:
        raise RuntimeError("functional reduction replay requires a captured trace")

    def _prepare_output(self, dataset_root: Path) -> Path:
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        if self.descriptor.output_strategy != "dataset_output_directory":
            self.output_root.mkdir(parents=True, exist_ok=True)
            return self.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        upstream_output = dataset_root / "output"
        if upstream_output.is_symlink():
            if upstream_output.resolve() != self.output_root.resolve():
                raise ValueError("dataset output link targets a different model run")
        elif upstream_output.exists():
            if upstream_output.resolve() != self.output_root.resolve():
                raise ValueError("Exact-GS dataset output directory is already occupied")
        else:
            upstream_output.symlink_to(self.output_root, target_is_directory=True)
        return self.output_root


def _find_volume(output: Path) -> Path:
    patterns = (
        "**/vol_pred.npy", "**/volume.npy", "**/reconstruction.npy",
        "**/vol_pred.tif", "**/vol_pred.tiff",
    )
    for pattern in patterns:
        candidates = sorted(output.glob(pattern))
        if candidates:
            return candidates[-1]
    raise RuntimeError("model run produced no recognized reconstructed volume")


def get_model_adapter(
    model_id: str,
    *,
    workspace: WorkspacePaths | None = None,
    output_root: Path | None = None,
    python_executable: Path | None = None,
) -> Any:
    paths = workspace or WorkspacePaths.discover()
    descriptors = model_descriptors(paths.repository)
    try:
        descriptor = descriptors[model_id]
    except KeyError as error:
        raise KeyError(f"unknown model adapter: {model_id}") from error
    output = output_root or paths.results / model_id
    if descriptor.implementation == "independent_reimplementation":
        return GRGaussianAdapter(paths.repository, output, descriptor)
    return CommandModelAdapter(
        descriptor, paths.upstream / model_id, output,
        python_executable or Path(sys.executable),
    )


def prepare_campaign(
    model_id: str,
    dataset_id: str,
    dataset_root: Path,
    config: Any,
    *,
    workspace: WorkspacePaths | None = None,
    python_executable: Path | None = None,
) -> PreparedRun:
    """Convert a catalogued dataset and bind one model's reference command."""

    from .datasets import get_dataset_adapter

    paths = (workspace or WorkspacePaths.discover()).ensure()
    dataset_adapter = get_dataset_adapter(dataset_id)
    source_manifest = dataset_adapter.load(dataset_root)
    identity = sha256_tree(source_manifest.root)[:16]
    prepared_root = paths.cache / "prepared" / dataset_id / identity
    if (prepared_root / "metadata.json").is_file():
        prepared = dataset_adapter.load(prepared_root)
    else:
        prepared = dataset_adapter.convert(source_manifest, prepared_root)
    model_adapter = get_model_adapter(
        model_id, workspace=paths, python_executable=python_executable,
    )
    return model_adapter.prepare(prepared, config)
