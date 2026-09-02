"""Registry and command adapters for supported reconstruction models."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import json

from gala_sim.assets import AssetCatalog
from gala_sim.identity import sha256_tree
from gala_sim.metrics import QualityConfig
from gala_sim.trace import TraceReader, TraceWriter
from gala_sim.workspace import WorkspacePaths

from .gr_gaussian import GRGaussianAdapter
from .protocol import PreparedRun, ReferenceArtifact, TraceArtifact
from .trace_capture import TRACE_HOOK_PROFILES
from .trace_process import TraceProcessError, run_trace_process


_MODEL_ENVIRONMENT_IMPORTS = {
    "r2_gaussian": (
        "torch",
        "r2_gaussian.gaussian.gaussian_model",
        "xray_gaussian_rasterization_voxelization",
    ),
    "fact_gs": (
        "torch",
        "fact_gs.r2_gaussian.gaussian.gaussian_model",
        "fused_ssim_cuda",
        "gs_ct_rasterizer.rasterize",
        "gs_voxelizer.voxelize",
    ),
    "exact_gs": (
        "torch",
        "exact_gs.gaussian.gaussian_model",
        "exact_gaussian_rasterization",
        "xray_gaussian_rasterization_voxelization",
    ),
}


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
    capture_iteration_range: tuple[int, int] = (1, 1)

    def build_command(
        self,
        dataset_root: Path,
        output_root: Path,
        *,
        initialization_root: Path | None = None,
    ) -> tuple[str, ...]:
        dataset_root = Path(dataset_root)
        metadata_root = Path(initialization_root or dataset_root)
        initialization = _dataset_initialization_path(metadata_root)
        evaluation_enabled = _evaluation_reference_available(metadata_root)
        bindings = {
            "python_executable": str(self.python_executable),
            "dataset_root": str(dataset_root),
            "output_root": str(Path(output_root)),
            "initialization_path": str(initialization),
            "initialization_mode": (
                "precomputed" if initialization.is_file() else "gradient"
            ),
            "evaluation_enabled": str(evaluation_enabled).lower(),
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
        command_dataset_root = (
            self.output_root / "input"
            if self.descriptor.output_strategy == "dataset_output_directory"
            else dataset_root
        )
        return PreparedRun(
            model_name=self.descriptor.display_name,
            dataset_name=getattr(dataset, "id", "dataset"),
            source_root=self.source_root.resolve(),
            dataset_root=dataset_root,
            config_sha256=config.sha256,
            quality_config=QualityConfig.from_gala(config),
            seed=0,
            official_command=self.build_command(
                command_dataset_root,
                self.output_root,
                initialization_root=dataset_root,
            ),
            output_root=self.output_root.resolve(),
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
        if self.descriptor.id not in TRACE_HOOK_PROFILES:
            raise RuntimeError(
                f"{self.descriptor.display_name} trace capture requires its installed adapter hooks"
            )
        from .r2_gaussian import (
            _bind_trace_identity, _push_trace_chunks, _repository_commit,
        )

        execution_output = self.output_root
        trace_root = execution_output / "trace"
        repository_root = Path(__file__).resolve().parents[2]
        train_script = run.source_root / run.official_command[1]
        train_arguments = list(run.official_command[2:])
        if self.descriptor.id == "exact_gs":
            # Exact-GS evaluates at iteration one by default.  Keep evaluation
            # outside the bounded trace window, especially for datasets that
            # deliberately have no reference reconstruction.
            train_arguments.extend((
                "--test_iterations",
                str(self.capture_iteration_range[1] + 1),
            ))
        command = (
            run.official_command[0], "-m", "gala_sim.adapters.trace_runner",
            "--trace-output", str(trace_root),
            "--model-id", self.descriptor.id,
            "--dataset-id", run.dataset_name,
            "--capture-iteration-range",
            f"{self.capture_iteration_range[0]}:{self.capture_iteration_range[1]}",
            "--stop-after-capture-range",
            str(train_script), *train_arguments,
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (
                str(repository_root), environment.get("PYTHONPATH"),
            ) if item
        )
        gpu_reference = run_trace_process(
            command, cwd=run.source_root, environment=environment,
            trace_root=trace_root, inactivity_timeout_seconds=300.0,
            preflight_fn=lambda: self.preflight_environment(environment),
            prepare_fn=lambda: self._prepare_output(run.dataset_root),
        )
        trace = TraceReader().read(trace_root, mmap_mode="r")
        model_commit = self.descriptor.commit
        if not model_commit:
            raise RuntimeError("official trace capture has no pinned model commit")
        trace = _bind_trace_identity(
            trace, run, model_commit, _repository_commit(repository_root),
        )
        TraceWriter().write(trace, trace_root, validate=True)
        _push_trace_chunks(trace, sink)
        volume = _find_volume(execution_output, required=False)
        gpu_reference["trace_event_count"] = trace.event_count
        return TraceArtifact(
            trace_root,
            trace,
            ReferenceArtifact(execution_output, volume, {}, gpu_reference),
        )

    def replay_reductions(self, run: PreparedRun, order: Any) -> ReferenceArtifact:
        raise RuntimeError("functional reduction replay requires a captured trace")

    def preflight_environment(self, environment: Mapping[str, str]) -> None:
        """Verify model imports without exposing a GPU or launching training."""

        imports = _MODEL_ENVIRONMENT_IMPORTS.get(self.descriptor.id)
        if imports is None:
            raise TraceProcessError("model_environment_preflight_not_defined")
        script = (
            "import importlib,json,torch;"
            "mps=getattr(torch,'mps',None);"
            "mps is not None and not hasattr(mps,'is_available') and "
            "setattr(mps,'is_available',lambda:False);"
            f"names=json.loads({json.dumps(json.dumps(imports))});"
            "[importlib.import_module(name) for name in names]"
        )
        check_environment = dict(environment)
        check_environment["CUDA_VISIBLE_DEVICES"] = ""
        try:
            completed = subprocess.run(
                (str(self.python_executable), "-c", script),
                cwd=self.source_root,
                env=check_environment,
                check=False,
                text=True,
                capture_output=True,
                timeout=60.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TraceProcessError("model_environment_preflight_failed") from error
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise TraceProcessError(f"model_environment_preflight_failed{suffix}")

    def _prepare_output(self, dataset_root: Path) -> Path:
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        if self.descriptor.output_strategy != "dataset_output_directory":
            self.output_root.mkdir(parents=True, exist_ok=True)
            return self.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        source = Path(dataset_root).resolve()
        dataset_view = self.output_root / "input"
        dataset_view.mkdir(exist_ok=True)
        for item in source.iterdir():
            if item.name == "output":
                continue
            link = dataset_view / item.name
            if link.is_symlink():
                if link.resolve() != item.resolve():
                    raise ValueError("Exact-GS dataset view contains a stale input link")
            elif link.exists():
                raise ValueError("Exact-GS dataset view contains an unmanaged input")
            else:
                link.symlink_to(item, target_is_directory=item.is_dir())
        model_output = self.output_root / "model-output"
        model_output.mkdir(exist_ok=True)
        upstream_output = dataset_view / "output"
        if upstream_output.is_symlink():
            if upstream_output.resolve() != model_output.resolve():
                raise ValueError("Exact-GS dataset view has a stale output link")
        elif upstream_output.exists():
            raise ValueError("Exact-GS dataset view output is unmanaged")
        else:
            upstream_output.symlink_to(model_output, target_is_directory=True)
        return model_output


def _find_volume(output: Path, *, required: bool = True) -> Path | None:
    patterns = (
        "**/vol_pred.npy", "**/volume.npy", "**/reconstruction.npy",
        "**/vol_pred.tif", "**/vol_pred.tiff",
    )
    for pattern in patterns:
        candidates = sorted(output.glob(pattern))
        if candidates:
            return candidates[-1]
    if required:
        raise RuntimeError("model run produced no recognized reconstructed volume")
    return None


def _evaluation_reference_available(dataset_root: Path) -> bool:
    metadata_path = Path(dataset_root) / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid prepared dataset metadata: {metadata_path}") from error
        return isinstance(metadata.get("reference_volume"), str)
    r2_metadata_path = Path(dataset_root) / "meta_data.json"
    if not r2_metadata_path.is_file():
        return False
    try:
        metadata = json.loads(r2_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid upstream dataset metadata: {r2_metadata_path}") from error
    volume = metadata.get("vol")
    return isinstance(volume, str) and (Path(dataset_root) / volume).is_file()


def _dataset_initialization_path(dataset_root: Path) -> Path:
    """Resolve the prepared initializer without depending on a cache name."""

    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid prepared dataset metadata: {metadata_path}") from error
        value = metadata.get("initialization")
        if isinstance(value, str):
            candidate = (dataset_root / value).resolve()
            try:
                candidate.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ValueError("prepared initialization escapes the dataset root") from error
            return candidate
    return dataset_root / f"init_{dataset_root.name}.npy"


def get_model_adapter(
    model_id: str,
    *,
    workspace: WorkspacePaths | None = None,
    output_root: Path | None = None,
    python_executable: Path | None = None,
    capture_iteration_range: tuple[int, int] = (1, 1),
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
    workspace_python = paths.build / "envs" / model_id / "bin" / "python"
    selected_python = python_executable or (
        workspace_python if workspace_python.is_file() else Path(sys.executable)
    )
    return CommandModelAdapter(
        descriptor, paths.upstream / model_id, output,
        selected_python, capture_iteration_range,
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

    from .datasets import (
        ensure_official_dataset_layout, ensure_projection_initialization, get_dataset_adapter,
        initialize_conversion_cache,
    )

    paths = (workspace or WorkspacePaths.discover()).ensure()
    dataset_adapter = get_dataset_adapter(dataset_id)
    source_manifest = dataset_adapter.load(dataset_root)
    source_sha256 = sha256_tree(source_manifest.root)
    identity = source_sha256[:16]
    prepared_root = paths.cache / "prepared" / dataset_id / identity
    if (prepared_root / "metadata.json").is_file():
        prepared = dataset_adapter.load(prepared_root)
    else:
        initialize_conversion_cache(source_manifest, prepared_root, source_sha256)
        prepared = dataset_adapter.convert(source_manifest, prepared_root)
    ensure_official_dataset_layout(prepared)
    if model_id in {"r2_gaussian", "fact_gs", "exact_gs"} and prepared.reference_volume is None:
        prepared = ensure_projection_initialization(prepared)
    model_adapter = get_model_adapter(
        model_id,
        workspace=paths,
        output_root=paths.results / model_id / dataset_id,
        python_executable=python_executable,
    )
    return model_adapter.prepare(prepared, config)
