"""Input-freeze records for the first R²-Gaussian + Chest combination."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import os
import platform
import shutil
import subprocess
import copy
from pathlib import Path
import sys
from typing import Any

import numpy as np

from gala_sim.config import GalaConfig
from gala_sim.config.loader import _yaml_load
from gala_sim.identity import canonical_json, sha256_bytes, sha256_file, sha256_tree


QUALITY_PARAMETERS = (
    "quality.data_min",
    "quality.data_max",
    "quality.ssim_window",
    "quality.ssim_sigma",
    "quality.ssim_boundary",
    "quality.lpips_slices",
    "quality.lpips_network",
    "quality.lpips_version",
    "quality.lpips_backbone_sha256",
    "quality.lpips_calibration_sha256",
)


@dataclass(frozen=True)
class SourceRecord:
    name: str
    url: str
    commit: str
    root: str
    tree_sha256: str
    upstream_patch_sha256: str
    license_path: str
    license_sha256: str


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    status: str
    source_url: str
    license_url: str
    root: str | None
    manifest_sha256: str | None
    reason: str | None
    files: list[dict[str, Any]] | None = None
    metadata_sha256: str | None = None
    geometry: dict[str, Any] | None = None
    reference_volume: dict[str, Any] | None = None


TRAINING_GROUP_CLASSES = {
    "model": "ModelParams",
    "pipeline": "PipelineParams",
    "optimization": "OptimizationParams",
}


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _python_value(executable: str, expression: str) -> str | None:
    return _command(executable, "-c", f"print({expression})")


def environment_snapshot(python_executable: str = sys.executable) -> dict[str, Any]:
    executable = str(Path(python_executable).resolve())
    return {
        "python_executable": executable,
        "python": _python_value(executable, "__import__('platform').python_version()"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _command("lscpu"),
        "gpu": _command(
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        "git": _command("git", "--version"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "cuda_compiler_path": shutil.which("nvcc"),
        "cuda": _command("nvcc", "--version"),
        "c_compiler_path": shutil.which(os.environ.get("CC", "cc")),
        "c_compiler": _command(os.environ.get("CC", "cc"), "--version"),
        "cxx_compiler_path": shutil.which(os.environ.get("CXX", "c++")),
        "cxx_compiler": _command(os.environ.get("CXX", "c++"), "--version"),
        "torch_cuda": _command(
            executable, "-c", "import torch; print(torch.version.cuda)"
        ),
        "numpy": _python_value(executable, "__import__('importlib.metadata').metadata.version('numpy')"),
        "scikit_image": _python_value(executable, "__import__('importlib.metadata').metadata.version('scikit-image')"),
        "torch": _python_value(executable, "__import__('importlib.metadata').metadata.version('torch')"),
        "torchvision": _python_value(executable, "__import__('importlib.metadata').metadata.version('torchvision')"),
        "lpips": _python_value(executable, "__import__('importlib.metadata').metadata.version('lpips')"),
        "numba": _python_value(executable, "__import__('importlib.metadata').metadata.version('numba')"),
        "ramulator2": None,
    }


def source_record(root: Path, name: str, url: str, commit: str) -> SourceRecord:
    root = root.resolve()
    actual = _command("git", "-C", str(root), "rev-parse", "HEAD")
    if actual != commit:
        raise ValueError(f"{name} is at {actual!r}, expected {commit!r}")
    dirty = _command("git", "-C", str(root), "status", "--porcelain")
    if dirty:
        raise ValueError(f"{name} checkout has uncommitted changes")
    license_path = root / "LICENSE.md"
    if not license_path.is_file():
        raise ValueError(f"{name} license file is missing: {license_path}")
    return SourceRecord(name, url, commit, str(root), sha256_tree(root),
                        sha256_bytes(b""), str(license_path), sha256_file(license_path))


def _class_defaults(path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if class_node is None:
        raise ValueError(f"upstream training class is missing: {class_name}")
    init_node = next(
        (node for node in class_node.body
         if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
        None,
    )
    if init_node is None:
        raise ValueError(f"upstream training class has no initializer: {class_name}")
    defaults: dict[str, Any] = {}
    for node in init_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            try:
                defaults[target.attr.lstrip("_")] = ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"upstream default for {class_name}.{target.attr} is not literal"
                ) from error
    return defaults


def _runtime_defaults(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            flag = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        destination = flag[2:]
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        if "default" in keywords:
            defaults[destination] = ast.literal_eval(keywords["default"])
        elif "action" in keywords and ast.literal_eval(keywords["action"]) == "store_true":
            defaults[destination] = False
        else:
            defaults[destination] = None
    return defaults


def _call_arguments(tree: ast.AST, dotted_name: str) -> list[tuple[Any, ...]]:
    found: list[tuple[Any, ...]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts: list[str] = []
        function: ast.expr = node.func
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        if ".".join(reversed(parts)) != dotted_name:
            continue
        try:
            found.append(tuple(ast.literal_eval(argument) for argument in node.args))
        except (ValueError, TypeError):
            continue
    return found


def _cuda_device(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Call) or len(node.args) != 1
                or not isinstance(node.args[0], ast.Call)):
            continue
        parts: list[str] = []
        function: ast.expr = node.func
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        if ".".join(reversed(parts)) != "torch.cuda.set_device":
            continue
        device_call = node.args[0]
        if (isinstance(device_call.func, ast.Attribute)
                and isinstance(device_call.func.value, ast.Name)
                and device_call.func.value.id == "torch" and device_call.func.attr == "device"
                and len(device_call.args) == 1):
            try:
                device = ast.literal_eval(device_call.args[0])
            except (ValueError, TypeError):
                return None
            return device if isinstance(device, str) else None
    return None


def _has_schedule_append(tree: ast.AST, field: str, value: str | int) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (node.func.attr != "append" or not isinstance(owner, ast.Attribute)
                or not isinstance(owner.value, ast.Name) or owner.value.id != "args"
                or owner.attr != field or len(node.args) != 1):
            continue
        argument = node.args[0]
        if isinstance(value, str):
            if (isinstance(argument, ast.Attribute) and isinstance(argument.value, ast.Name)
                    and argument.value.id == "args" and argument.attr == value):
                return True
        elif isinstance(argument, ast.Constant) and argument.value == value:
            return True
    return False


def training_record(profile_path: Path, source: SourceRecord, dataset_root: Path | None,
                    model_output: Path | None, python_executable: Path) -> dict[str, Any]:
    """Validate and bind the frozen R2-Gaussian training profile."""

    profile_path = profile_path.resolve()
    document = _yaml_load(profile_path)
    if not isinstance(document, Mapping):
        raise ValueError("campaign manifest root must be a mapping")
    training = document.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("campaign manifest has no training profile")
    profile = _json_value(training)
    if profile.get("schema_version") != "gala-r2-training-freeze-v1":
        raise ValueError("unsupported training profile schema")
    if profile.get("model_commit") != source.commit:
        raise ValueError("training profile model commit does not match source")
    if profile.get("configuration_file") is not None:
        raise ValueError("frozen official training must not use a configuration override")

    source_root = Path(source.root)
    arguments_path = source_root / "r2_gaussian/arguments/__init__.py"
    train_path = source_root / str(profile.get("entrypoint"))
    state_path = source_root / "r2_gaussian/utils/general_utils.py"
    for path in (arguments_path, train_path, state_path):
        if not path.is_file():
            raise ValueError(f"upstream training source is missing: {path}")

    groups = profile.get("parameter_groups")
    if not isinstance(groups, dict):
        raise ValueError("training parameter_groups must be a mapping")
    for group, class_name in TRAINING_GROUP_CLASSES.items():
        expected = groups.get(group)
        actual = _class_defaults(arguments_path, class_name)
        if expected != actual:
            raise ValueError(f"frozen {group} parameters do not match upstream defaults")

    runtime = profile.get("runtime_arguments")
    if runtime != _runtime_defaults(train_path):
        raise ValueError("frozen runtime arguments do not match upstream defaults")
    iterations = groups["optimization"]["iterations"]
    expected_schedule = {
        "test_iterations": [*runtime["test_iterations"], iterations, 1],
        "save_iterations": [*runtime["save_iterations"], iterations],
        "checkpoint_iterations": list(runtime["checkpoint_iterations"]),
    }
    train_tree = ast.parse(train_path.read_text(encoding="utf-8"), filename=str(train_path))
    if not all((
        _has_schedule_append(train_tree, "save_iterations", "iterations"),
        _has_schedule_append(train_tree, "test_iterations", "iterations"),
        _has_schedule_append(train_tree, "test_iterations", 1),
    )) or profile.get("effective_schedule") != expected_schedule:
        raise ValueError("frozen evaluation/save schedule does not match upstream")

    state_tree = ast.parse(state_path.read_text(encoding="utf-8"), filename=str(state_path))
    safe_state = next(
        (node for node in state_tree.body
         if isinstance(node, ast.FunctionDef) and node.name == "safe_state"),
        None,
    )
    if safe_state is None:
        raise ValueError("upstream safe_state function is missing")
    frozen_state = profile.get("random_state")
    actual_state = {
        "python_random_seed": _call_arguments(safe_state, "random.seed"),
        "numpy_seed": _call_arguments(safe_state, "np.random.seed"),
        "torch_seed": _call_arguments(safe_state, "torch.manual_seed"),
    }
    normalized_state = {
        key: values[0][0] if len(values) == 1 and len(values[0]) == 1 else None
        for key, values in actual_state.items()
    }
    normalized_state["cuda_device"] = _cuda_device(safe_state)
    if frozen_state != normalized_state:
        raise ValueError("frozen random state does not match upstream safe_state")

    command = profile.get("command")
    if not isinstance(command, dict) or not isinstance(command.get("argv"), list):
        raise ValueError("training command must provide an argv list")
    resolved_python = python_executable.resolve()
    if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
        raise ValueError(f"official Python interpreter is not executable: {resolved_python}")
    bindings = {
        "source_root": str(source_root.resolve()),
        "python_executable": str(resolved_python),
        "dataset_root": (str(dataset_root.resolve()) if dataset_root is not None
                         else "<chest-data-root>"),
        "model_output": (str(model_output.resolve()) if model_output is not None
                         else "<model-output-root>"),
    }
    try:
        argv = [str(value).format(**bindings) for value in command["argv"]]
        working_directory = str(command["working_directory"]).format(**bindings)
    except (KeyError, ValueError) as error:
        raise ValueError("training command contains an invalid binding") from error
    if dataset_root is not None and model_output is None:
        raise ValueError("--model-output is required with --dataset-root")
    command_record = {"working_directory": working_directory, "argv": argv}
    effective_arguments = {
        **groups["model"], **groups["pipeline"], **groups["optimization"], **runtime,
        "source_path": bindings["dataset_root"], "model_path": bindings["model_output"],
        **expected_schedule,
    }
    files = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in (arguments_path, train_path, state_path)
    }
    return {
        "profile": profile,
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "source_files": files,
        "command": {**command_record, "sha256": sha256_bytes(canonical_json(command_record))},
        "effective_arguments": effective_arguments,
    }


def dataset_record(root: Path | None, name: str, source_url: str, license_url: str,
                   reason: str | None = None) -> DatasetRecord:
    if root is None:
        if not reason:
            raise ValueError("a missing dataset requires a machine-readable reason")
        return DatasetRecord(name, "unavailable_data", source_url, license_url, None, None, reason)
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    metadata_path = root / "meta_data.json"
    metadata: dict[str, Any] | None = None
    metadata_digest: str | None = None
    geometry: dict[str, Any] | None = None
    if metadata_path.is_file():
        try:
            parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"dataset metadata is not valid JSON: {metadata_path}") from error
        if not isinstance(parsed, dict):
            raise ValueError("dataset metadata root must be an object")
        metadata = parsed
        metadata_digest = sha256_file(metadata_path)
        if isinstance(parsed.get("scanner"), dict):
            geometry = parsed["scanner"]
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()
                       and ".git" not in item.relative_to(root).parts):
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
    volume_name = metadata.get("vol", "vol_gt.npy") if metadata is not None else "vol_gt.npy"
    volume_path = root / str(volume_name)
    reference_volume: dict[str, Any] | None = None
    if volume_path.is_file():
        try:
            volume = np.load(volume_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"reference volume is not a valid NumPy array: {volume_path}") from error
        if volume.ndim != 3 or volume.size == 0 or not np.isfinite(volume).all():
            raise ValueError("reference volume must be a finite non-empty 3D array")
        reference_volume = {
            "path": volume_path.relative_to(root).as_posix(),
            "sha256": sha256_file(volume_path),
            "shape": [int(value) for value in volume.shape],
            "dtype": volume.dtype.str,
            "data_min": float(volume.min()),
            "data_max": float(volume.max()),
        }
    return DatasetRecord(name, "planned", source_url, license_url, str(root), sha256_tree(root), None,
                         files, metadata_digest, geometry, reference_volume)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_value(child) for child in value]
    return value


def _quality_snapshot(config: GalaConfig) -> dict[str, Any]:
    try:
        return {
            name: _json_value(config.parameter(name))
            for name in QUALITY_PARAMETERS
        }
    except KeyError as error:
        raise ValueError(f"quality configuration is incomplete: {error.args[0]}") from error


def _validate_quality_reference(config: GalaConfig, dataset: DatasetRecord) -> None:
    from gala_sim.metrics import QualityConfig

    quality = QualityConfig.from_gala(config)
    reference = dataset.reference_volume
    if reference is None:
        return
    if (
        float(reference["data_min"]) != quality.data_min
        or float(reference["data_max"]) != quality.data_max
    ):
        raise ValueError("quality data range does not match the frozen reference volume")
    shape = tuple(int(value) for value in reference["shape"])
    if len(shape) != 3 or any(
        index >= shape[axis]
        for axis, indexes in enumerate(quality.lpips_slices)
        for index in indexes
    ):
        raise ValueError("quality LPIPS slices do not fit the frozen reference volume")


def build_freeze_record(config: GalaConfig, source: SourceRecord, dataset: DatasetRecord,
                        training: Mapping[str, Any], seed: int,
                        repository: Path) -> dict[str, Any]:
    _validate_quality_reference(config, dataset)
    random_state = training.get("profile", {}).get("random_state", {})
    if seed != 0 or any(random_state.get(name) != seed for name in (
        "python_random_seed", "numpy_seed", "torch_seed",
    )):
        raise ValueError("random seed does not match the frozen upstream safe_state")
    command = training.get("command")
    if (not isinstance(command, Mapping) or not isinstance(command.get("argv"), list)
            or not command["argv"] or not isinstance(command["argv"][0], str)):
        raise ValueError("training record has no bound Python command")
    record: dict[str, Any] = {
        "schema_version": "gala-input-freeze-v3",
        "workflow_step": "freeze_inputs",
        "status": "planned" if dataset.status == "planned" and config.ready else dataset.status,
        "model": asdict(source),
        "dataset": asdict(dataset),
        "config": {"path": str(config.path), "sha256": config.sha256, "ready": config.ready},
        "quality": {"parameters": _quality_snapshot(config)},
        "training": _json_value(training),
        "random_seed": seed,
        "repository": {"root": str(repository.resolve()), "commit": _command("git", "-C", str(repository), "rev-parse", "HEAD")},
        "environment": environment_snapshot(command["argv"][0]),
    }
    record["run_manifest_sha256"] = sha256_bytes(canonical_json(record))
    return record


def write_freeze_record(record: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def verify_freeze_record(record: dict[str, Any]) -> None:
    """Check the self-hash before a record is used as a workflow input."""

    recorded = record.get("run_manifest_sha256")
    if not isinstance(recorded, str):
        raise ValueError("freeze record has no run_manifest_sha256")
    unsigned = copy.deepcopy(record)
    del unsigned["run_manifest_sha256"]
    if sha256_bytes(canonical_json(unsigned)) != recorded:
        raise ValueError("freeze record self-hash does not match its contents")
