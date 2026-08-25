"""Strict loader for the parameter registry's YAML representation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping
from types import MappingProxyType

from gala_sim.identity import canonical_json, sha256_bytes


class ConfigError(ValueError):
    """Raised when a configuration violates the registry contract."""


@dataclass(frozen=True)
class GalaConfig:
    """A validated, immutable configuration snapshot."""

    path: Path
    parameters: Mapping[str, Any]
    sha256: str
    ready: bool

    def parameter(self, dotted_path: str) -> Mapping[str, Any]:
        current: Any = self.parameters
        for component in dotted_path.split("."):
            if not isinstance(current, Mapping) or component not in current:
                raise KeyError(dotted_path)
            current = current[component]
        if not isinstance(current, Mapping) or "value" not in current:
            raise KeyError(dotted_path)
        return current

    def value(self, dotted_path: str) -> Any:
        return self.parameter(dotted_path)["value"]

    def require_ready(self) -> None:
        if not self.ready:
            pending = pending_parameters(self.parameters)
            raise ConfigError("configuration has unfrozen parameters: " + ", ".join(pending))


def _yaml_load(path: Path) -> Any:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment diagnosis
        raise ConfigError("PyYAML is required to load GalaConfig") from error
    try:
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML: {path}") from error


def _walk_parameters(value: Any, path: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"parameter group {path or '<root>'} must be a mapping")
    leaves: list[tuple[str, Mapping[str, Any]]] = []
    for key, child in value.items():
        if not isinstance(key, str) or not key or key.startswith("_"):
            raise ConfigError(f"invalid parameter name at {path or '<root>'}")
        child_path = f"{path}.{key}" if path else key
        if isinstance(child, Mapping) and "value" in child:
            required = {"value", "unit", "source", "scope"}
            missing = required.difference(child)
            if missing:
                raise ConfigError(f"{child_path} missing metadata: {sorted(missing)}")
            if set(child).difference(required | {"status", "allowed_range"}):
                raise ConfigError(f"{child_path} contains unknown metadata")
            if not all(isinstance(child[item], str) and child[item] for item in ("unit", "source", "scope")):
                raise ConfigError(f"{child_path} has invalid unit/source/scope metadata")
            status = child.get("status", "frozen")
            if not isinstance(status, str) or status not in {"frozen", "pending", "design_parameter_pending_freeze"}:
                raise ConfigError(f"{child_path} has an invalid status")
            value_item = child["value"]
            if isinstance(value_item, float) and not math.isfinite(value_item):
                raise ConfigError(f"{child_path} contains a non-finite value")
            allowed = child.get("allowed_range")
            if allowed is not None:
                if (not isinstance(allowed, (list, tuple)) or len(allowed) != 2
                        or not all(isinstance(item, (int, float)) and not isinstance(item, bool)
                                   and math.isfinite(float(item)) for item in allowed)
                        or allowed[0] > allowed[1]):
                    raise ConfigError(f"{child_path} has an invalid allowed_range")
                if value_item is not None and isinstance(value_item, (int, float)) and not isinstance(value_item, bool):
                    if not allowed[0] <= value_item <= allowed[1]:
                        raise ConfigError(f"{child_path} value is outside allowed_range")
            leaves.append((child_path, child))
        else:
            leaves.extend(_walk_parameters(child, child_path))
    return leaves


def pending_parameters(parameters: Mapping[str, Any]) -> list[str]:
    return sorted(path for path, item in _walk_parameters(parameters)
                  if item.get("status") != "frozen" or item.get("value") is None)


def load_config(path: str | Path) -> GalaConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    document = _yaml_load(config_path)
    if not isinstance(document, Mapping) or set(document) != {"schema_version", "parameters"}:
        raise ConfigError("configuration root must contain only schema_version and parameters")
    if document["schema_version"] != "gala-config-v1":
        raise ConfigError("unsupported configuration schema")
    parameter_map = document["parameters"]
    leaves = _walk_parameters(parameter_map)
    if not leaves:
        raise ConfigError("configuration has no parameters")
    ready = not pending_parameters(parameter_map)
    encoded = canonical_json(document)
    frozen_map = _freeze(parameter_map)
    return GalaConfig(config_path, frozen_map, sha256_bytes(encoded), ready)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def write_canonical_json(config: GalaConfig, output: Path) -> None:
    """Write a stable human-readable snapshot without changing its identity."""

    output.write_text(json.dumps({"schema_version": "gala-config-v1", "parameters": _thaw(config.parameters)},
                                 ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value
