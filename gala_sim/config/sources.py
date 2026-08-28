"""Strict loaders for model and dataset source manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .loader import ConfigError, _yaml_load


@dataclass(frozen=True)
class SourceManifest:
    schema_version: str
    name: str
    values: Mapping[str, Any]

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise ConfigError(f"source manifest lacks {key}")
        return self.values[key]


def load_source_manifest(path: str | Path, *, schema_version: str) -> SourceManifest:
    manifest_path = Path(path).resolve()
    document = _yaml_load(manifest_path)
    if not isinstance(document, Mapping):
        raise ConfigError("source manifest root must be a mapping")
    if document.get("schema_version") != schema_version:
        raise ConfigError(f"unsupported source manifest schema: {manifest_path}")
    name = document.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("source manifest name is required")
    if schema_version == "gala-model-source-v1":
        _require_string_fields(document, ("repository", "commit", "evidence_level", "license"))
        commands = document.get("official_commands")
        if not isinstance(commands, Mapping) or not all(isinstance(value, str) and value for value in commands.values()):
            raise ConfigError("model manifest official_commands must be non-empty strings")
    if schema_version == "gala-dataset-source-v1":
        _require_string_fields(document, ("source_url", "license_url", "input_format"))
        required_files = document.get("required_files")
        if not isinstance(required_files, list) or not required_files or not all(isinstance(value, str) and value for value in required_files):
            raise ConfigError("dataset manifest required_files is invalid")
    return SourceManifest(schema_version, name, document)


def _require_string_fields(document: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if not isinstance(document.get(field), str) or not document[field]]
    if missing:
        raise ConfigError("source manifest has invalid fields: " + ", ".join(missing))
