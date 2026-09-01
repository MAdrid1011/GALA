"""Manifest-driven acquisition of model repositories and datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import time
from typing import Any, Iterable, Mapping
from urllib.parse import quote
import zipfile

from gala_sim.config.loader import _yaml_load
from gala_sim.workspace import WorkspacePaths


@dataclass(frozen=True)
class AssetFile:
    name: str
    size: int
    md5: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ModelAsset:
    id: str
    name: str
    implementation: str
    repository: str | None
    commit: str | None
    paper_doi: str | None
    license: str
    official_commands: Mapping[str, tuple[str, ...]]
    stage_boundaries: tuple[str, ...]
    output_strategy: str
    submodules: bool
    exclude_paths: tuple[str, ...]
    manifest: Path


@dataclass(frozen=True)
class DatasetAsset:
    id: str
    name: str
    provider: str
    source_url: str
    license: str
    license_url: str
    files: tuple[AssetFile, ...]
    record: int | None
    source_subpath: str | None
    manifest: Path


@dataclass(frozen=True)
class AssetCatalog:
    models: Mapping[str, ModelAsset]
    datasets: Mapping[str, DatasetAsset]

    @classmethod
    def load(cls, config_root: Path) -> "AssetCatalog":
        root = Path(config_root).resolve()
        models = {
            item.id: item
            for path in sorted((root / "models").glob("*.yaml"))
            for item in (_load_model(path),)
        }
        datasets = {
            item.id: item
            for path in sorted((root / "datasets").glob("*.yaml"))
            for item in (_load_dataset(path),)
        }
        if len(models) != len(list((root / "models").glob("*.yaml"))):
            raise ValueError("duplicate model asset identifier")
        if len(datasets) != len(list((root / "datasets").glob("*.yaml"))):
            raise ValueError("duplicate dataset asset identifier")
        return cls(models, datasets)


@dataclass(frozen=True)
class AssetSelection:
    models: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    all: bool = False


@dataclass(frozen=True)
class AcquisitionItem:
    kind: str
    id: str
    action: str
    destination: str
    sha256: str | None = None


@dataclass(frozen=True)
class AcquisitionReport:
    items: tuple[AcquisitionItem, ...]
    dry_run: bool

    def as_dict(self) -> dict[str, Any]:
        return {"dry_run": self.dry_run, "items": [asdict(item) for item in self.items]}


def _mapping(path: Path, schema: str) -> Mapping[str, Any]:
    document = _yaml_load(path)
    if not isinstance(document, Mapping) or document.get("schema_version") != schema:
        raise ValueError(f"invalid asset manifest: {path}")
    if "status" in document:
        raise ValueError(f"asset manifest must describe provenance, not status: {path}")
    return document


def _load_model(path: Path) -> ModelAsset:
    item = _mapping(path, "gala-model-source-v1")
    implementation = str(item.get("implementation", "upstream"))
    repository = str(item["repository"]) if item.get("repository") else None
    commit = str(item["commit"]) if item.get("commit") else None
    paper_doi = str(item["paper_doi"]) if item.get("paper_doi") else None
    if implementation == "upstream" and (not repository or not commit):
        raise ValueError(f"upstream model manifest lacks repository identity: {path}")
    if implementation == "independent_reimplementation" and not paper_doi:
        raise ValueError(f"independent model manifest lacks paper DOI: {path}")
    raw_commands = item.get("official_commands", {})
    if not isinstance(raw_commands, Mapping):
        raise ValueError(f"model commands must be a mapping: {path}")
    commands = {
        str(name): tuple(str(argument) for argument in arguments)
        for name, arguments in raw_commands.items()
        if isinstance(arguments, list)
    }
    if "train" not in commands:
        raise ValueError(f"model manifest lacks a structured training command: {path}")
    return ModelAsset(
        id=str(item.get("id") or path.stem), name=str(item["name"]),
        implementation=implementation, repository=repository, commit=commit,
        paper_doi=paper_doi, license=str(item.get("license", "unknown")),
        official_commands=commands,
        stage_boundaries=tuple(str(value) for value in item.get("stage_boundaries", ())),
        output_strategy=str(item.get("output_strategy", "explicit_output_argument")),
        submodules=bool(item.get("submodules", False)),
        exclude_paths=tuple(str(value) for value in item.get("exclude_paths", ())),
        manifest=path,
    )


def _load_dataset(path: Path) -> DatasetAsset:
    item = _mapping(path, "gala-dataset-source-v1")
    raw_files = item.get("files")
    if raw_files is None:
        raw_files = [{
            "name": item.get("archive_name") or Path(str(item["source_subpath"])).name,
            "size": item.get("archive_size", 0),
            "sha256": item.get("sha256"),
        }]
    files = tuple(
        AssetFile(
            name=str(entry["name"]), size=int(entry.get("size", 0)),
            md5=str(entry["md5"]) if entry.get("md5") else None,
            sha256=str(entry["sha256"]) if entry.get("sha256") else None,
        )
        for entry in raw_files
    )
    return DatasetAsset(
        id=str(item.get("id") or path.stem), name=str(item["name"]),
        provider=str(item.get("provider", "http")), source_url=str(item["source_url"]),
        license=str(item.get("license", "publisher-terms")),
        license_url=str(item["license_url"]), files=files,
        record=int(item["record"]) if item.get("record") is not None else None,
        source_subpath=str(item["source_subpath"]) if item.get("source_subpath") else None,
        manifest=path,
    )


def _selected(values: Iterable[str], available: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(value for value in values if value))
    unknown = sorted(set(result).difference(available))
    if unknown:
        raise KeyError(f"unknown {kind} assets: {', '.join(unknown)}")
    return result


def acquire_assets(
    catalog: AssetCatalog,
    selection: AssetSelection,
    workspace: WorkspacePaths,
    resume: bool = True,
    *,
    dry_run: bool = False,
) -> AcquisitionReport:
    """Acquire selected assets and write the report only to the workspace."""

    model_ids = tuple(catalog.models) if selection.all else _selected(
        selection.models, catalog.models, "model",
    )
    dataset_ids = tuple(catalog.datasets) if selection.all else _selected(
        selection.datasets, catalog.datasets, "dataset",
    )
    items: list[AcquisitionItem] = []
    for model_id in model_ids:
        model = catalog.models[model_id]
        destination = workspace.upstream / model.id
        action = "clone" if model.repository else "local"
        digest = None
        if not dry_run and model.repository:
            workspace.ensure()
            action = _checkout_repository(model, destination)
            digest = model.commit
        items.append(AcquisitionItem("model", model.id, action, workspace.reference(destination), digest))

    for dataset_id in dataset_ids:
        dataset = catalog.datasets[dataset_id]
        destination = workspace.datasets / dataset.id
        digest = None
        action = "download"
        if not dry_run:
            workspace.ensure()
            digest, action = _acquire_dataset(
                dataset, destination, workspace.cache, resume=resume,
            )
        items.append(AcquisitionItem(
            "dataset", dataset.id, action, workspace.reference(destination), digest,
        ))

    report = AcquisitionReport(tuple(items), dry_run)
    if not dry_run:
        report_path = workspace.cache / "acquisition-report.json"
        report_path.write_text(
            json.dumps(report.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )
    return report


def _checkout_repository(asset: ModelAsset, destination: Path) -> str:
    if asset.repository is None or asset.commit is None:
        return "local"
    existed = (destination / ".git").is_dir()
    if existed:
        subprocess.run(["git", "-C", str(destination), "fetch", "--all", "--tags"], check=True)
    else:
        if destination.exists():
            raise ValueError(f"model destination is not a Git checkout: {destination}")
        subprocess.run([
            "git", "clone", "--filter=blob:none", "--no-checkout",
            asset.repository, str(destination),
        ], check=True)
    if asset.exclude_paths:
        patterns = ["/*", *(f"!/{value.strip('/')}" for value in asset.exclude_paths)]
        subprocess.run(
            ["git", "-C", str(destination), "sparse-checkout", "set", "--no-cone", *patterns],
            check=True,
        )
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", asset.commit], check=True)
    if asset.submodules:
        subprocess.run(
            ["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"],
            check=True,
        )
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != asset.commit:
        raise ValueError(f"model checkout identity mismatch for {asset.id}")
    return "verify" if existed else "clone"


def _acquire_dataset(
    asset: DatasetAsset, destination: Path, cache: Path, *, resume: bool,
) -> tuple[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    source_cache = cache / "downloads" / asset.id
    source_cache.mkdir(parents=True, exist_ok=True)
    existing = _existing_dataset_identity(asset, destination, source_cache)
    if existing is not None:
        return existing, "verify"
    digests: list[str] = []
    if asset.provider == "google_drive":
        files = _download_google_drive(asset, source_cache)
    else:
        files = []
        for entry in asset.files:
            target = source_cache / entry.name
            if asset.provider == "zenodo":
                if asset.record is None:
                    raise ValueError("Zenodo dataset manifest has no record")
                url = (
                    f"https://zenodo.org/api/records/{asset.record}/files/"
                    f"{quote(entry.name)}/content"
                )
            else:
                url = asset.source_url
            download_file(url, target, expected_size=entry.size, resume=resume)
            _verify_file(target, entry)
            files.append(target)
    for path in files:
        digest = _digest(path, "sha256")
        digests.append(digest)
        if path.suffix.lower() == ".zip":
            safe_extract_zip(path, destination)
        else:
            shutil.copy2(path, destination / path.name)
    identity = hashlib.sha256("".join(sorted(digests)).encode("ascii")).hexdigest()
    (destination / "asset-manifest.json").write_text(json.dumps({
        "schema_version": "gala-local-dataset-v1",
        "id": asset.id,
        "license": asset.license,
        "license_url": asset.license_url,
        "source_files": [
            {"name": path.name, "sha256": digest}
            for path, digest in zip(files, digests, strict=True)
        ],
        "identity_sha256": identity,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return identity, "download"


def _existing_dataset_identity(
    asset: DatasetAsset, destination: Path, source_cache: Path,
) -> str | None:
    manifest_path = destination / "asset-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if document.get("id") != asset.id or not isinstance(document.get("identity_sha256"), str):
        return None
    recorded = document.get("source_files")
    if not isinstance(recorded, list) or len(recorded) != len(asset.files):
        return None
    for entry in recorded:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            return None
        path = source_cache / entry["name"]
        if not path.is_file() or _digest(path, "sha256") != entry.get("sha256"):
            return None
    return str(document["identity_sha256"])


def _download_google_drive(asset: DatasetAsset, destination: Path) -> list[Path]:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("Google Drive acquisition requires the assets extra") from error
    downloaded = gdown.download_folder(url=asset.source_url, output=str(destination), quiet=False)
    if not downloaded:
        raise RuntimeError(f"Google Drive returned no files for {asset.id}")
    candidates = [Path(path) for path in downloaded]
    selected = []
    for entry in asset.files:
        match = next((path for path in candidates if path.name == entry.name), None)
        if match is None:
            raise FileNotFoundError(f"downloaded folder lacks {entry.name}")
        _verify_file(match, entry)
        selected.append(match)
    return selected


def download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int = 0,
    resume: bool = True,
    retries: int = 3,
) -> Path:
    """Download one file with a partial-file resume and space preflight."""

    try:
        import requests
    except ImportError as error:
        raise RuntimeError("HTTP acquisition requires the assets extra") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (not expected_size or destination.stat().st_size == expected_size):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if resume and partial.exists() else 0
    remaining = max(0, expected_size - offset)
    if remaining and shutil.disk_usage(destination.parent).free < remaining:
        raise OSError(f"insufficient free space for {destination.name}")
    for attempt in range(retries):
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if expected_size and partial.stat().st_size != expected_size:
                raise OSError(f"download size mismatch for {destination.name}")
            partial.replace(destination)
            return destination
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
            offset = partial.stat().st_size if resume and partial.exists() else 0
    raise RuntimeError("unreachable download retry state")


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: AssetFile) -> None:
    if expected.size and path.stat().st_size != expected.size:
        raise ValueError(f"asset size mismatch: {expected.name}")
    if expected.md5 and _digest(path, "md5") != expected.md5:
        raise ValueError(f"asset MD5 mismatch: {expected.name}")
    if expected.sha256 and _digest(path, "sha256") != expected.sha256:
        raise ValueError(f"asset SHA-256 mismatch: {expected.name}")


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract regular ZIP members without allowing root escape or links."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if (
                name.is_absolute()
                or ".." in name.parts
                or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}
            ):
                raise ValueError(f"unsafe archive member: {member.filename}")
            target = (root / Path(*name.parts)).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError(f"unsafe archive member: {member.filename}") from error
        stream.extractall(root)
