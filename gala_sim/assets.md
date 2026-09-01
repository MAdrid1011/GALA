# assets.py

## External Interfaces

`AssetCatalog.load(config_root)` validates model and dataset provenance
manifests. `AssetSelection` identifies requested catalog entries.

`acquire_assets(catalog, selection, workspace, resume=True, dry_run=False)`
checks out pinned repositories, downloads selected dataset files, verifies
publisher identities, safely extracts archives, and writes a local report.

`download_file()` performs resumable HTTP transfer with free-space and expected
size checks. `safe_extract_zip()` rejects absolute, parent-traversing, device,
and symbolic-link members.

## Internal Helpers

Manifest loaders enforce source identity without progress fields. Repository
checkout uses detached commits. Dataset acquisition keeps source archives in
the workspace cache, computes SHA-256, and writes prepared content separately.
