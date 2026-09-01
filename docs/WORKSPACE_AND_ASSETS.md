# Workspace and Assets

## Purpose

The repository separates versioned implementation material from machine-local
assets. Source code, configuration, schemas, tests, and design documentation
belong in Git. Downloaded repositories, datasets, native builds, traces,
profiles, checkpoints, and generated results belong in the local workspace.

## Workspace Discovery

The workspace root is selected in the following order:

1. An explicit command-line `--workspace` path
2. The `GALA_WORKSPACE` environment variable
3. `<repository>/workspace`

Repository discovery searches upward for both `pyproject.toml` and `configs/`,
then falls back to the installed package location. The standard layout is:

```text
workspace/
  upstream/
  datasets/
  build/
  cache/
  traces/
  profiles/
  results/
  archive/
```

The complete directory is ignored. No file below it is force-added to Git.

## Portable Paths

Tracked configuration uses repository-relative paths. A relative path in a
manifest is resolved against that manifest or the repository root according to
its schema. Commands may accept an explicit absolute path for externally stored
assets, but such a path is never a tracked default.

Generated local manifests use repository- or workspace-relative references
when the target is within those roots. Environment snapshots may contain the
actual interpreter or compiler path because they describe one execution; those
snapshots remain under the ignored workspace.

## Asset Catalog

Model manifests live in `configs/models/` and dataset manifests in
`configs/datasets/`. A catalog entry defines immutable provenance rather than a
progress state.

| Identifier | Source |
| --- | --- |
| `r2_gaussian` | Pinned R2-Gaussian author repository |
| `fact_gs` | Pinned FaCT-GS author repository |
| `exact_gs` | Pinned Exact-GS author repository |
| `gr_gaussian` | Independent implementation of DOI `10.1109/TCI.2026.3701585` |

| Dataset identifier | Source and license |
| --- | --- |
| `chest` | R2-Gaussian Chest package; underlying LIDC-IDRI terms apply |
| `walnut` | Zenodo record `6986012`, CC BY 4.0 |
| `hdtomo_usb` | `usb.zip` from Zenodo record `4822516`, CC BY 4.0 |

The GR-Gaussian implementation is authored in this repository from the
published description. It is not represented as author-released source.

## Acquisition Rules

`gala-sim acquire` downloads only URLs declared in the catalog. It uses a
partial file for resumable transfers, checks free space before a transfer,
verifies the publisher checksum when present, and computes SHA-256 after the
download completes. Git repositories are checked out in detached mode at the
declared commit.

Archives are extracted only after validation. Absolute paths, parent-directory
entries, device files, and escaping symbolic links are rejected. Source
archives are retained in `workspace/cache/`; prepared inputs are written to
`workspace/datasets/` without modifying the downloaded source.

Acquisition reports are local records under `workspace/`. The public
repository does not contain a completion table or generated result inventory.

## Dataset Normalization

Dataset adapters expose projections, scanner geometry, view partitions,
intensity transforms, and an optional reference volume through one manifest.
Conversion is deterministic and preserves a provenance link to the raw asset.
Validation covers array shape, numeric range, finite values, angle convention,
detector dimensions, and source-to-object/source-to-detector distances.

Small test fixtures validate parsers and transformations. They are testing
inputs only and are never substituted for a catalogued dataset in a generated
campaign.

## Storage and Migration

Moving an existing workspace on the same filesystem must use a directory
rename, not a copy. Before migration, verify that no process has an open file
or working directory below the source. Historical local outputs may be retained
under `workspace/archive/legacy/`; they stay ignored and are not part of the
published repository.
