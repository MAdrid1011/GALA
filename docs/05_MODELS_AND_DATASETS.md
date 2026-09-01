# Model Adapters and Datasets

## Model Catalog

| Identifier | Implementation source | Identity |
| --- | --- | --- |
| `r2_gaussian` | R2-Gaussian author repository | `f2579bfddd9aac009cb797c8503bef8119bbd022` |
| `fact_gs` | FaCT-GS author repository | `9b95ea9` pinned by its model manifest |
| `exact_gs` | Exact-GS author repository | `c8f9251` pinned by its model manifest |
| `gr_gaussian` | Independent implementation in this repository | DOI `10.1109/TCI.2026.3701585` |

An official-source adapter invokes the pinned author entry point and preserves
its configuration, initialization, optimizer, densification, split, prune, and
termination behavior. Instrumentation adds task mapping and trace output but
does not replace the model's formulas.

The GR-Gaussian adapter implements the published method independently. Its
manifest records paper provenance and implementation type rather than an
upstream repository claim.

## Dataset Catalog

| Identifier | Public source | Content |
| --- | --- | --- |
| `chest` | R2-Gaussian data release | Cone-beam projections, geometry, and reference volume |
| `walnut` | Zenodo `6986012` | FIPS measured projections and scan metadata |
| `hdtomo_usb` | `usb.zip`, Zenodo `4822516` | TXRM/TIFF projections, metadata, and vendor reconstruction |

Raw assets retain publisher licenses. Conversion is deterministic, read-only
with respect to the source, and records the original checksum, parameters, and
converter version.

## Dataset Interface

```python
class DatasetAdapter(Protocol):
    descriptor: DatasetDescriptor
    def load(self, root: Path) -> DatasetManifest: ...
    def validate(self, manifest: DatasetManifest) -> None: ...
    def convert(self, manifest: DatasetManifest, output: Path) -> DatasetManifest: ...
```

`DatasetManifest` provides projection paths, projection shape and dtype,
angles in radians, volume shape, detector shape, DSO, DSD, intensity transform,
view partitions, and an optional reference volume.

## Model Interface

```python
class ModelAdapter(Protocol):
    descriptor: ModelDescriptor
    def prepare(self, dataset: DatasetManifest, config: ModelConfig) -> PreparedRun: ...
    def run_reference(self, run: PreparedRun) -> ReferenceArtifact: ...
    def capture_trace(self, run: PreparedRun, sink: DeviceTraceSink) -> TraceArtifact: ...
    def replay_reductions(self, run: PreparedRun, order: ReductionOrder) -> Reconstruction: ...
```

The adapter exports stage boundaries, official command construction, trace
hooks, and the mapping from model operations to CLAMP tasks. With tracing
disabled, an official-source adapter must follow the pinned author path.

## Composition

Models and datasets are selected by catalog identifier. Dataset conversion
normalizes scanner geometry without changing measured values. Model-specific
preparation then creates the exact input layout expected by that model.

The registry validates identifiers and compatibility before constructing a
campaign. It does not generate a table of executions or infer that an adapter
has produced a measurement.

## Data Validation

Validation checks archive identity, required files, projection count, shape,
dtype, finite range, angle convention, scanner distances, detector dimensions,
view partitions, and reference-volume geometry. A failed check stops campaign
construction; no synthetic replacement is generated.
