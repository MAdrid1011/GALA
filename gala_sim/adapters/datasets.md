# datasets.py

## External Interfaces

`DatasetManifest`, `DatasetProjection`, and `ScannerGeometry` form the common
projection and geometry contract. `DatasetAdapter` defines `load`, `validate`,
and `convert` operations.

`get_dataset_adapter(id)` returns adapters for `chest`, `walnut`, or
`hdtomo_usb`. `dataset_descriptors()` exposes their stable catalog metadata.

## Internal Helpers

Exported-data helpers read JSON or publisher text metadata, discover NumPy and
TIFF arrays, convert angles to radians, validate geometry and partitions, and
write prepared NumPy data without modifying source files. Walnut conversion
parses the published INI keys, derives object-space extent, applies the declared
detector-center correction, and reserves every eighth view for testing.

The HDTomo-USB adapter reads the nested Xradia recipe streams for scanner
distances and angular range. A `recon/` TIFF sequence is treated as one volume
and is copied to a NumPy memory map one slice at a time. Initialization samples
large reference volumes through a bounded downsampled view.
