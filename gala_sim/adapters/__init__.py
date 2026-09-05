"""Model adapter contracts; concrete adapters must preserve official math."""

from .protocol import ModelAdapter, PreparedRun, ReferenceArtifact, TraceArtifact
from .r2_gaussian import R2GaussianAdapter, R2GaussianChestAdapter, TraceCaptureUnavailable
from .native_reference import NativeReferenceError, run_native_reference
from .chest import ChestDatasetManifest, ProjectionRecord, load_chest_manifest
from .datasets import (
    DatasetAdapter, DatasetDescriptor, DatasetManifest, DatasetProjection,
    ScannerGeometry, dataset_descriptors, ensure_projection_initialization,
    get_dataset_adapter,
)
from .registry import (
    CommandModelAdapter, ModelDescriptor, get_model_adapter, model_descriptors,
    prepare_campaign,
)

__all__ = ["ModelAdapter", "PreparedRun", "ReferenceArtifact", "TraceArtifact",
           "R2GaussianAdapter", "R2GaussianChestAdapter", "TraceCaptureUnavailable", "NativeReferenceError",
           "run_native_reference", "ChestDatasetManifest", "ProjectionRecord",
           "load_chest_manifest", "DatasetAdapter", "DatasetDescriptor",
           "DatasetManifest", "DatasetProjection", "ScannerGeometry",
           "dataset_descriptors", "ensure_projection_initialization",
           "get_dataset_adapter", "CommandModelAdapter",
           "ModelDescriptor", "get_model_adapter", "model_descriptors",
           "prepare_campaign"]
