"""Model adapter contracts; concrete adapters must preserve official math."""

from .protocol import ModelAdapter, PreparedRun, ReferenceArtifact, TraceArtifact
from .r2_gaussian import R2GaussianChestAdapter, TraceCaptureUnavailable
from .chest import ChestDatasetManifest, ProjectionRecord, load_chest_manifest

__all__ = ["ModelAdapter", "PreparedRun", "ReferenceArtifact", "TraceArtifact",
           "R2GaussianChestAdapter", "TraceCaptureUnavailable", "ChestDatasetManifest",
           "ProjectionRecord", "load_chest_manifest"]
