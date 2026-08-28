"""Formal cycle-run preflight and machine-readable failure records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gala_sim.config import GalaConfig, pending_parameters
from gala_sim.timing.memory import Ramulator2Backend
from gala_sim.timing.resources import ResourceEnvelope, ResourceUsage


@dataclass(frozen=True)
class CyclePreflightReport:
    status: str
    reason: str | None
    checks: dict[str, Any]
    pending: tuple[str, ...]
    missing_bindings: tuple[str, ...]
    reproduction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "checks": self.checks,
            "pending": list(self.pending),
            "missing_bindings": list(self.missing_bindings),
            "reproduction": self.reproduction,
        }


def run_cycle_preflight(
    config: GalaConfig,
    *,
    memory_backend: Any = None,
    resource_usage: ResourceUsage | None = None,
    reproduction: str,
) -> CyclePreflightReport:
    """Check all inputs that must be frozen before a formal cycle run.

    A recorded completion table is useful for unit tests and experimental smoke,
    but it is not a formal LPDDR5 command model and therefore cannot satisfy the
    Ramulator 2 check.
    """

    pending = tuple(pending_parameters(config.parameters))
    missing: list[str] = []
    checks: dict[str, Any] = {
        "configuration_ready": not pending,
        "configuration_sha256": config.sha256,
    }
    if pending:
        missing.append("configuration_parameters")

    required_binding_methods = (
        "metadata", "try_issue", "tick", "drain_completions", "clone",
    )
    binding_ready = isinstance(memory_backend, Ramulator2Backend) and all(
        callable(getattr(memory_backend.binding, name, None))
        for name in required_binding_methods
    )
    binding_metadata: dict[str, Any] | None = None
    binding_reason: str | None = None
    if not binding_ready:
        binding_reason = "async_binding_api_missing"
    if binding_ready:
        try:
            binding_metadata = dict(memory_backend.metadata())
            implementation = str(binding_metadata["implementation"])
            version = str(binding_metadata["version"])
            channels = int(binding_metadata["channels"])
            transaction_bytes = int(binding_metadata["transaction_bytes"])
            channel_width_bits = int(binding_metadata["channel_width_bits"])
            data_rate_mtps = int(binding_metadata["data_rate_mtps"])
            if implementation != "Ramulator 2" or not version:
                raise ValueError("Ramulator metadata identity is incomplete")
            if version != str(config.value("memory.ramulator_version")):
                raise ValueError("Ramulator version disagrees with configuration")
            if channels != int(config.value("memory.channels")):
                raise ValueError("Ramulator channel count disagrees with configuration")
            if transaction_bytes != int(config.value("cache.sector_bytes")):
                raise ValueError("Ramulator transaction size disagrees with configuration")
            if channel_width_bits != int(config.value("memory.channel_width_bits")):
                raise ValueError("Ramulator channel width disagrees with configuration")
            if data_rate_mtps != int(config.value("memory.data_rate")):
                raise ValueError("Ramulator data rate disagrees with configuration")
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            binding_ready = False
            binding_reason = str(error)
    checks["ramulator2_binding"] = binding_ready
    if binding_reason is not None:
        checks["ramulator2_reason"] = binding_reason
    if binding_metadata is not None:
        checks["ramulator2_metadata"] = binding_metadata
    if not binding_ready:
        missing.append("ramulator2_binding")

    resource_ready = False
    resource_reason: str | None = None
    try:
        envelope = ResourceEnvelope.from_gala(config)
        if resource_usage is None:
            resource_reason = "resource_usage_snapshot_missing"
        else:
            envelope.check(resource_usage)
            resource_ready = True
    except (KeyError, TypeError, ValueError) as error:
        resource_reason = str(error)
    checks["resource_envelope"] = resource_ready
    if resource_reason is not None:
        checks["resource_envelope_reason"] = resource_reason
    if not resource_ready:
        missing.append("resource_envelope")

    status = "passed" if not missing else "failed_preflight"
    return CyclePreflightReport(
        status=status,
        reason=None if status == "passed" else missing[0],
        checks=checks,
        pending=pending,
        missing_bindings=tuple(missing),
        reproduction=reproduction,
    )


def write_cycle_preflight(report: CyclePreflightReport, output: Path) -> None:
    """Write the preflight report and the workflow status atomically enough for audit."""

    import json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report.as_dict(), ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    (output / "preflight.json").write_text(encoded, encoding="utf-8")
    status = {
        "status": report.status,
        "reason": report.reason,
        "checks": report.checks,
        "reproduction": report.reproduction,
    }
    (output / "status.json").write_text(
        json.dumps(status, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
