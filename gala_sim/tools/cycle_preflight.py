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

    binding_ready = (
        isinstance(memory_backend, Ramulator2Backend)
        and callable(getattr(memory_backend.binding, "submit", None))
    )
    checks["ramulator2_binding"] = binding_ready
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
