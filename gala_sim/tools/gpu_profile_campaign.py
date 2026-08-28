"""Validate the frozen R2-Gaussian GPU profiling campaign and its evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from gala_sim.adapters.stage_profile import IterationRange, parse_iteration_range
from gala_sim.identity import sha256_file


CAMPAIGN_SCHEMA_VERSION = "gala-gpu-profiling-campaign-v1"
EVIDENCE_SCHEMA_VERSION = "gala-gpu-profiling-campaign-evidence-v1"


@dataclass(frozen=True)
class RepresentativeIteration:
    iteration: int
    roles: tuple[str, ...]


@dataclass(frozen=True)
class GpuProfileCampaign:
    path: Path
    training_campaign: Path
    calibration_config: Path
    cuda_event_ranges: tuple[IterationRange, ...]
    representatives: tuple[RepresentativeIteration, ...]
    required_stage_roles: Mapping[str, tuple[str, ...]]
    training_iterations: int

    @classmethod
    def load(cls, path: Path) -> "GpuProfileCampaign":
        path = path.resolve()
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or document.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("unsupported GPU profiling campaign")
        repository = path.parents[2]
        training_path = (repository / str(document["training_campaign"])).resolve()
        calibration_path = (repository / str(document["calibration_config"])).resolve()
        training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
        if not isinstance(training, Mapping):
            raise ValueError("training campaign is invalid")
        optimization = training.get("training", {}).get("parameter_groups", {}).get("optimization", {})
        schedule = training.get("training", {}).get("effective_schedule", {})
        if not isinstance(optimization, Mapping) or not isinstance(schedule, Mapping):
            raise ValueError("training campaign has no frozen optimization schedule")
        representatives = tuple(
            RepresentativeIteration(
                int(item["iteration"]), tuple(str(role) for role in item["roles"])
            )
            for item in document.get("representative_iterations", ())
        )
        required = {
            str(stage): tuple(str(role) for role in roles)
            for stage, roles in document.get("required_stage_roles", {}).items()
        }
        campaign = cls(
            path=path,
            training_campaign=training_path,
            calibration_config=calibration_path,
            cuda_event_ranges=tuple(
                parse_iteration_range(str(value))
                for value in document.get("cuda_event_iteration_ranges", ())
            ),
            representatives=representatives,
            required_stage_roles=required,
            training_iterations=int(optimization["iterations"]),
        )
        campaign._validate(optimization, schedule)
        return campaign

    def _validate(self, optimization: Mapping[str, Any], schedule: Mapping[str, Any]) -> None:
        if self.cuda_event_ranges != (IterationRange(1, self.training_iterations),):
            raise ValueError("CUDA Event profiling must cover the complete training run")
        iterations = [item.iteration for item in self.representatives]
        if not iterations or iterations != sorted(set(iterations)):
            raise ValueError("representative iterations must be unique and sorted")
        role_iterations: dict[str, set[int]] = {}
        for item in self.representatives:
            if item.iteration < 1 or item.iteration > self.training_iterations or not item.roles:
                raise ValueError("representative iteration is invalid")
            for role in item.roles:
                role_iterations.setdefault(role, set()).add(item.iteration)
        densify_from = int(optimization["densify_from_iter"])
        densify_until = int(optimization["densify_until_iter"])
        interval = int(optimization["densification_interval"])
        densify_midpoint = (densify_from + densify_until) // 2
        role_bounds = {
            "initial": (1, 1),
            "pre_densification": (2, densify_from),
            "early_densification": (densify_from + 1, densify_midpoint),
            "late_densification": (densify_midpoint + 1, densify_until - 1),
            "post_densification": (densify_until, self.training_iterations),
        }
        for role, (start, end) in role_bounds.items():
            if role not in role_iterations or any(
                iteration < start or iteration > end
                for iteration in role_iterations[role]
            ):
                raise ValueError(f"profiling role {role} is outside its frozen phase")
        collection = role_iterations.get("collection", set())
        if not collection or any(
            iteration <= densify_from or iteration >= densify_until or iteration % interval
            for iteration in collection
        ):
            raise ValueError("collection representatives do not follow the frozen schedule")
        evaluations = set(int(value) for value in schedule.get("test_iterations", ()))
        if role_iterations.get("periodic_evaluation") != evaluations:
            raise ValueError("profiling campaign must cover every frozen evaluation iteration")
        if role_iterations.get("final_reconstruction") != {self.training_iterations}:
            raise ValueError("profiling campaign must cover final reconstruction")
        required_roles = {role for roles in self.required_stage_roles.values() for role in roles}
        if not self.required_stage_roles or not required_roles.issubset(role_iterations):
            raise ValueError("required stage roles have no representative iteration")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "status": "passed",
            "campaign_path": str(self.path),
            "campaign_sha256": sha256_file(self.path),
            "training_campaign_path": str(self.training_campaign),
            "training_campaign_sha256": sha256_file(self.training_campaign),
            "calibration_config_path": str(self.calibration_config),
            "calibration_config_sha256": sha256_file(self.calibration_config),
            "training_iterations": self.training_iterations,
            "cuda_event_iteration_ranges": [
                f"{item.start}:{item.end}" for item in self.cuda_event_ranges
            ],
            "nsys_capture_mode": "cudaProfilerApi",
            "nsys_capture_range_end": f"repeat:{len(self.representatives)}",
            "representative_iterations": [
                {"iteration": item.iteration, "roles": list(item.roles)}
                for item in self.representatives
            ],
            "required_stage_roles": {
                stage: list(roles) for stage, roles in self.required_stage_roles.items()
            },
            "formal_performance_eligible": False,
            "eligibility_reason": "campaign_plan_only_pending_measured_evidence",
        }

    def validate_evidence(
        self, stage_profile: Mapping[str, Any], nsys_profiles: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        reasons: list[str] = []
        expected_iterations = {item.iteration for item in self.representatives}
        timing_coverage = stage_profile.get("requested_iteration_coverage")
        timing_ranges = stage_profile.get("iteration_ranges")
        timing_identity = stage_profile.get("run_identity")
        timing_campaign = (
            timing_identity.get("profiling_campaign")
            if isinstance(timing_identity, Mapping) else None
        )
        expected_ranges = [
            {"start": item.start, "end": item.end} for item in self.cuda_event_ranges
        ]
        if (
            stage_profile.get("status") != "passed"
            or not isinstance(timing_coverage, Mapping)
            or timing_coverage.get("status") != "complete"
            or timing_ranges != expected_ranges
            or not isinstance(timing_campaign, Mapping)
            or timing_campaign.get("profile_mode") != "full_timing"
        ):
            reasons.append("full_cuda_event_timing_incomplete")
        calls_by_iteration: dict[int, set[str]] = {}
        coverage_records: list[dict[str, Any]] = []
        seen_iterations: set[int] = set()
        source_hashes: list[str] = []
        for profile in nsys_profiles:
            run_identity = profile.get("run_identity")
            if not isinstance(run_identity, Mapping):
                reasons.append("nsys_run_identity_missing")
            coverage = profile.get("kernel_coverage")
            if (
                profile.get("status") != "passed"
                or not isinstance(coverage, Mapping)
                or coverage.get("status") != "complete"
            ):
                reasons.append("nsys_kernel_coverage_incomplete")
                continue
            for record in coverage.get("records", ()):
                if not isinstance(record, Mapping):
                    continue
                iteration = int(record["iteration"])
                if iteration in seen_iterations:
                    reasons.append(f"duplicate_nsys_iteration:{iteration}")
                seen_iterations.add(iteration)
                coverage_records.append(dict(record))
            for call in profile.get("stage_calls", ()):
                if isinstance(call, Mapping):
                    calls_by_iteration.setdefault(int(call["iteration"]), set()).add(
                        str(call["stage"])
                    )
            source_hash = profile.get("source_sha256")
            if isinstance(source_hash, str):
                source_hashes.append(source_hash)
        for iteration in sorted(expected_iterations - seen_iterations):
            reasons.append(f"missing_nsys_iteration:{iteration}")
        for iteration in sorted(seen_iterations - expected_iterations):
            reasons.append(f"unexpected_nsys_iteration:{iteration}")
        role_iterations: dict[str, set[int]] = {}
        for item in self.representatives:
            for role in item.roles:
                role_iterations.setdefault(role, set()).add(item.iteration)
        stage_role_coverage: dict[str, dict[str, list[int]]] = {}
        for stage, roles in self.required_stage_roles.items():
            stage_role_coverage[stage] = {}
            for role in roles:
                matched = sorted(
                    iteration for iteration in role_iterations[role]
                    if stage in calls_by_iteration.get(iteration, set())
                )
                stage_role_coverage[stage][role] = matched
                if not matched:
                    reasons.append(f"missing_stage_role:{stage}:{role}")
        kernel_count = sum(int(record.get("kernel_count", 0)) for record in coverage_records)
        assigned_count = sum(
            int(record.get("assigned_kernel_count", 0)) for record in coverage_records
        )
        exact_coverage = (
            not any(
                reason.startswith((
                    "nsys_", "duplicate_nsys_", "missing_nsys_", "unexpected_nsys_",
                ))
                for reason in reasons
            )
            and seen_iterations == expected_iterations
        )
        status = "passed" if not reasons else "provisional_profiling_evidence"
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": status,
            "formal_performance_eligible": False,
            "campaign_sha256": sha256_file(self.path),
            "stage_profile_schema_version": stage_profile.get("schema_version"),
            "nsys_source_sha256": sorted(source_hashes),
            "kernel_coverage": {
                "status": "complete" if exact_coverage else "incomplete",
                "iteration_count": len(coverage_records),
                "kernel_count": kernel_count,
                "assigned_kernel_count": assigned_count,
                "unassigned_kernel_count": sum(
                    int(record.get("unassigned_kernel_count", 0))
                    for record in coverage_records
                ),
                "multiply_assigned_kernel_count": sum(
                    int(record.get("multiply_assigned_kernel_count", 0))
                    for record in coverage_records
                ),
                "records": sorted(coverage_records, key=lambda item: int(item["iteration"])),
            },
            "stage_role_coverage": stage_role_coverage,
            "reasons": sorted(set(reasons)),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_profile_campaign")
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-profile", type=Path)
    parser.add_argument("--nsys-profile", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    campaign = GpuProfileCampaign.load(args.campaign)
    if (args.stage_profile is None) != (not args.nsys_profile):
        raise ValueError("evidence validation requires stage and NSYS profiles together")
    result = (
        campaign.validate_evidence(
            json.loads(args.stage_profile.read_text(encoding="utf-8")),
            [json.loads(path.read_text(encoding="utf-8")) for path in args.nsys_profile],
        )
        if args.stage_profile is not None else campaign.manifest()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
