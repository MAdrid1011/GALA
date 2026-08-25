from __future__ import annotations

from pathlib import Path

from gala_sim.results import RunManifest
from gala_sim.results.run import RunOutputWriter
from gala_sim.timing import CycleConfig, CycleEngine, ModuleTiming
from tests.test_trace_cycle import _Memory, _trace


def test_run_output_writer_emits_machine_readable_cycle_files(tmp_path: Path) -> None:
    timing = ModuleTiming(latency=1, initiation_interval=1, queue_capacity=8, ports=1, banks=1)
    config = CycleConfig(
        modules={name: timing for name in (
            "relation_constructor", "fusion_issue", "semantic_cache", "compute_pod",
            "bidirectional_query", "reconstruction_update", "shared_sram",
        )},
        memory=_Memory(), clock_frequency_hz=1, relation_seed_fifo_entries=1,
        candidate_lanes=1,
    )
    result = CycleEngine(config).run(_trace())
    writer = RunOutputWriter(tmp_path / "run")
    writer.write_cycles(result)
    writer.write_quality({"status": "not_run"})
    writer.write_gpu_reference({"status": "not_run"})
    writer.write_manifest(RunManifest(
        run_id="fixture", status="passed", model={}, dataset={}, config_sha256="a" * 64,
        ablation_bits="0000", random_seed=0, repository_commit="b" * 40, environment={},
    ).as_dict())
    writer.write_status("passed")
    assert (tmp_path / "run" / "cycles.json").is_file()
    assert (tmp_path / "run" / "stalls.parquet").is_file()
    assert (tmp_path / "run" / "status.json").is_file()
