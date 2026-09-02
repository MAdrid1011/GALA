from __future__ import annotations

from pathlib import Path

import pytest

from gala_sim.adapters.trace_process import TraceProcessError, run_trace_process
from gala_sim.tools.preflight import ComputeProcess, GpuSample


def _sample(
    *, processes: tuple[ComputeProcess, ...] = (), memory_used: int = 0,
    memory_total: int | None = None,
) -> GpuSample:
    return GpuSample(
        0.0, 0.0, memory_used, None, None, None, None,
        compute_processes=processes, memory_total_bytes=memory_total,
    )


def test_trace_process_does_not_start_while_external_gpu_work_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    started = False

    def unexpected_popen(*args: object, **kwargs: object) -> object:
        nonlocal started
        started = True
        raise AssertionError("trace process must not start")

    monkeypatch.setattr("gala_sim.adapters.trace_process.subprocess.Popen", unexpected_popen)
    external = ComputeProcess("GPU-fixture", 42, "external", 1024)
    prepared = False

    def prepare() -> None:
        nonlocal prepared
        prepared = True

    with pytest.raises(TraceProcessError, match="gpu_busy_external"):
        run_trace_process(
            ("python", "train.py"), cwd=tmp_path, environment={},
            trace_root=tmp_path / "trace", sample_fn=lambda: _sample(processes=(external,)),
            prepare_fn=prepare,
        )
    assert not started
    assert not prepared


def test_trace_process_runs_environment_preflight_before_gpu_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    order: list[str] = []

    def preflight() -> None:
        order.append("preflight")

    def sample() -> GpuSample:
        order.append("gpu_sample")
        external = ComputeProcess("GPU-fixture", 42, "external", 1024)
        return _sample(processes=(external,))

    monkeypatch.setattr(
        "gala_sim.adapters.trace_process.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trace process must not start")
        ),
    )
    with pytest.raises(TraceProcessError, match="gpu_busy_external"):
        run_trace_process(
            ("python", "train.py"), cwd=tmp_path, environment={},
            trace_root=tmp_path / "trace", sample_fn=sample,
            preflight_fn=preflight,
        )
    assert order == ["preflight", "gpu_sample"]


def test_trace_process_does_not_sample_or_launch_after_environment_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "gala_sim.adapters.trace_process.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trace process must not start")
        ),
    )

    def fail() -> None:
        raise TraceProcessError("model_environment_preflight_failed")

    with pytest.raises(TraceProcessError, match="model_environment_preflight_failed"):
        run_trace_process(
            ("python", "train.py"), cwd=tmp_path, environment={},
            trace_root=tmp_path / "trace",
            sample_fn=lambda: (_ for _ in ()).throw(
                AssertionError("GPU must not be sampled")
            ),
            preflight_fn=fail,
        )


def test_trace_process_requires_two_consecutive_idle_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    samples = iter((
        _sample(),
        _sample(processes=(ComputeProcess("GPU-fixture", 42, "external", 1024),)),
    ))
    prepared = False

    def prepare() -> None:
        nonlocal prepared
        prepared = True

    monkeypatch.setattr(
        "gala_sim.adapters.trace_process.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trace process must not start")
        ),
    )
    with pytest.raises(TraceProcessError, match="gpu_busy_external"):
        run_trace_process(
            ("python", "train.py"), cwd=tmp_path, environment={},
            trace_root=tmp_path / "trace", sample_fn=lambda: next(samples),
            prepare_fn=prepare, sleep_fn=lambda _seconds: None,
        )
    assert not prepared


def test_trace_process_terminates_its_own_inactive_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class Clock:
        value = 0.0

        def now(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    class Process:
        pid = 987654321

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode if self.returncode is not None else 0

    clock = Clock()
    monkeypatch.setattr("gala_sim.adapters.trace_process.subprocess.Popen", Process)
    with pytest.raises(TraceProcessError, match="watchdog_inactivity_timeout"):
        run_trace_process(
            ("python", "train.py"), cwd=tmp_path, environment={},
            trace_root=tmp_path / "trace", inactivity_timeout_seconds=2.0,
            sample_interval_seconds=1.0, sample_fn=lambda: _sample(),
            monotonic_fn=clock.now, wall_time_fn=clock.now, sleep_fn=clock.sleep,
            host_memory_fn=lambda: None,
        )
    report = (tmp_path / "trace/capture_process.json").read_text(encoding="utf-8")
    assert '"failure": "watchdog_inactivity_timeout"' in report


def test_trace_process_stops_before_exhausting_gpu_memory_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class Process:
        pid = 4242

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode if self.returncode is not None else 0

    samples = iter((
        _sample(), _sample(),
        _sample(
            processes=(ComputeProcess("GPU-fixture", 4242, "train", 15),),
            memory_used=15, memory_total=16,
        ),
    ))
    monkeypatch.setattr("gala_sim.adapters.trace_process.subprocess.Popen", Process)

    with pytest.raises(TraceProcessError, match="gpu_memory_reserve_exhausted"):
        run_trace_process(
            ("python", "train.py"), cwd=tmp_path, environment={},
            trace_root=tmp_path / "trace", sample_fn=lambda: next(samples),
            sample_interval_seconds=1.0, minimum_free_memory_bytes=2,
            sleep_fn=lambda _seconds: None, host_memory_fn=lambda: None,
        )
    report = (tmp_path / "trace/capture_process.json").read_text(encoding="utf-8")
    assert '"failure": "gpu_memory_reserve_exhausted"' in report


def test_trace_process_stops_before_exhausting_host_memory_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class Process:
        pid = 4242

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode if self.returncode is not None else 0

    samples = iter((
        _sample(), _sample(),
        _sample(processes=(ComputeProcess("GPU-fixture", 4242, "train", 1),)),
    ))
    monkeypatch.setattr("gala_sim.adapters.trace_process.subprocess.Popen", Process)

    with pytest.raises(TraceProcessError, match="host_memory_reserve_exhausted"):
        run_trace_process(
            ("python", "train.py"), cwd=tmp_path, environment={},
            trace_root=tmp_path / "trace", sample_fn=lambda: next(samples),
            sample_interval_seconds=1.0, minimum_available_host_memory_bytes=8,
            sleep_fn=lambda _seconds: None, host_memory_fn=lambda: 7,
        )
    report = (tmp_path / "trace/capture_process.json").read_text(encoding="utf-8")
    assert '"host_memory_available_bytes": 7' in report
    assert '"failure": "host_memory_reserve_exhausted"' in report
