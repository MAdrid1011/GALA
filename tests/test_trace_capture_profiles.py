from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gala_sim.adapters import trace_capture
from gala_sim.adapters.trace_capture import TraceSession, get_trace_hook_profile


def test_exact_profile_splits_projection_and_voxel_extensions() -> None:
    profile = get_trace_hook_profile("exact_gs")
    assert profile.raster_extension == "exact_gaussian_rasterization"
    assert profile.voxel_extension == "xray_gaussian_rasterization_voxelization"
    assert profile.gaussian_model_module == "exact_gs.gaussian.gaussian_model"
    assert profile.pointwise_loss == "l2_loss"
    assert profile.tv_loss is None


def test_fact_profile_uses_split_pipeline_and_external_ssim_module() -> None:
    profile = get_trace_hook_profile("fact_gs")
    assert profile.raster_extension == "gs_ct_rasterizer.rasterize"
    assert profile.voxel_extension == "gs_voxelizer.voxelize"
    assert profile.gaussian_model_module == "fact_gs.r2_gaussian.gaussian.gaussian_model"
    assert profile.capture_backend == "fact_split"
    assert profile.ssim_module == "fused_ssim"
    assert profile.ssim_loss == "fused_ssim"
    assert profile.tv_loss == "tv_3d_loss"


def test_fact_profile_installs_and_restores_every_official_hook(
    tmp_path: Path, monkeypatch,
) -> None:
    def function(*_args, **_kwargs):
        return None

    class GaussianModel:
        update_learning_rate = function
        training_setup = function
        prune_points = function
        densification_postfix = function
        densify_and_clone = function
        densify_and_split = function
        densify_and_prune = function

    raster_cuda = SimpleNamespace(
        rasterize_forward=function,
        rasterize_backward=function,
        rasterize_backward_per_gaussian=function,
    )
    voxel_cuda = SimpleNamespace(
        voxelize_forward=function,
        voxelize_backward=function,
        voxelize_backward_per_gaussian=function,
    )
    modules = {
        "gs_ct_rasterizer.rasterize": SimpleNamespace(
            rasterize_gaussians=function, _C=raster_cuda,
        ),
        "gs_voxelizer.voxelize": SimpleNamespace(
            voxelize_gaussians=function, _C=voxel_cuda,
        ),
        "fact_gs.r2_gaussian.gaussian.gaussian_model": SimpleNamespace(
            GaussianModel=GaussianModel,
        ),
        "fact_gs.r2_gaussian.utils.loss_utils": SimpleNamespace(
            l1_loss=function, tv_3d_loss=function,
        ),
        "fused_ssim": SimpleNamespace(fused_ssim=function),
    }
    monkeypatch.setattr(
        trace_capture.importlib, "import_module", lambda name: modules[name],
    )
    session = TraceSession(tmp_path / "trace", model_id="fact_gs")

    session.install()
    assert len(session._originals) == 18
    assert modules["gs_ct_rasterizer.rasterize"].rasterize_gaussians is not function
    assert raster_cuda.rasterize_forward is not function
    assert modules["fused_ssim"].fused_ssim is not function

    session.restore()
    assert not session._originals
    assert modules["gs_ct_rasterizer.rasterize"].rasterize_gaussians is function
    assert raster_cuda.rasterize_forward is function
    assert modules["fused_ssim"].fused_ssim is function


def test_trace_session_normalizes_string_output_root(tmp_path: Path) -> None:
    session = TraceSession(str(tmp_path / "trace"), model_id="exact_gs")
    assert session.output_root == tmp_path / "trace"


def test_virtual_capture_preserves_model_and_dataset_provenance(
    tmp_path: Path,
) -> None:
    session = TraceSession(
        tmp_path / "trace", model_id="fact_gs", dataset_name="chest",
        virtual_capture=True,
    )

    consumer = session._virtual_consumer

    assert consumer is not None
    assert consumer.model_id == "fact_gs"
    assert consumer.model_name == "FaCT-GS"
    assert consumer.dataset_id == "chest"
