from __future__ import annotations

import json
from pathlib import Path

import pytest

from gala_sim.tools.r2_gaussian_compiler_overlay import (
    EXTENSION_RELATIVE,
    R2CompilerOverlayError,
    prepare_overlay,
    render_raster_backward_overlay,
    render_voxel_backward_overlay,
)


def _source(block: str) -> str:
    return (
        "#include <cooperative_groups/reduce.h>\n"
        "namespace cg = cooperative_groups;\n"
        "template <uint32_t C>\n"
        "__global__ void __launch_bounds__(1) renderCUDA() {\n"
        f"{block}\n"
        "}\n"
        "void launch() {\n"
        "\trenderCUDA<NUM_CHANNELS> << <grid, block >> >(\n"
        "\t\targument\n"
        "\t\t);\n"
        "}\n"
    )


def _raster_source() -> str:
    return _source(
        "\t\t\t// Update gradients w.r.t. 2D mean position of the Gaussian\n"
        "\t\t\tatomicAdd(&dL_dmean2D[global_id].x, value);\n"
        "\t\t\tatomicAdd(&(dL_dmu[global_id]), con_o.w * G * dL_dalpha);"
    )


def _voxel_source() -> str:
    return _source(
        "\t\t\tatomicAdd(&dL_dmean3D_norm[global_id].x, "
        "dL_dG * dG_ddelx * ddelx_dx);\n"
        "\t\t\tatomicAdd(&dL_dopacity[global_id], G * dL_dalpha);"
    )


def test_r2_overlay_adds_independent_query_and_semantic_controls() -> None:
    raster, raster_transforms = render_raster_backward_overlay(_raster_source())
    voxel, voxel_transforms = render_voxel_backward_overlay(_voxel_source())

    assert raster.count("accumulate_gaussian_gradient(active,") == 7
    assert voxel.count("accumulate_gaussian_gradient(active,") == 10
    assert "GALA_QUERY_WARP_REDUCE" in raster
    assert "GALA_SEMANTIC_WARP_REDUCE" in voxel
    assert "renderCUDA<NUM_CHANNELS, true>" in raster
    assert "renderCUDA<NUM_CHANNELS, false>" in voxel
    assert len(raster_transforms) == len(voxel_transforms) == 4


def test_r2_overlay_rejects_source_drift_without_hash_gate() -> None:
    with pytest.raises(R2CompilerOverlayError, match="expected one source fragment"):
        render_raster_backward_overlay(
            _raster_source().replace("namespace cg = cooperative_groups;\n", "")
        )


def test_prepare_r2_overlay_records_both_generated_files(tmp_path: Path) -> None:
    extension = tmp_path / "source" / EXTENSION_RELATIVE
    (extension / "cuda_rasterizer").mkdir(parents=True)
    (extension / "cuda_voxelizer").mkdir()
    (extension / "third_party/glm/glm").mkdir(parents=True)
    (extension / "setup.py").write_text("# fixture\n", encoding="utf-8")
    (extension / "cuda_rasterizer/backward.cu").write_text(
        _raster_source(), encoding="utf-8",
    )
    (extension / "cuda_voxelizer/backward.cu").write_text(
        _voxel_source(), encoding="utf-8",
    )
    (extension / "third_party/glm/glm/glm.hpp").write_text(
        "// fixture\n", encoding="utf-8",
    )

    output = prepare_overlay(tmp_path / "source", tmp_path / "output")

    manifest = json.loads(
        (output / "gala-overlay-manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["files"]) == {"raster", "voxel"}
    assert manifest["hash_policy"] == "record_only_no_hash_rejection"
    assert manifest["environment_controls"] == {
        "query": "GALA_QUERY_WARP_REDUCE",
        "semantic": "GALA_SEMANTIC_WARP_REDUCE",
    }
