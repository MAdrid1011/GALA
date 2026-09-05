from __future__ import annotations

import json
from pathlib import Path

import pytest

from gala_sim.tools.exact_gs_compiler_overlay import (
    _QUERY_TARGETS,
    _SEMANTIC_TARGETS,
    _SEMANTIC_ORDER_SENSITIVE_TARGETS,
    ExactCompilerOverlayError,
    build_overlay,
    prepare_overlay,
    render_exact_backward_overlay,
)


def _source() -> str:
    atomics = "\n".join(
        f"atomicAdd({target}, 1.0f);"
        for target in (*_SEMANTIC_TARGETS, *_QUERY_TARGETS)
    )
    return (
        "#include <cooperative_groups/reduce.h>\n"
        "namespace cg = cooperative_groups;\n"
        "template <uint32_t C>\n"
        "__global__ void __launch_bounds__\n"
        "void kernel(\n"
        "    const  float   DSD\n"
        "\t) {\n"
        f"{atomics}\n"
        "}\n"
        "void launch() {\n"
        "\trenderCUDA<NUM_CHANNELS> << <grid, block >> >(\n"
        "        DSO,  \n"
        "        DSD\n"
        "\t\t);\n"
        "}\n"
    )


def test_exact_overlay_adds_independent_query_and_semantic_controls() -> None:
    transformed, transforms = render_exact_backward_overlay(_source())

    assert transformed.count("compiler_accumulate<SemanticAggregate>(") == 6
    assert transformed.count("compiler_accumulate_query_pair<QueryAggregate>(") == 1
    assert transformed.count("atomicAdd(&dL_dmeans[global_id]") == 3
    assert transformed.count("atomicAdd(&(dL_dopacity[global_id])") == 1
    assert transformed.count("atomicAdd(&(dL_dmu[global_id])") == 1
    assert "renderCUDA<NUM_CHANNELS, true, false>" in transformed
    assert "GALA_QUERY_WARP_REDUCE" in transformed
    assert "GALA_SEMANTIC_WARP_REDUCE" in transformed
    assert "__match_any_sync(active_mask, target_label)" in transformed
    assert "if (__popc(target_mask) == 1)" in transformed
    assert "__shfl_sync" in transformed
    assert "cg::reduce(matching_target" not in transformed
    assert len(transforms) == 13


def test_exact_overlay_rejects_source_drift_without_hash_gate() -> None:
    with pytest.raises(ExactCompilerOverlayError, match="expected one source fragment"):
        render_exact_backward_overlay(_source().replace(
            "#include <cooperative_groups/reduce.h>\n", "",
        ))


def test_prepare_overlay_copies_explicit_glm_and_records_provenance(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    extension = source_root / "exact_gs/submodules/exact-gaussian-rasterization"
    (extension / "cuda_rasterizer").mkdir(parents=True)
    (extension / "setup.py").write_text("# fixture\n", encoding="utf-8")
    (extension / "cuda_rasterizer/backward.cu").write_text(
        _source(), encoding="utf-8",
    )
    glm_root = tmp_path / "glm"
    (glm_root / "glm").mkdir(parents=True)
    (glm_root / "glm/glm.hpp").write_text("// fixture\n", encoding="utf-8")

    output = prepare_overlay(
        source_root,
        tmp_path / "output",
        glm_root=glm_root,
    )

    assert (output / "third_party/glm/glm/glm.hpp").is_file()
    manifest = json.loads(
        (output / "gala-overlay-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["hash_policy"] == "record_only_no_hash_rejection"
    assert manifest["environment_controls"] == {
        "query": "GALA_QUERY_WARP_REDUCE",
        "semantic": "GALA_SEMANTIC_WARP_REDUCE",
    }


def test_exact_build_records_tools_and_keeps_both_failure_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 1
        stdout = "compiler stdout marker"
        stderr = "compiler stderr marker"

    monkeypatch.setattr(
        "gala_sim.tools.exact_gs_compiler_overlay.subprocess.run",
        lambda *_args, **_kwargs: Completed(),
    )
    python = tmp_path / "python"
    python.touch()
    cxx = tmp_path / "g++"
    cxx.touch()
    nvcc = tmp_path / "nvcc"
    nvcc.touch()

    with pytest.raises(ExactCompilerOverlayError) as raised:
        build_overlay(tmp_path, python, cxx=cxx, nvcc=nvcc)

    assert "compiler stdout marker" in str(raised.value)
    assert "compiler stderr marker" in str(raised.value)
    report = json.loads(
        (tmp_path / "gala-build-manifest.json").read_text(encoding="utf-8")
    )
    assert report["cxx"] == str(cxx.resolve())
    assert report["nvcc"] == str(nvcc.resolve())
    assert report["max_jobs"] == 1
