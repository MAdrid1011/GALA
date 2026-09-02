"""Prepare an isolated R2-Gaussian CUDA extension with compiler controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


EXTENSION_RELATIVE = Path(
    "r2_gaussian/submodules/xray-gaussian-rasterization-voxelization"
)
RASTER_BACKWARD_RELATIVE = Path("cuda_rasterizer/backward.cu")
VOXEL_BACKWARD_RELATIVE = Path("cuda_voxelizer/backward.cu")


class R2CompilerOverlayError(RuntimeError):
    """The R2-Gaussian source cannot produce the compiler overlay."""


def _replace_once(source: str, expected: str, replacement: str, transform_id: str) -> str:
    count = source.count(expected)
    if count != 1:
        raise R2CompilerOverlayError(
            f"overlay transform {transform_id} expected one source fragment; found {count}"
        )
    return source.replace(expected, replacement, 1)


def _replace_span_once(
    source: str, start: str, end: str, replacement: str, transform_id: str,
) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise R2CompilerOverlayError(
            f"overlay transform {transform_id} expected one source span"
        )
    begin = source.index(start)
    finish = source.index(end, begin) + len(end)
    return source[:begin] + replacement + source[finish:]


_HELPERS = """#include <cooperative_groups/reduce.h>
#include <cstdlib>
namespace cg = cooperative_groups;

template <typename Group>
__device__ __forceinline__ void accumulate_gaussian_gradient(
\tconst Group& active, float* target, float value)
{
\tconst float sum = cg::reduce(active, value, cg::plus<float>());
\tif (active.thread_rank() == 0)
\t\tatomicAdd(target, sum);
}

static bool compiler_flag_enabled(const char* name)
{
\tconst char* value = std::getenv(name);
\treturn value != nullptr && value[0] == '1' && value[1] == '\\0';
}
"""


_QUERY_AGGREGATE_BLOCK = """\t\t\tif constexpr (Aggregate)
\t\t\t{
\t\t\t\tauto active = cg::coalesced_threads();
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic2D[global_id].y, -1.0f * gdx * d.y * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dopacity[global_id], mu * G * dL_dalpha);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dmu[global_id], con_o.w * G * dL_dalpha);
\t\t\t}
\t\t\telse
\t\t\t{
{baseline}
\t\t\t}"""


_SEMANTIC_AGGREGATE_BLOCK = """\t\t\tif constexpr (Aggregate)
\t\t\t{
\t\t\t\tauto active = cg::coalesced_threads();
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dmean3D_norm[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dmean3D_norm[global_id].y, dL_dG * dG_ddely * ddely_dy);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dmean3D_norm[global_id].z, dL_dG * dG_ddelz * ddelz_dz);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic3D[global_id * 6 + 0], -0.5f * gdx * d.x * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic3D[global_id * 6 + 1], -1.0f * gdx * d.y * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic3D[global_id * 6 + 2], -1.0f * gdx * d.z * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic3D[global_id * 6 + 3], -0.5f * gdy * d.y * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic3D[global_id * 6 + 4], -1.0f * gdy * d.z * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dconic3D[global_id * 6 + 5], -0.5f * gdz * d.z * dL_dG);
\t\t\t\taccumulate_gaussian_gradient(active, &dL_dopacity[global_id], G * dL_dalpha);
\t\t\t}
\t\t\telse
\t\t\t{
{baseline}
\t\t\t}"""


def _render_backward_overlay(
    source: str,
    *,
    block_start: str,
    block_end: str,
    aggregate_template: str,
    environment_control: str,
) -> tuple[str, tuple[str, ...]]:
    transformed = _replace_once(
        source,
        "#include <cooperative_groups/reduce.h>\nnamespace cg = cooperative_groups;\n",
        _HELPERS,
        "compiler-aggregation-helpers",
    )
    transformed = _replace_once(
        transformed,
        "template <uint32_t C>\n__global__ void __launch_bounds__",
        "template <uint32_t C, bool Aggregate>\n__global__ void __launch_bounds__",
        "compile-time-aggregation-control",
    )
    if transformed.count(block_start) != 1 or transformed.count(block_end) != 1:
        raise R2CompilerOverlayError(
            "overlay transform gradient-aggregation expected one source span"
        )
    begin = transformed.index(block_start)
    finish = transformed.index(block_end, begin) + len(block_end)
    baseline = transformed[begin:finish]
    transformed = _replace_span_once(
        transformed,
        block_start,
        block_end,
        aggregate_template.replace("{baseline}", baseline),
        "gradient-aggregation",
    )
    launch_start = "\trenderCUDA<NUM_CHANNELS> << <grid, block >> >("
    launch_end = "\t\t);"
    if transformed.count(launch_start) != 1:
        raise R2CompilerOverlayError(
            "overlay transform launch-compiler-control expected one source span"
        )
    launch_begin = transformed.index(launch_start)
    launch_finish = transformed.index(launch_end, launch_begin) + len(launch_end)
    baseline_launch = transformed[launch_begin:launch_finish]
    enabled_launch = baseline_launch.replace(
        "renderCUDA<NUM_CHANNELS>", "renderCUDA<NUM_CHANNELS, true>", 1,
    )
    disabled_launch = baseline_launch.replace(
        "renderCUDA<NUM_CHANNELS>", "renderCUDA<NUM_CHANNELS, false>", 1,
    )
    controlled_launch = (
        f"\tif (compiler_flag_enabled(\"{environment_control}\"))\n"
        f"{enabled_launch}\n\telse\n{disabled_launch}"
    )
    transformed = transformed[:launch_begin] + controlled_launch + transformed[launch_finish:]
    return transformed, (
        "compiler-aggregation-helpers",
        "compile-time-aggregation-control",
        "gradient-aggregation",
        "launch-compiler-control",
    )


def render_raster_backward_overlay(source: str) -> tuple[str, tuple[str, ...]]:
    """Add the query-side aggregation control to the projection backward pass."""

    return _render_backward_overlay(
        source,
        block_start=(
            "\t\t\t// Update gradients w.r.t. 2D mean position of the Gaussian\n"
        ),
        block_end=(
            "\t\t\tatomicAdd(&(dL_dmu[global_id]), con_o.w * G * dL_dalpha);"
        ),
        aggregate_template=_QUERY_AGGREGATE_BLOCK,
        environment_control="GALA_QUERY_WARP_REDUCE",
    )


def render_voxel_backward_overlay(source: str) -> tuple[str, tuple[str, ...]]:
    """Add the semantic aggregation control to the volume backward pass."""

    return _render_backward_overlay(
        source,
        block_start=(
            "\t\t\tatomicAdd(&dL_dmean3D_norm[global_id].x, "
            "dL_dG * dG_ddelx * ddelx_dx);"
        ),
        block_end="\t\t\tatomicAdd(&dL_dopacity[global_id], G * dL_dalpha);",
        aggregate_template=_SEMANTIC_AGGREGATE_BLOCK,
        environment_control="GALA_SEMANTIC_WARP_REDUCE",
    )


def prepare_overlay(source_root: Path, output_root: Path) -> Path:
    """Copy the official extension and apply both independent controls."""

    source_extension = Path(source_root) / EXTENSION_RELATIVE
    required = (
        source_extension / "setup.py",
        source_extension / RASTER_BACKWARD_RELATIVE,
        source_extension / VOXEL_BACKWARD_RELATIVE,
        source_extension / "third_party/glm/glm/glm.hpp",
    )
    if not all(path.is_file() for path in required):
        raise R2CompilerOverlayError(
            f"R2-Gaussian CUDA extension or GLM submodule is incomplete: {source_extension}"
        )
    if output_root.exists():
        raise R2CompilerOverlayError(f"overlay output already exists: {output_root}")
    shutil.copytree(
        source_extension,
        output_root,
        ignore=shutil.ignore_patterns(
            ".git", "build", "*.egg-info", "__pycache__", "*.pyc", "*.so",
        ),
    )
    files = {
        "raster": output_root / RASTER_BACKWARD_RELATIVE,
        "voxel": output_root / VOXEL_BACKWARD_RELATIVE,
    }
    transforms: dict[str, list[str]] = {}
    digests: dict[str, dict[str, str]] = {}
    for name, path in files.items():
        original = path.read_text(encoding="utf-8")
        if name == "raster":
            generated, ids = render_raster_backward_overlay(original)
        else:
            generated, ids = render_voxel_backward_overlay(original)
        path.write_text(generated, encoding="utf-8")
        transforms[name] = list(ids)
        digests[name] = {
            "source_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "generated_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
        }
    manifest = {
        "schema_version": "gala-r2-gaussian-compiler-overlay-v1",
        "source": EXTENSION_RELATIVE.as_posix(),
        "hash_policy": "record_only_no_hash_rejection",
        "files": digests,
        "transforms": transforms,
        "environment_controls": {
            "query": "GALA_QUERY_WARP_REDUCE",
            "semantic": "GALA_SEMANTIC_WARP_REDUCE",
        },
    }
    (output_root / "gala-overlay-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return output_root


def build_overlay(
    output_root: Path,
    python_executable: Path,
    *,
    cxx: Path | None = None,
    nvcc: Path | None = None,
) -> Path:
    """Build one isolated extension without modifying the official checkout."""

    environment = os.environ.copy()
    environment["MAX_JOBS"] = "1"
    selected_cxx = cxx or (
        Path(candidate) if (candidate := shutil.which("g++-11")) else None
    )
    if selected_cxx is not None:
        selected_cxx = Path(selected_cxx).resolve()
        if not selected_cxx.is_file():
            raise R2CompilerOverlayError(f"C++ compiler is unavailable: {selected_cxx}")
        environment["CXX"] = str(selected_cxx)
        gcc = selected_cxx.with_name(selected_cxx.name.replace("g++", "gcc", 1))
        if gcc.is_file():
            environment["CC"] = str(gcc)
            environment["CUDAHOSTCXX"] = str(gcc)
    selected_nvcc = Path(nvcc).resolve() if nvcc is not None else None
    if selected_nvcc is not None:
        if not selected_nvcc.is_file():
            raise R2CompilerOverlayError(f"NVCC is unavailable: {selected_nvcc}")
        environment["NVCC"] = str(selected_nvcc)
        environment["CUDA_HOME"] = str(selected_nvcc.parent.parent)
        environment["PATH"] = os.pathsep.join((
            str(selected_nvcc.parent), environment.get("PATH", ""),
        ))
    command = [str(python_executable), "setup.py", "build_ext", "--inplace"]
    completed = subprocess.run(
        command,
        cwd=output_root,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    build_report = {
        "schema_version": "gala-cuda-overlay-build-v1",
        "command": command,
        "cxx": str(selected_cxx) if selected_cxx is not None else environment.get("CXX"),
        "nvcc": str(selected_nvcc) if selected_nvcc is not None else environment.get("NVCC"),
        "max_jobs": 1,
        "returncode": completed.returncode,
    }
    (output_root / "gala-build-manifest.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    if completed.returncode:
        combined = "\n".join(
            value for value in (completed.stdout, completed.stderr) if value
        )
        raise R2CompilerOverlayError(
            "R2-Gaussian compiler overlay build failed:\n" + combined[-12000:]
        )
    candidates = tuple(
        (output_root / "xray_gaussian_rasterization_voxelization").glob("_C*.so")
    )
    if len(candidates) != 1:
        raise R2CompilerOverlayError(
            "R2-Gaussian overlay build produced no unique extension"
        )
    return candidates[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cxx", type=Path)
    parser.add_argument("--nvcc", type=Path)
    args = parser.parse_args(argv)
    try:
        output = prepare_overlay(args.source_root.resolve(), args.output_root.resolve())
        extension = build_overlay(
            output,
            args.python.resolve(),
            cxx=args.cxx.resolve() if args.cxx is not None else None,
            nvcc=args.nvcc.resolve() if args.nvcc is not None else None,
        ) if args.build else None
    except (OSError, ValueError, R2CompilerOverlayError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(output),
        "extension": str(extension) if extension is not None else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
