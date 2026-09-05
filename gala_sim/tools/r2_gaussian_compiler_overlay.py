"""Prepare an isolated R2-Gaussian CUDA extension with compiler controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
\t// The render loop has one logical Gaussian per active warp.  Use the
\t// warp shuffle reduction to avoid constructing a cooperative-group object
\t// for every Gaussian/pixel contribution while retaining the active mask.
\tconst unsigned mask = __activemask();
\tif (__popc(mask) == 1)
\t{
\t\tatomicAdd(target, value);
\t\treturn;
\t}
\tconst unsigned leader = static_cast<unsigned>(__ffs(mask) - 1);
\tconst unsigned member_count = __popc(mask);
\tconst unsigned contiguous_mask = member_count == 32
\t\t? 0xffffffffu
\t\t: (((1u << member_count) - 1u) << leader);
\tconst unsigned linear_lane =
\t\t(threadIdx.x + blockDim.x * threadIdx.y +
\t\t blockDim.x * blockDim.y * threadIdx.z) & 31u;
\tfloat sum = value;
\tif (mask == contiguous_mask)
\t{
\t\tfor (unsigned offset = 1; offset < 32; offset <<= 1)
\t\t{
\t\t\tconst float peer = __shfl_down_sync(mask, sum, offset);
\t\t\tif (linear_lane + offset < 32
\t\t\t\t&& (mask & (1u << (linear_lane + offset))))
\t\t\t\tsum += peer;
\t\t}
\t}
\telse
\t{
\t\tsum = 0.0f;
\t\tunsigned remaining = mask;
\t\twhile (remaining)
\t\t{
\t\t\tconst unsigned source_lane =
\t\t\t\tstatic_cast<unsigned>(__ffs(remaining) - 1);
\t\t\tconst float source_value = __shfl_sync(mask, value, source_lane);
\t\t\tif (linear_lane == leader)
\t\t\t\tsum += source_value;
\t\t\tremaining &= remaining - 1;
\t\t}
\t}
\tif (linear_lane == leader)
\t\tatomicAdd(target, sum);
}

template <typename Group>
__device__ __forceinline__ void accumulate_gaussian_gradient_batch(
\tconst Group& active,
\tfloat* target0, float value0, float* target1, float value1,
\tfloat* target2, float value2, float* target3, float value3,
\tfloat* target4, float value4, float* target5, float value5,
\tfloat* target6, float value6, float* target7, float value7,
\tfloat* target8, float value8, float* target9, float value9)
{
\tconst unsigned mask = __activemask();
\tif (__popc(mask) == 1)
\t{
\t\tatomicAdd(target0, value0); atomicAdd(target1, value1);
\t\tatomicAdd(target2, value2); atomicAdd(target3, value3);
\t\tatomicAdd(target4, value4); atomicAdd(target5, value5);
\t\tatomicAdd(target6, value6); atomicAdd(target7, value7);
\t\tatomicAdd(target8, value8); atomicAdd(target9, value9);
\t\treturn;
\t}
\tconst unsigned lane =
\t\t(threadIdx.x + blockDim.x * threadIdx.y +
\t\t blockDim.x * blockDim.y * threadIdx.z) & 31u;
\tconst unsigned leader = static_cast<unsigned>(__ffs(mask) - 1);
\tfloat sum0 = value0, sum1 = value1, sum2 = value2, sum3 = value3;
\tfloat sum4 = value4, sum5 = value5, sum6 = value6, sum7 = value7;
\tfloat sum8 = value8, sum9 = value9;
\tfor (unsigned offset = 1; offset < 32; offset <<= 1)
\t{
\t\tconst float peer0 = __shfl_down_sync(mask, sum0, offset);
\t\tconst float peer1 = __shfl_down_sync(mask, sum1, offset);
\t\tconst float peer2 = __shfl_down_sync(mask, sum2, offset);
\t\tconst float peer3 = __shfl_down_sync(mask, sum3, offset);
\t\tconst float peer4 = __shfl_down_sync(mask, sum4, offset);
\t\tconst float peer5 = __shfl_down_sync(mask, sum5, offset);
\t\tconst float peer6 = __shfl_down_sync(mask, sum6, offset);
\t\tconst float peer7 = __shfl_down_sync(mask, sum7, offset);
\t\tconst float peer8 = __shfl_down_sync(mask, sum8, offset);
\t\tconst float peer9 = __shfl_down_sync(mask, sum9, offset);
\t\tif (lane + offset < 32 && (mask & (1u << (lane + offset))))
\t\t{
\t\t\tsum0 += peer0; sum1 += peer1; sum2 += peer2; sum3 += peer3;
\t\t\tsum4 += peer4; sum5 += peer5; sum6 += peer6; sum7 += peer7;
\t\t\tsum8 += peer8; sum9 += peer9;
\t\t}
\t}
\tif (lane == leader)
\t{
\t\tatomicAdd(target0, sum0); atomicAdd(target1, sum1);
\t\tatomicAdd(target2, sum2); atomicAdd(target3, sum3);
\t\tatomicAdd(target4, sum4); atomicAdd(target5, sum5);
\t\tatomicAdd(target6, sum6); atomicAdd(target7, sum7);
\t\tatomicAdd(target8, sum8); atomicAdd(target9, sum9);
\t}
}

static bool compiler_flag_enabled(const char* name)
{
\tconst char* value = std::getenv(name);
\treturn value != nullptr && value[0] == '1' && value[1] == '\\0';
}
"""

# The legacy checked-out source already has the common helper block.  Keep
# the batch helper separately addressable so an idempotent rebuild can inject
# only the new definition when it has the fused call but not its definition.
_BATCH_HELPER = _HELPERS.split(
    "template <typename Group>\n__device__ __forceinline__ void accumulate_gaussian_gradient_batch",
    1,
)[1]
_BATCH_HELPER = (
    "template <typename Group>\n__device__ __forceinline__ void "
    "accumulate_gaussian_gradient_batch" + _BATCH_HELPER.split(
        "static bool compiler_flag_enabled", 1
    )[0]
)

_LEGACY_REDUCTION = """\tconst float sum = cg::reduce(active, value, cg::plus<float>());
\tif (active.thread_rank() == 0)
\t\tatomicAdd(target, sum);"""

_WARP_REDUCTION = """\t// The render loop has one logical Gaussian per active warp.  Use the
\t// warp shuffle primitive to avoid constructing a cooperative-group object
\t// for every Gaussian/pixel contribution while retaining the active mask.
\tconst unsigned mask = __activemask();
\tif (__popc(mask) == 1)
\t{
\t\tatomicAdd(target, value);
\t\treturn;
\t}
\tconst unsigned linear_lane =
\t\t(threadIdx.x + blockDim.x * threadIdx.y +
\t\t blockDim.x * blockDim.y * threadIdx.z) & 31u;
\tconst unsigned leader = static_cast<unsigned>(__ffs(mask) - 1);
\tconst unsigned member_count = __popc(mask);
\tconst unsigned contiguous_mask = member_count == 32
\t\t? 0xffffffffu
\t\t: (((1u << member_count) - 1u) << leader);
\tif (mask == contiguous_mask)
\t{
\t\tfloat sum = value;
\t\tfor (unsigned offset = 1; offset < 32; offset <<= 1)
\t\t{
\t\t\tconst float peer = __shfl_down_sync(mask, sum, offset);
\t\t\tif (linear_lane + offset < 32
\t\t\t\t&& (mask & (1u << (linear_lane + offset))))
\t\t\t\tsum += peer;
\t\t}
\t\tif (linear_lane == leader)
\t\t\tatomicAdd(target, sum);
\t\treturn;
\t}
\tunsigned remaining = mask;
\tfloat sum = 0.0f;
\twhile (remaining)
\t{
\t\tconst unsigned source_lane =
\t\t\tstatic_cast<unsigned>(__ffs(remaining) - 1);
\t\tconst float source_value = __shfl_sync(mask, value, source_lane);
\t\tif (linear_lane == leader)
\t\t\tsum += source_value;
\t\tremaining &= remaining - 1;
\t}
\tif (linear_lane == leader)
\t\tatomicAdd(target, sum);"""


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
\t\t\t\taccumulate_gaussian_gradient_batch(active,
\t\t\t\t\t&dL_dmean3D_norm[global_id].x, dL_dG * dG_ddelx * ddelx_dx,
\t\t\t\t\t&dL_dmean3D_norm[global_id].y, dL_dG * dG_ddely * ddely_dy,
\t\t\t\t\t&dL_dmean3D_norm[global_id].z, dL_dG * dG_ddelz * ddelz_dz,
\t\t\t\t\t&dL_dconic3D[global_id * 6 + 0], -0.5f * gdx * d.x * dL_dG,
\t\t\t\t\t&dL_dconic3D[global_id * 6 + 1], -1.0f * gdx * d.y * dL_dG,
\t\t\t\t\t&dL_dconic3D[global_id * 6 + 2], -1.0f * gdx * d.z * dL_dG,
\t\t\t\t\t&dL_dconic3D[global_id * 6 + 3], -0.5f * gdy * d.y * dL_dG,
\t\t\t\t\t&dL_dconic3D[global_id * 6 + 4], -1.0f * gdy * d.z * dL_dG,
\t\t\t\t\t&dL_dconic3D[global_id * 6 + 5], -0.5f * gdz * d.z * dL_dG,
\t\t\t\t\t&dL_dopacity[global_id], G * dL_dalpha);
\t\t\t}
\t\t\telse
\t\t\t{
{baseline}
\t\t\t}"""


_LEGACY_SEMANTIC_TARGETS = (
    "&dL_dmean3D_norm[global_id].x",
    "&dL_dmean3D_norm[global_id].y",
    "&dL_dmean3D_norm[global_id].z",
    "&dL_dconic3D[global_id * 6 + 0]",
    "&dL_dconic3D[global_id * 6 + 1]",
    "&dL_dconic3D[global_id * 6 + 2]",
    "&dL_dconic3D[global_id * 6 + 3]",
    "&dL_dconic3D[global_id * 6 + 4]",
    "&dL_dconic3D[global_id * 6 + 5]",
    "&dL_dopacity[global_id]",
)


def _fuse_legacy_semantic_calls(source: str) -> str:
    """Fuse the earlier ten-call semantic block when rebuilding an overlay."""

    matches: list[re.Match[str]] = []
    for target in _LEGACY_SEMANTIC_TARGETS:
        match = re.search(
            r"(?P<indent>[ \\t]*)accumulate_gaussian_gradient\(active,\s*"
            + re.escape(target)
            + r",\s*(?P<value>[^;]+)\);",
            source,
        )
        if match is None:
            return source
        matches.append(match)
    matches.sort(key=lambda item: item.start())
    values = [match.group("value").strip() for match in matches]
    indent = matches[0].group("indent")
    replacement = (
        f"{indent}accumulate_gaussian_gradient_batch(active,\n"
        + ",\n".join(
            f"{indent}\t{target}, {value}"
            for target, value in zip(_LEGACY_SEMANTIC_TARGETS, values)
        )
        + ");"
    )
    return source[:matches[0].start()] + replacement + source[matches[-1].end():]


def _render_backward_overlay(
    source: str,
    *,
    block_start: str,
    block_end: str,
    aggregate_template: str,
    environment_control: str,
) -> tuple[str, tuple[str, ...]]:
    # Some checked-out R2 workspaces contain the earlier single-control
    # overlay. Keep that source usable as an input: preserve its launch and
    # compile-time control, but replace the expensive cooperative-group
    # reduction with the native warp primitive. This makes rebuilding an
    # overlay deterministic without requiring a destructive checkout reset.
    if (
        "template <uint32_t C, bool Aggregate>" in source
        and "accumulate_gaussian_gradient" in source
    ):
        if source.count(_LEGACY_REDUCTION) == 1:
            transformed = source.replace(_LEGACY_REDUCTION, _WARP_REDUCTION, 1)
        elif "__reduce_add_sync" in source:
            transformed = source
        else:
            raise R2CompilerOverlayError(
                "overlay transform compiler-aggregation-helpers expected one source fragment"
            )
        if environment_control == "GALA_SEMANTIC_WARP_REDUCE":
            transformed = _fuse_legacy_semantic_calls(transformed)
            if (
                "accumulate_gaussian_gradient_batch" in transformed
                and transformed.count("accumulate_gaussian_gradient_batch") == 1
            ):
                namespace_anchor = "namespace cg = cooperative_groups;\n"
                if transformed.count(namespace_anchor) != 1:
                    raise R2CompilerOverlayError(
                        "overlay transform compiler-aggregation-helpers expected one namespace anchor"
                    )
                transformed = transformed.replace(
                    namespace_anchor,
                    namespace_anchor + "\n" + _BATCH_HELPER,
                    1,
                )
        return transformed, (
            "compiler-aggregation-helpers",
            "compile-time-aggregation-control",
            "gradient-aggregation",
            "launch-compiler-control",
        )
    helper_anchor = (
        "#include <cooperative_groups/reduce.h>\n"
        "namespace cg = cooperative_groups;\n"
    )
    if source.count(helper_anchor) == 1:
        transformed = _replace_once(
            source, helper_anchor, _HELPERS, "compiler-aggregation-helpers",
        )
    else:
        # The checked-out upstream submodule may already contain the earlier
        # cooperative-group overlay. Replace only its reduction primitive so
        # rebuilding remains deterministic and idempotent.
        if source.count(_LEGACY_REDUCTION) != 1:
            raise R2CompilerOverlayError(
                "overlay transform compiler-aggregation-helpers expected one source fragment"
            )
        transformed = source.replace(_LEGACY_REDUCTION, _WARP_REDUCTION, 1)

    template_anchor = "template <uint32_t C>\n__global__ void __launch_bounds__"
    if transformed.count(template_anchor) == 1:
        transformed = transformed.replace(
            template_anchor,
            "template <uint32_t C, bool Aggregate>\n__global__ void __launch_bounds__",
            1,
        )
    elif "template <uint32_t C, bool Aggregate>" not in transformed:
        raise R2CompilerOverlayError(
            "overlay transform compile-time-aggregation-control expected one source fragment"
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
    if transformed.count(launch_start) == 1:
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
    elif "renderCUDA<NUM_CHANNELS, true>" not in transformed:
        raise R2CompilerOverlayError(
            "overlay transform launch-compiler-control expected one source span"
        )
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
