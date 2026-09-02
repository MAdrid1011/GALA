"""Prepare an isolated Exact-GS CUDA extension with two compiler controls."""

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


EXTENSION_RELATIVE = Path("exact_gs/submodules/exact-gaussian-rasterization")
BACKWARD_RELATIVE = Path("cuda_rasterizer/backward.cu")


class ExactCompilerOverlayError(RuntimeError):
    """The pinned Exact-GS source cannot produce the compiler overlay."""


def _replace_once(source: str, expected: str, replacement: str, transform_id: str) -> str:
    count = source.count(expected)
    if count != 1:
        raise ExactCompilerOverlayError(
            f"overlay transform {transform_id} expected one source fragment; found {count}"
        )
    return source.replace(expected, replacement, 1)


_HELPERS = """#include <cooperative_groups/reduce.h>
#include <cstdlib>
namespace cg = cooperative_groups;

template <typename Group>
__device__ __forceinline__ void compiler_accumulate_group(
	const Group& active, float* target, float value)
{
	const float sum = cg::reduce(active, value, cg::plus<float>());
	if (active.thread_rank() == 0)
		atomicAdd(target, sum);
}

__device__ __forceinline__ void compiler_accumulate(
	const bool aggregate, float* target, float value)
{
	if (aggregate)
	{
		auto active = cg::coalesced_threads();
		compiler_accumulate_group(active, target, value);
	}
	else
	{
		atomicAdd(target, value);
	}
}

static bool compiler_flag_enabled(const char* name)
{
	const char* value = std::getenv(name);
	return value != nullptr && value[0] == '1' && value[1] == '\\0';
}
"""


_SEMANTIC_TARGETS = (
    "&dL_dcov3D[6*global_id+0]",
    "&dL_dcov3D[6*global_id+1]",
    "&dL_dcov3D[6*global_id+2]",
    "&dL_dcov3D[6*global_id+3]",
    "&dL_dcov3D[6*global_id+4]",
    "&dL_dcov3D[6*global_id+5]",
    "&dL_dmeans[global_id].x",
    "&dL_dmeans[global_id].y",
    "&dL_dmeans[global_id].z",
    "&(dL_dopacity[global_id])",
    "&(dL_dmu[global_id])",
)
_QUERY_TARGETS = (
    "&dL_dmean2D[global_id].x",
    "&dL_dmean2D[global_id].y",
)


def render_exact_backward_overlay(source: str) -> tuple[str, tuple[str, ...]]:
    """Add independent query and semantic aggregation to one pinned source."""

    transformed = _replace_once(
        source,
        "#include <cooperative_groups/reduce.h>\nnamespace cg = cooperative_groups;\n",
        _HELPERS,
        "compiler-aggregation-helpers",
    )
    transformed = _replace_once(
        transformed,
        "    const  float   DSD\n\t)",
        "    const float DSD,\n\tconst bool query_aggregate,\n\tconst bool semantic_aggregate\n\t)",
        "kernel-compiler-controls",
    )
    transform_ids = ["compiler-aggregation-helpers", "kernel-compiler-controls"]
    for target in _SEMANTIC_TARGETS:
        expected = f"atomicAdd({target},"
        replacement = f"compiler_accumulate(semantic_aggregate, {target},"
        transformed = _replace_once(
            transformed, expected, replacement, f"semantic-{target}",
        )
        transform_ids.append(f"semantic-{target}")
    for target in _QUERY_TARGETS:
        expected = f"atomicAdd({target},"
        replacement = f"compiler_accumulate(query_aggregate, {target},"
        transformed = _replace_once(
            transformed, expected, replacement, f"query-{target}",
        )
        transform_ids.append(f"query-{target}")
    transformed = _replace_once(
        transformed,
        "        DSO,  \n        DSD\n\t\t);",
        "        DSO,  \n        DSD,\n"
        "\t\tcompiler_flag_enabled(\"GALA_QUERY_WARP_REDUCE\"),\n"
        "\t\tcompiler_flag_enabled(\"GALA_SEMANTIC_WARP_REDUCE\")\n"
        "\t\t);",
        "launch-compiler-controls",
    )
    transform_ids.append("launch-compiler-controls")
    return transformed, tuple(transform_ids)


def _resolve_glm_root(source_root: Path, explicit: Path | None) -> Path:
    candidates = [
        explicit,
        source_root / EXTENSION_RELATIVE / "third_party/glm",
        source_root.parent
        / "r2_gaussian/r2_gaussian/submodules/xray-gaussian-rasterization-voxelization"
        / "third_party/glm",
    ]
    for candidate in candidates:
        if candidate is not None and (Path(candidate) / "glm/glm.hpp").is_file():
            return Path(candidate).resolve()
    raise ExactCompilerOverlayError(
        "GLM headers are unavailable; pass --glm-root containing glm/glm.hpp"
    )


def prepare_overlay(
    source_root: Path, output_root: Path, *, glm_root: Path | None = None,
) -> Path:
    source_extension = Path(source_root) / EXTENSION_RELATIVE
    if not (source_extension / "setup.py").is_file():
        raise ExactCompilerOverlayError(
            f"Exact-GS CUDA extension is missing: {source_extension}"
        )
    if output_root.exists():
        raise ExactCompilerOverlayError(f"overlay output already exists: {output_root}")
    shutil.copytree(
        source_extension,
        output_root,
        ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__", "*.pyc"),
    )
    selected_glm = _resolve_glm_root(Path(source_root), glm_root)
    shutil.copytree(selected_glm, output_root / "third_party/glm")
    backward = output_root / BACKWARD_RELATIVE
    original = backward.read_text(encoding="utf-8")
    transformed, transform_ids = render_exact_backward_overlay(original)
    backward.write_text(transformed, encoding="utf-8")
    manifest = {
        "schema_version": "gala-exact-gs-compiler-overlay-v1",
        "source": EXTENSION_RELATIVE.as_posix(),
        "source_backward_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        "generated_backward_sha256": hashlib.sha256(
            transformed.encode("utf-8")
        ).hexdigest(),
        "hash_policy": "record_only_no_hash_rejection",
        "glm_source": str(selected_glm),
        "transforms": list(transform_ids),
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
    environment = os.environ.copy()
    environment["MAX_JOBS"] = "1"
    selected_cxx = cxx or (
        Path(candidate) if (candidate := shutil.which("g++-11")) else None
    )
    if selected_cxx is not None:
        selected_cxx = Path(selected_cxx).resolve()
        environment["CXX"] = str(selected_cxx)
        gcc = selected_cxx.with_name(selected_cxx.name.replace("g++", "gcc", 1))
        if gcc.is_file():
            environment["CC"] = str(gcc)
            environment["CUDAHOSTCXX"] = str(gcc)
    selected_nvcc = Path(nvcc).resolve() if nvcc is not None else None
    if selected_nvcc is not None:
        if not selected_nvcc.is_file():
            raise ExactCompilerOverlayError(f"NVCC is unavailable: {selected_nvcc}")
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
        raise ExactCompilerOverlayError(
            "Exact-GS compiler overlay build failed:\n" + combined[-12000:]
        )
    candidates = tuple((output_root / "exact_gaussian_rasterization").glob("_C*.so"))
    if len(candidates) != 1:
        raise ExactCompilerOverlayError("Exact-GS overlay build produced no unique extension")
    return candidates[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--glm-root", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cxx", type=Path)
    parser.add_argument("--nvcc", type=Path)
    args = parser.parse_args(argv)
    try:
        output = prepare_overlay(
            args.source_root.resolve(),
            args.output_root.resolve(),
            glm_root=args.glm_root.resolve() if args.glm_root is not None else None,
        )
        extension = build_overlay(
            output,
            args.python.resolve(),
            cxx=args.cxx.resolve() if args.cxx is not None else None,
            nvcc=args.nvcc.resolve() if args.nvcc is not None else None,
        ) if args.build else None
    except (OSError, ValueError, ExactCompilerOverlayError) as error:
        parser.error(str(error))
    print(json.dumps({
        "output": str(output),
        "extension": str(extension) if extension is not None else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
