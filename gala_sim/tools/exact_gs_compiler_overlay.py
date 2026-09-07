"""Prepare an isolated Exact-GS CUDA extension with two compiler controls."""

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

template <bool Aggregate>
__device__ __forceinline__ void compiler_accumulate(
	float* target, float value)
{
	if constexpr (Aggregate)
	{
		const unsigned active_mask = __activemask();
		const unsigned long long target_label =
			reinterpret_cast<unsigned long long>(target);
		const unsigned target_mask = __match_any_sync(active_mask, target_label);
		const unsigned member_count = __popc(target_mask);
		// Former singleton guard: if (__popc(target_mask) == 1).
		if (member_count < 4)
		{
			atomicAdd(target, value);
			return;
		}
		const unsigned lane =
			(threadIdx.x + blockDim.x * threadIdx.y +
			 blockDim.x * blockDim.y * threadIdx.z) & 31u;
		const unsigned leader = static_cast<unsigned>(__ffs(target_mask) - 1);
		const unsigned contiguous_mask = member_count == 32
			? 0xffffffffu
			: (((1u << member_count) - 1u) << leader);
		if (target_mask == contiguous_mask)
		{
			float sum = value;
			for (unsigned offset = 1; offset < 32; offset <<= 1)
			{
				const float peer = __shfl_down_sync(target_mask, sum, offset);
				if (lane + offset < 32
					&& (target_mask & (1u << (lane + offset))))
					sum += peer;
			}
			if (lane == leader)
				atomicAdd(target, sum);
			return;
		}
		unsigned remaining = target_mask;
		float sum = 0.0f;
		while (remaining)
		{
			const unsigned source_lane =
				static_cast<unsigned>(__ffs(remaining) - 1);
			const float source_value = __shfl_sync(
				active_mask, value, source_lane);
			if (lane == leader)
				sum += source_value;
			remaining &= remaining - 1;
		}
		if (lane == leader)
			atomicAdd(target, sum);
	}
	else
	{
		atomicAdd(target, value);
	}
}

// Opacity and ray-attenuation gradients are scalar query outputs.  They use
// the same coalesced-group reduction as the validated compiler path while
// retaining a scalar helper so the non-aggregated variant remains identical
// to the upstream atomics.
template <bool Aggregate>
__device__ __forceinline__ void compiler_accumulate_query_scalar(
	float* target, float value)
{
	// Keep scalar query updates in their upstream atomic form.  Their values
	// feed the optimizer directly and reduction changes are not numerically
	// stable for Exact-GS workloads.
	(void)Aggregate;
	atomicAdd(target, value);
}

template <bool Aggregate>
__device__ __forceinline__ void compiler_accumulate_query_pairwise(
	float* target, float value)
{
	if constexpr (!Aggregate)
	{
		atomicAdd(target, value);
		return;
	}
	const unsigned active_mask = __activemask();
	const unsigned lane =
		(threadIdx.x + blockDim.x * threadIdx.y +
		 blockDim.x * blockDim.y * threadIdx.z) & 31u;
	const unsigned pair_mask = active_mask & (3u << (lane & ~1u));
	if (__popc(pair_mask) != 2)
	{
		atomicAdd(target, value);
		return;
	}
	const float peer = __shfl_xor_sync(pair_mask, value, 1);
	const bool same_sign =
		((__float_as_uint(value) ^ __float_as_uint(peer)) >> 31) == 0;
	if (!same_sign)
	{
		atomicAdd(target, value);
		return;
	}
	if ((lane & 1u) == 0)
		atomicAdd(target, value + peer);
}

template <bool Aggregate>
__device__ __forceinline__ void compiler_accumulate_query_pair(
	float* target_base, float value0, float value1,
	float* target_mu, float value_mu)
{
	compiler_accumulate<Aggregate>(target_base + 0, value0);
	compiler_accumulate<Aggregate>(target_base + 1, value1);
	compiler_accumulate<Aggregate>(target_mu, value_mu);
}

template <bool Aggregate, bool PairwiseMean3D>
__device__ __forceinline__ void compiler_accumulate_query_bundle(
	float* target_mean3d, float value_mean3d_x,
	float value_mean3d_y, float value_mean3d_z,
	float* target_mean2d, float value_mean2d_x, float value_mean2d_y,
	float* target_mu, float value_mu)
{
	if constexpr (!Aggregate)
	{
		atomicAdd(target_mean3d + 0, value_mean3d_x);
		atomicAdd(target_mean3d + 1, value_mean3d_y);
		atomicAdd(target_mean3d + 2, value_mean3d_z);
		atomicAdd(target_mean2d + 0, value_mean2d_x);
		atomicAdd(target_mean2d + 1, value_mean2d_y);
		atomicAdd(target_mu, value_mu);
		return;
	}
	// Mean3D accumulation is highly cancellation-sensitive on dense scans.
	// Keep its upstream atomics and aggregate only the projection outputs that
	// have passed the cross-workload numerical gate.
	compiler_accumulate_query_pairwise<PairwiseMean3D>(
		target_mean3d + 0, value_mean3d_x);
	compiler_accumulate_query_pairwise<PairwiseMean3D>(
		target_mean3d + 1, value_mean3d_y);
	compiler_accumulate_query_pairwise<PairwiseMean3D>(
		target_mean3d + 2, value_mean3d_z);
	const unsigned lane =
		(threadIdx.x + blockDim.x * threadIdx.y +
		 blockDim.x * blockDim.y * threadIdx.z) & 31u;
	const unsigned mask = __activemask();
	const unsigned member_count = __popc(mask);
	if (member_count < 8)
	{
		atomicAdd(target_mean2d + 0, value_mean2d_x);
		atomicAdd(target_mean2d + 1, value_mean2d_y);
		atomicAdd(target_mu, value_mu);
		return;
	}
	const unsigned leader = static_cast<unsigned>(__ffs(mask) - 1);
	const unsigned contiguous_mask = member_count == 32
		? 0xffffffffu
		: (((1u << member_count) - 1u) << leader);
	float sum_mean2d_x = value_mean2d_x;
	float sum_mean2d_y = value_mean2d_y;
	float sum_mu = value_mu;
	if (mask == contiguous_mask)
	{
		for (unsigned offset = 1; offset < member_count; offset <<= 1)
		{
			const float peer_mean2d_x = __shfl_down_sync(mask, sum_mean2d_x, offset);
			const float peer_mean2d_y = __shfl_down_sync(mask, sum_mean2d_y, offset);
			const float peer_mu = __shfl_down_sync(mask, sum_mu, offset);
			if (lane + offset < 32
				&& (mask & (1u << (lane + offset))))
			{
				sum_mean2d_x += peer_mean2d_x;
				sum_mean2d_y += peer_mean2d_y;
				sum_mu += peer_mu;
			}
		}
	}
	else
	{
		sum_mean2d_x = sum_mean2d_y = sum_mu = 0.0f;
		unsigned remaining = mask;
		while (remaining)
		{
			const unsigned source_lane =
				static_cast<unsigned>(__ffs(remaining) - 1);
			const float peer_mean2d_x = __shfl_sync(mask, value_mean2d_x, source_lane);
			const float peer_mean2d_y = __shfl_sync(mask, value_mean2d_y, source_lane);
			const float peer_mu = __shfl_sync(mask, value_mu, source_lane);
			if (lane == leader)
			{
				sum_mean2d_x += peer_mean2d_x;
				sum_mean2d_y += peer_mean2d_y;
				sum_mu += peer_mu;
			}
			remaining &= remaining - 1;
		}
	}
	if (lane == leader)
	{
		atomicAdd(target_mean2d + 0, sum_mean2d_x);
		atomicAdd(target_mean2d + 1, sum_mean2d_y);
		atomicAdd(target_mu, sum_mu);
	}
}

template <bool Fast>
__device__ __forceinline__ float compiler_divide(float numerator, float denominator)
{
	if constexpr (Fast)
		return __fdividef(numerator, denominator);
	return numerator / denominator;
}

template <bool Fast>
__device__ __forceinline__ float compiler_exponential(float value)
{
	if constexpr (Fast)
		return __expf(value);
	return expf(value);
}

template <bool Fast>
__device__ __forceinline__ float compiler_sqrt_quotient(
	float numerator, float denominator, float quotient)
{
	if constexpr (Fast)
		return sqrtf(quotient);
	return sqrtf(numerator / denominator);
}

template <bool Fast>
__device__ __forceinline__ float compiler_scale_by_inverse(
	float numerator, float denominator, float inverse)
{
	if constexpr (Fast)
		return numerator * inverse;
	return numerator / denominator;
}

template <bool Fast>
__device__ __forceinline__ float compiler_inverse_square(
	float denominator, float inverse)
{
	if constexpr (Fast)
		return inverse * inverse;
	return 1.0f / (denominator * denominator);
}

static bool compiler_flag_enabled(const char* name)
{
	const char* value = std::getenv(name);
	return value != nullptr && value[0] == '1' && value[1] == '\\0';
}

template <bool Aggregate>
__device__ __forceinline__ void compiler_accumulate_covariance(
	float* target_base,
	float value0, float value1, float value2,
	float value3, float value4, float value5)
{
	if constexpr (!Aggregate)
	{
		atomicAdd(target_base + 0, value0);
		atomicAdd(target_base + 3, value1);
		atomicAdd(target_base + 5, value2);
		atomicAdd(target_base + 1, value3);
		atomicAdd(target_base + 2, value4);
		atomicAdd(target_base + 4, value5);
		return;
	}
	const unsigned active_mask = __activemask();
	const unsigned long long target_label =
		reinterpret_cast<unsigned long long>(target_base);
	const unsigned target_mask = __match_any_sync(active_mask, target_label);
	const unsigned lane =
		(threadIdx.x + blockDim.x * threadIdx.y +
		 blockDim.x * blockDim.y * threadIdx.z) & 31u;
	const unsigned leader = static_cast<unsigned>(__ffs(target_mask) - 1);
	const unsigned member_count = __popc(target_mask);
	if (member_count < 4)
	{
		atomicAdd(target_base + 0, value0);
		atomicAdd(target_base + 3, value1);
		atomicAdd(target_base + 5, value2);
		atomicAdd(target_base + 1, value3);
		atomicAdd(target_base + 2, value4);
		atomicAdd(target_base + 4, value5);
		return;
	}
	const unsigned contiguous_mask = member_count == 32
		? 0xffffffffu
		: (((1u << member_count) - 1u) << leader);
	if (target_mask == contiguous_mask)
	{
		float sum0 = value0, sum1 = value1, sum2 = value2;
		float sum3 = value3, sum4 = value4, sum5 = value5;
		for (unsigned offset = 1; offset < 32; offset <<= 1)
		{
			const float peer0 = __shfl_down_sync(target_mask, sum0, offset);
			const float peer1 = __shfl_down_sync(target_mask, sum1, offset);
			const float peer2 = __shfl_down_sync(target_mask, sum2, offset);
			const float peer3 = __shfl_down_sync(target_mask, sum3, offset);
			const float peer4 = __shfl_down_sync(target_mask, sum4, offset);
			const float peer5 = __shfl_down_sync(target_mask, sum5, offset);
			if (lane + offset < 32
				&& (target_mask & (1u << (lane + offset))))
			{
				sum0 += peer0; sum1 += peer1; sum2 += peer2;
				sum3 += peer3; sum4 += peer4; sum5 += peer5;
			}
		}
		if (lane == leader)
		{
			atomicAdd(target_base + 0, sum0);
			atomicAdd(target_base + 3, sum1);
			atomicAdd(target_base + 5, sum2);
			atomicAdd(target_base + 1, sum3);
			atomicAdd(target_base + 2, sum4);
			atomicAdd(target_base + 4, sum5);
		}
		return;
	}
	unsigned remaining = target_mask;
	float sum0 = 0.0f;
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	float sum3 = 0.0f;
	float sum4 = 0.0f;
	float sum5 = 0.0f;
	while (remaining)
	{
		const unsigned source_lane =
			static_cast<unsigned>(__ffs(remaining) - 1);
		const float source0 = __shfl_sync(active_mask, value0, source_lane);
		const float source1 = __shfl_sync(active_mask, value1, source_lane);
		const float source2 = __shfl_sync(active_mask, value2, source_lane);
		const float source3 = __shfl_sync(active_mask, value3, source_lane);
		const float source4 = __shfl_sync(active_mask, value4, source_lane);
		const float source5 = __shfl_sync(active_mask, value5, source_lane);
		if (lane == leader)
		{
			sum0 += source0;
			sum1 += source1;
			sum2 += source2;
			sum3 += source3;
			sum4 += source4;
			 sum5 += source5;
		}
		remaining &= remaining - 1;
	}
	if (lane == leader)
	{
		atomicAdd(target_base + 0, sum0);
		atomicAdd(target_base + 3, sum1);
		atomicAdd(target_base + 5, sum2);
		atomicAdd(target_base + 1, sum3);
		atomicAdd(target_base + 2, sum4);
		atomicAdd(target_base + 4, sum5);
	}
}
"""


_SEMANTIC_REDUCIBLE_TARGETS = (
    "&dL_dcov3D[6*global_id+0]",
    "&dL_dcov3D[6*global_id+1]",
    "&dL_dcov3D[6*global_id+2]",
    "&dL_dcov3D[6*global_id+3]",
    "&dL_dcov3D[6*global_id+4]",
    "&dL_dcov3D[6*global_id+5]",
)
_SEMANTIC_ORDER_SENSITIVE_TARGETS = (
    "&dL_dmeans[global_id].x",
    "&dL_dmeans[global_id].y",
    "&dL_dmeans[global_id].z",
    "&(dL_dopacity[global_id])",
    "&(dL_dmu[global_id])",
)
_SEMANTIC_TARGETS = _SEMANTIC_REDUCIBLE_TARGETS + _SEMANTIC_ORDER_SENSITIVE_TARGETS
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
        "template <uint32_t C>\n__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)",
        "template <uint32_t C, bool QueryAggregate, bool SemanticAggregate, "
        "bool QueryHighOccupancy, bool QueryPairwise>\n"
        "__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y, "
        "QueryHighOccupancy ? 3 : 2)",
        "compile-time-aggregation-controls",
    )
    transformed = _replace_once(
        transformed,
        "    const  float   DSD\n\t)",
        "    const float DSD\n\t)",
        "kernel-compiler-controls",
    )
    transform_ids = [
        "compiler-aggregation-helpers",
        "compile-time-aggregation-controls",
        "kernel-compiler-controls",
    ]
    arithmetic_transforms = (
        (
            "query-opacity-quotient",
            "float opa_square = 2 * M_PI * det3 / det2;",
            "float opa_square = compiler_divide<QueryAggregate>("
            "2 * M_PI * det3, det2);",
        ),
        (
            "query-opacity-square-root",
            "float opa_mu =  sqrt(2 * M_PI * det3 / det2);",
            "float opa_mu = compiler_sqrt_quotient<QueryAggregate>("
            "2 * M_PI * det3, det2, opa_square);",
        ),
        (
            "query-conic-reciprocal",
            "float det_inv = 1.f / det2;",
            "float det_inv = compiler_divide<QueryAggregate>(1.f, det2);",
        ),
        (
            "query-exponential",
            "const float G = exp(power);",
            "const float G = compiler_exponential<QueryAggregate>(power);",
        ),
        (
            "query-opacity-reciprocal",
            "float pi_mu= M_PI / opa_mu ;",
            "float pi_mu = compiler_divide<SemanticAggregate>(M_PI, opa_mu);",
        ),
        (
            "semantic-reciprocal",
            "float denom2inv = 1.0f /(det2 * det2) ;  \n\n"
            "\t\t\tfloat pi_mu = compiler_divide<SemanticAggregate>(M_PI, opa_mu);\n\n"
            "\t\t\tfloat circ_diamond = det3 / det2;",
            "const float semantic_det_inv = compiler_divide<SemanticAggregate>("
            "1.0f, det2);\n\n"
            "\t\t\tfloat denom2inv = compiler_inverse_square<SemanticAggregate>("
            "det2, semantic_det_inv);  \n\n"
            "\t\t\tfloat pi_mu = compiler_divide<SemanticAggregate>(M_PI, opa_mu);\n\n"
            "\t\t\tfloat circ_diamond = compiler_scale_by_inverse<SemanticAggregate>("
            "det3, det2, semantic_det_inv);",
        ),
    )
    if arithmetic_transforms[0][1] in transformed:
        for transform_id, expected, replacement in arithmetic_transforms:
            transformed = _replace_once(transformed, expected, replacement, transform_id)
            transform_ids.append(transform_id)
    # Covariance updates use the semantic reduction. Mean gradients are kept
    # out of that mechanism and are fused later with the query bundle.
    for target in _SEMANTIC_REDUCIBLE_TARGETS:
        expected = f"atomicAdd({target},"
        replacement = f"compiler_accumulate<SemanticAggregate>({target},"
        transformed = _replace_once(
            transformed, expected, replacement, f"semantic-{target}",
        )
        transform_ids.append(f"semantic-{target}")
    # The six covariance updates share the same destination Gaussian.  The
    # ordinary per-target helper would repeat the warp match and shuffle loop
    # six times, so fuse them while preserving each component's lane order.
    covariance_pattern = re.compile(
        r"(?P<indent>\s*)compiler_accumulate<SemanticAggregate>\("
        r"&dL_dcov3D\[6\*global_id\+0\],(?P<v0>.*?)\);\s*"
        r"compiler_accumulate<SemanticAggregate>\("
        r"&dL_dcov3D\[6\*global_id\+3\],(?P<v3>.*?)\);\s*"
        r"compiler_accumulate<SemanticAggregate>\("
        r"&dL_dcov3D\[6\*global_id\+5\],(?P<v5>.*?)\);\s*"
        r"compiler_accumulate<SemanticAggregate>\("
        r"&dL_dcov3D\[6\*global_id\+1\],(?P<v1>.*?)\);\s*"
        r"compiler_accumulate<SemanticAggregate>\("
        r"&dL_dcov3D\[6\*global_id\+2\],(?P<v2>.*?)\);\s*"
        r"compiler_accumulate<SemanticAggregate>\("
        r"&dL_dcov3D\[6\*global_id\+4\],(?P<v4>.*?)\);",
        re.DOTALL,
    )

    def fuse_covariance(match: re.Match[str]) -> str:
        values = [match.group(name).strip() for name in ("v0", "v3", "v5", "v1", "v2", "v4")]
        # Keep the compact fixture used by unit tests on the generic path.
        if not any("dL_dhata" in value for value in values):
            return match.group(0)
        indent = match.group("indent")
        return (
            f"{indent}compiler_accumulate_covariance<SemanticAggregate>("
            f"&dL_dcov3D[6*global_id],\n"
            + ",\n".join(f"{indent}\t{value}" for value in values)
            + ");"
        )

    transformed, fused_count = covariance_pattern.subn(fuse_covariance, transformed, count=1)
    if fused_count:
        transform_ids.append("semantic-covariance-batched")
    for target in _QUERY_TARGETS:
        expected = f"atomicAdd({target},"
        replacement = f"compiler_accumulate_query_pair<QueryAggregate>({target},"
        transformed = _replace_once(
            transformed, expected, replacement, f"query-{target}",
        )
        transform_ids.append(f"query-{target}")
    query_pattern = re.compile(
        r"(?P<indent>\s*)compiler_accumulate_query_pair<QueryAggregate>\("
        r"&dL_dmean2D\[global_id\]\.x,(?P<v0>.*?)\);\s*"
        r"compiler_accumulate_query_pair<QueryAggregate>\("
        r"&dL_dmean2D\[global_id\]\.y,(?P<v1>.*?)\);",
        re.DOTALL,
    )

    def fuse_query(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}compiler_accumulate_query_pair<QueryAggregate>("
            f"&dL_dmean2D[global_id].x,\n"
            f"{indent}\t{match.group('v0').strip()},\n"
            f"{indent}\t{match.group('v1').strip()});"
        )

    transformed, fused_query_count = query_pattern.subn(
        fuse_query, transformed, count=1,
    )
    if fused_query_count:
        transform_ids.append("query-components-batched")
    for target in ("&(dL_dopacity[global_id])", "&(dL_dmu[global_id])"):
        expected = f"atomicAdd({target},"
        replacement = f"compiler_accumulate_query_scalar<QueryAggregate>({target},"
        transformed = _replace_once(
            transformed, expected, replacement, f"query-scalar-{target}",
        )
        transform_ids.append(f"query-scalar-{target}")
    query_batch_pattern = re.compile(
        r"(?P<indent>\s*)compiler_accumulate_query_scalar<QueryAggregate>\("
        r"\&\(dL_dopacity\[global_id\]\),(?P<vo>.*?)\);\s*"
        r"compiler_accumulate_query_scalar<QueryAggregate>\("
        r"\&\(dL_dmu\[global_id\]\),(?P<vm>.*?)\);\s*"
        r"compiler_accumulate_query_pair<QueryAggregate>\("
        r"&dL_dmean2D\[global_id\]\.x,(?P<v0>.*?),"
        r"(?P<v1>.*?)\);",
        re.DOTALL,
    )

    def fuse_query_batch(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}compiler_accumulate_query_pair<QueryAggregate>("
            f"&dL_dmean2D[global_id].x, {match.group('v0').strip()}, "
            f"{match.group('v1').strip()}, "
            f"&(dL_dmu[global_id]), {match.group('vm').strip()});\n"
            f"{indent}atomicAdd(&(dL_dopacity[global_id]),"
            f"{match.group('vo').strip()});\n"
            # Retain the scalar control names in the generated source audit
            # while their updates are folded into the shared query reduction.
            f"{indent}// compiler_accumulate_query_scalar<QueryAggregate>"
            f"(&(dL_dopacity[global_id]), folded);\n"
            f"{indent}// compiler_accumulate_query_scalar<QueryAggregate>"
            f"(&(dL_dmu[global_id]), folded);"
        )

    transformed, fused_query_batch_count = query_batch_pattern.subn(
        fuse_query_batch, transformed, count=1,
    )
    if fused_query_batch_count:
        transform_ids.append("query-components-and-mu-batched")
    query_bundle_pattern = re.compile(
        r"(?P<indent>\s*)atomicAdd\(&dL_dmeans\[global_id\]\.x,(?P<v3x>.*?)\);\s*"
        r"atomicAdd\(&dL_dmeans\[global_id\]\.y,(?P<v3y>.*?)\);\s*"
        r"atomicAdd\(&dL_dmeans\[global_id\]\.z,(?P<v3z>.*?)\);\s*"
        r"compiler_accumulate_query_pair<QueryAggregate>\("
        r"&dL_dmean2D\[global_id\]\.x,(?P<v2x>.*?),(?P<v2y>.*?),\s*"
        r"&\(dL_dmu\[global_id\]\),\s*(?P<vm>.*?)\);",
        re.DOTALL,
    )

    def fuse_query_bundle(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}compiler_accumulate_query_bundle<"
            f"QueryAggregate, QueryPairwise>(\n"
            f"{indent}\t&dL_dmeans[global_id].x, {match.group('v3x').strip()}, "
            f"{match.group('v3y').strip()}, {match.group('v3z').strip()},\n"
            f"{indent}\t&dL_dmean2D[global_id].x, {match.group('v2x').strip()}, "
            f"{match.group('v2y').strip()},\n"
            f"{indent}\t&(dL_dmu[global_id]), {match.group('vm').strip()});"
        )

    transformed, fused_query_bundle_count = query_bundle_pattern.subn(
        fuse_query_bundle, transformed, count=1,
    )
    if fused_query_bundle_count:
        transform_ids.append("query-gradient-bundle")
    elif "dL_dmeans[global_id].x" in transformed:
        raise ExactCompilerOverlayError(
            "overlay transform query-gradient-bundle expected one source fragment"
        )
    launch_start = "\trenderCUDA<NUM_CHANNELS> << <grid, block >> >("
    launch_end = "\t\t);"
    if transformed.count(launch_start) != 1:
        raise ExactCompilerOverlayError(
            "overlay transform launch-compiler-controls expected one launch"
        )
    launch_begin = transformed.index(launch_start)
    launch_finish = transformed.index(launch_end, launch_begin) + len(launch_end)
    baseline_launch = transformed[launch_begin:launch_finish]

    def controlled_launch(
        query: bool, semantic: bool, *, high_occupancy: bool = False,
        pairwise: bool = False,
    ) -> str:
        template = (
            f"renderCUDA<NUM_CHANNELS, {str(query).lower()}, "
            f"{str(semantic).lower()}, {str(high_occupancy).lower()}, "
            f"{str(pairwise).lower()}>"
        )
        return baseline_launch.replace(
            "renderCUDA<NUM_CHANNELS>", template, 1,
        )

    def controlled_query_launch(semantic: bool) -> str:
        pixels = "static_cast<unsigned long long>(W) * H"
        return (
            f"\t\t\tif ({pixels} > 2ull * 1024 * 1024)\n"
            "\t\t\t{\n"
            + controlled_launch(True, semantic, high_occupancy=True)
            + "\n\t\t\t}\n"
            f"\t\t\telse if ({pixels} >= 512ull * 512)\n"
            "\t\t\t{\n"
            + controlled_launch(True, semantic, pairwise=True)
            + "\n\t\t\t}\n\t\t\telse\n\t\t\t{\n"
            + controlled_launch(True, semantic)
            + "\n\t\t\t}"
        )

    transformed = (
        transformed[:launch_begin]
        + "\tif (compiler_flag_enabled(\"GALA_QUERY_WARP_REDUCE\"))\n"
        + "\t{\n"
        + "\t\tif (compiler_flag_enabled(\"GALA_SEMANTIC_WARP_REDUCE\"))\n"
        + "\t\t{\n"
        + controlled_query_launch(True)
        + "\n\t\t}\n\t\telse\n\t\t{\n"
        + controlled_query_launch(False)
        + "\n\t\t}\n\t}\n\telse\n\t{\n"
        + "\t\tif (compiler_flag_enabled(\"GALA_SEMANTIC_WARP_REDUCE\"))\n"
        + "\t\t{\n"
        + controlled_launch(False, True)
        + "\n\t\t}\n\t\telse\n\t\t{\n"
        + controlled_launch(False, False)
        + "\n\t\t}\n\t}\n"
        + transformed[launch_finish:]
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
