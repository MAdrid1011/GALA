#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <stdexcept>
#include <torch/extension.h>

namespace {

constexpr int kRasterBlockX = 16;
constexpr int kRasterBlockY = 16;
constexpr int kRasterQueriesPerCandidate = kRasterBlockX * kRasterBlockY;
constexpr int kRasterMaskWords = kRasterQueriesPerCandidate / 32;
constexpr int kVoxelBlockX = 8;
constexpr int kVoxelBlockY = 8;
constexpr int kVoxelBlockZ = 8;
constexpr int kVoxelQueriesPerCandidate = kVoxelBlockX * kVoxelBlockY * kVoxelBlockZ;
constexpr int kVoxelMaskWords = kVoxelQueriesPerCandidate / 32;
constexpr int64_t kCandidateRecord = 0;
constexpr int64_t kRelationRecord = 1;

__global__ void count_mask_bits_kernel(
    const std::uint32_t* masks, int words, int64_t candidate_count,
    int64_t* counts) {
    auto candidate = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) {
        return;
    }
    std::uint32_t count = 0;
    for (int word = 0; word < words; ++word) {
        count += __popc(masks[candidate * words + word]);
    }
    counts[candidate] = static_cast<int64_t>(count);
}

__global__ void write_trace_records_kernel(
    const std::uint32_t* point_list, const std::uint64_t* point_keys,
    const std::uint32_t* masks, const int64_t* relation_offsets,
    int words, int64_t candidate_count, int64_t* output) {
    auto candidate = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) {
        return;
    }
    auto candidate_row = output + candidate * 4;
    candidate_row[0] = kCandidateRecord;
    candidate_row[1] = candidate;
    candidate_row[2] = static_cast<int64_t>(point_list[candidate]);
    candidate_row[3] = static_cast<int64_t>(point_keys[candidate]);
    auto relation_row = candidate == 0 ? 0 : relation_offsets[candidate - 1];
    for (int word = 0; word < words; ++word) {
        auto bits = masks[candidate * words + word];
        while (bits != 0) {
            auto bit = __ffs(bits) - 1;
            auto row = output + (candidate_count + relation_row) * 4;
            row[0] = kRelationRecord;
            row[1] = candidate;
            row[2] = static_cast<int64_t>(word * 32 + bit);
            row[3] = 0;
            ++relation_row;
            bits &= bits - 1;
        }
    }
}

std::size_t align128(const torch::Tensor& buffer, std::size_t offset) {
    auto base = reinterpret_cast<std::uintptr_t>(buffer.data_ptr<std::uint8_t>());
    auto address = base + offset;
    return offset + ((address + 127u) & ~std::uintptr_t(127u)) - address;
}

template <typename T>
T* obtain(const torch::Tensor& buffer, std::size_t& offset, std::size_t count) {
    offset = align128(buffer, offset);
    auto base = reinterpret_cast<std::uintptr_t>(buffer.data_ptr<std::uint8_t>());
    auto pointer = reinterpret_cast<T*>(base + offset);
    offset += count * sizeof(T);
    if (offset > static_cast<std::size_t>(buffer.numel())) {
        throw std::invalid_argument("trace buffer is smaller than its declared layout");
    }
    return pointer;
}

void validate(const torch::Tensor& buffer, int64_t gaussian_count,
              int64_t candidate_count) {
    if (!buffer.defined() || !buffer.is_contiguous()
        || buffer.scalar_type() != torch::kUInt8
        || buffer.device().type() != torch::kCUDA) {
        throw std::invalid_argument("trace buffer must be a contiguous CUDA uint8 tensor");
    }
    if (gaussian_count < 0 || candidate_count < 0) {
        throw std::invalid_argument("trace buffer counts must be non-negative");
    }
}

torch::Tensor compact_trace_records(
    const torch::Tensor& binning_buffer, int64_t total_candidate_count,
    int64_t candidate_start, int64_t candidate_count,
    const torch::Tensor& masks, bool voxel) {
    if (candidate_start < 0 || candidate_count < 0
        || candidate_start > total_candidate_count
        || candidate_count > total_candidate_count - candidate_start) {
        throw std::invalid_argument("trace candidate range is invalid");
    }
    std::size_t binning_offset = 0;
    auto point_list_pointer = obtain<std::uint32_t>(
        binning_buffer, binning_offset, total_candidate_count);
    obtain<std::uint32_t>(binning_buffer, binning_offset, total_candidate_count);
    auto point_keys_pointer = obtain<std::uint64_t>(
        binning_buffer, binning_offset, total_candidate_count);
    if (voxel) {
        obtain<std::uint64_t>(binning_buffer, binning_offset, total_candidate_count);
    }
    point_list_pointer += candidate_start;
    point_keys_pointer += candidate_start;

    auto long_options = binning_buffer.options().dtype(torch::kInt64);
    auto count_options = masks.options().dtype(torch::kInt64);
    auto mask_words = static_cast<int>(masks.size(1));
    auto counts = torch::empty({candidate_count}, count_options);
    constexpr int threads = 256;
    if (candidate_count > 0) {
        count_mask_bits_kernel<<<(candidate_count + threads - 1) / threads, threads,
                                 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const std::uint32_t*>(masks.data_ptr<int>()),
            mask_words, candidate_count, counts.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto offsets = torch::cumsum(counts, 0);
    int64_t relation_count = 0;
    if (candidate_count > 0) {
        relation_count = offsets.index({candidate_count - 1}).item<int64_t>();
    }
    auto output = torch::empty({candidate_count + relation_count, 4}, long_options);
    if (candidate_count > 0) {
        write_trace_records_kernel<<<(candidate_count + threads - 1) / threads, threads,
                                     0, at::cuda::getCurrentCUDAStream()>>>(
            point_list_pointer, point_keys_pointer,
            reinterpret_cast<const std::uint32_t*>(masks.data_ptr<int>()),
            offsets.data_ptr<int64_t>(), mask_words, candidate_count,
            output.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output.contiguous();
}

__global__ void raster_masks_kernel(
    int64_t candidate_count, int image_height, int image_width,
    const std::uint32_t* point_list, const std::uint64_t* point_keys,
    const float2* means2d, const float4* conic_opacity, const float* mus,
    std::uint32_t* masks) {
    auto linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto total = candidate_count * kRasterQueriesPerCandidate;
    if (linear >= total) {
        return;
    }
    auto candidate = linear / kRasterQueriesPerCandidate;
    auto local = static_cast<int>(linear % kRasterQueriesPerCandidate);
    auto tile = static_cast<std::uint32_t>(point_keys[candidate] >> 32);
    auto horizontal_blocks = (image_width + kRasterBlockX - 1) / kRasterBlockX;
    auto pixel_x = static_cast<int>(tile % horizontal_blocks) * kRasterBlockX
                   + local % kRasterBlockX;
    auto pixel_y = static_cast<int>(tile / horizontal_blocks) * kRasterBlockY
                   + local / kRasterBlockX;
    if (pixel_x >= image_width || pixel_y >= image_height) {
        return;
    }
    auto gaussian = point_list[candidate];
    float2 xy = means2d[gaussian];
    float2 d = {xy.x - static_cast<float>(pixel_x),
                xy.y - static_cast<float>(pixel_y)};
    float4 con_o = conic_opacity[gaussian];
    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
                  - con_o.y * d.x * d.y;
    if (power > 0.0f) {
        return;
    }
    float alpha = con_o.w * mus[gaussian] * expf(power);
    if (alpha < 0.00001f) {
        return;
    }
    atomicOr(masks + candidate * kRasterMaskWords + local / 32,
             std::uint32_t(1u) << (local % 32));
}

__global__ void voxel_masks_kernel(
    int64_t candidate_count, int voxel_x, int voxel_y, int voxel_z,
    const std::uint32_t* point_list, const std::uint64_t* point_keys,
    const float3* means3d, const float* conic_opacity, std::uint32_t* masks) {
    auto linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto total = candidate_count * kVoxelQueriesPerCandidate;
    if (linear >= total) {
        return;
    }
    auto candidate = linear / kVoxelQueriesPerCandidate;
    auto local = static_cast<int>(linear % kVoxelQueriesPerCandidate);
    auto tile = static_cast<std::uint32_t>(point_keys[candidate] >> 32);
    auto blocks_x = (voxel_x + kVoxelBlockX - 1) / kVoxelBlockX;
    auto blocks_y = (voxel_y + kVoxelBlockY - 1) / kVoxelBlockY;
    auto tile_x = static_cast<int>(tile % blocks_x);
    auto tile_y = static_cast<int>((tile / blocks_x) % blocks_y);
    auto tile_z = static_cast<int>(tile / (blocks_x * blocks_y));
    auto local_x = local % kVoxelBlockX;
    auto local_y = (local / kVoxelBlockX) % kVoxelBlockY;
    auto local_z = local / (kVoxelBlockX * kVoxelBlockY);
    auto query_x = tile_x * kVoxelBlockX + local_x;
    auto query_y = tile_y * kVoxelBlockY + local_y;
    auto query_z = tile_z * kVoxelBlockZ + local_z;
    if (query_x >= voxel_x || query_y >= voxel_y || query_z >= voxel_z) {
        return;
    }
    auto gaussian = point_list[candidate];
    float3 xyz = means3d[gaussian];
    float3 d = {xyz.x - (static_cast<float>(query_x) + 0.5f),
                xyz.y - (static_cast<float>(query_y) + 0.5f),
                xyz.z - (static_cast<float>(query_z) + 0.5f)};
    auto conic = conic_opacity + gaussian * 7;
    float power = -0.5f * (conic[0] * d.x * d.x + conic[3] * d.y * d.y
                           + conic[5] * d.z * d.z)
                  - conic[1] * d.x * d.y - conic[2] * d.x * d.z
                  - conic[4] * d.y * d.z;
    if (power > 0.0f) {
        return;
    }
    float alpha = conic[6] * expf(power);
    if (alpha < 0.000001f) {
        return;
    }
    atomicOr(masks + candidate * kVoxelMaskWords + local / 32,
             std::uint32_t(1u) << (local % 32));
}

}  // namespace

torch::Tensor raster_valid_masks_range_cuda(const torch::Tensor& geometry_buffer,
                                            const torch::Tensor& binning_buffer,
                                            int64_t gaussian_count,
                                            int64_t total_candidate_count,
                                            int64_t candidate_start,
                                            int64_t candidate_count,
                                            int64_t image_height,
                                            int64_t image_width) {
    validate(geometry_buffer, gaussian_count, candidate_count);
    validate(binning_buffer, gaussian_count, total_candidate_count);
    if (candidate_start < 0 || candidate_count < 0
        || candidate_start > total_candidate_count
        || candidate_count > total_candidate_count - candidate_start) {
        throw std::invalid_argument("raster candidate range is invalid");
    }
    if (image_height <= 0 || image_width <= 0) {
        throw std::invalid_argument("raster dimensions must be positive");
    }
    std::size_t geometry_offset = 0;
    obtain<float>(geometry_buffer, geometry_offset, gaussian_count);
    obtain<int>(geometry_buffer, geometry_offset, gaussian_count);
    auto means2d = obtain<float2>(geometry_buffer, geometry_offset, gaussian_count);
    obtain<float>(geometry_buffer, geometry_offset, gaussian_count * 6);
    auto conic = obtain<float4>(geometry_buffer, geometry_offset, gaussian_count);
    auto mus = obtain<float>(geometry_buffer, geometry_offset, gaussian_count);
    std::size_t binning_offset = 0;
    auto point_list = obtain<std::uint32_t>(
        binning_buffer, binning_offset, total_candidate_count) + candidate_start;
    obtain<std::uint32_t>(binning_buffer, binning_offset, total_candidate_count);
    auto point_keys = obtain<std::uint64_t>(
        binning_buffer, binning_offset, total_candidate_count) + candidate_start;
    auto masks = torch::zeros({candidate_count, kRasterMaskWords},
                              geometry_buffer.options().dtype(torch::kInt32));
    auto total = candidate_count * kRasterQueriesPerCandidate;
    constexpr int threads = 256;
    if (total > 0) {
        raster_masks_kernel<<<(total + threads - 1) / threads, threads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
            candidate_count, static_cast<int>(image_height), static_cast<int>(image_width),
            point_list, point_keys, means2d, conic, mus,
            reinterpret_cast<std::uint32_t*>(masks.data_ptr<int>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return masks;
}

torch::Tensor voxel_valid_masks_range_cuda(const torch::Tensor& geometry_buffer,
                                           const torch::Tensor& binning_buffer,
                                           int64_t gaussian_count,
                                           int64_t total_candidate_count,
                                           int64_t candidate_start,
                                           int64_t candidate_count,
                                           int64_t voxel_x, int64_t voxel_y,
                                           int64_t voxel_z) {
    validate(geometry_buffer, gaussian_count, candidate_count);
    validate(binning_buffer, gaussian_count, total_candidate_count);
    if (candidate_start < 0 || candidate_count < 0
        || candidate_start > total_candidate_count
        || candidate_count > total_candidate_count - candidate_start) {
        throw std::invalid_argument("voxel candidate range is invalid");
    }
    if (voxel_x <= 0 || voxel_y <= 0 || voxel_z <= 0) {
        throw std::invalid_argument("voxel dimensions must be positive");
    }
    std::size_t geometry_offset = 0;
    obtain<float>(geometry_buffer, geometry_offset, gaussian_count);
    obtain<int>(geometry_buffer, geometry_offset, gaussian_count);
    obtain<int>(geometry_buffer, geometry_offset, gaussian_count);
    obtain<int>(geometry_buffer, geometry_offset, gaussian_count);
    auto means3d = obtain<float3>(geometry_buffer, geometry_offset, gaussian_count);
    obtain<float>(geometry_buffer, geometry_offset, gaussian_count * 6);
    auto conic = obtain<float>(geometry_buffer, geometry_offset, gaussian_count * 7);
    std::size_t binning_offset = 0;
    auto point_list = obtain<std::uint32_t>(
        binning_buffer, binning_offset, total_candidate_count) + candidate_start;
    obtain<std::uint32_t>(binning_buffer, binning_offset, total_candidate_count);
    auto point_keys = obtain<std::uint64_t>(
        binning_buffer, binning_offset, total_candidate_count) + candidate_start;
    auto masks = torch::zeros({candidate_count, kVoxelMaskWords},
                              geometry_buffer.options().dtype(torch::kInt32));
    auto total = candidate_count * kVoxelQueriesPerCandidate;
    constexpr int threads = 256;
    if (total > 0) {
        voxel_masks_kernel<<<(total + threads - 1) / threads, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
            candidate_count, static_cast<int>(voxel_x), static_cast<int>(voxel_y),
            static_cast<int>(voxel_z), point_list, point_keys, means3d, conic,
            reinterpret_cast<std::uint32_t*>(masks.data_ptr<int>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return masks;
}


torch::Tensor raster_trace_records_cuda(const torch::Tensor& geometry_buffer,
                                        const torch::Tensor& binning_buffer,
                                        int64_t gaussian_count, int64_t candidate_count,
                                        int64_t image_height, int64_t image_width) {
    auto masks = raster_valid_masks_range_cuda(
        geometry_buffer, binning_buffer, gaussian_count, candidate_count, 0,
        candidate_count, image_height, image_width);
    return compact_trace_records(
        binning_buffer, candidate_count, 0, candidate_count, masks, false);
}

torch::Tensor raster_valid_masks_cuda(const torch::Tensor& geometry_buffer,
                                      const torch::Tensor& binning_buffer,
                                      int64_t gaussian_count, int64_t candidate_count,
                                      int64_t image_height, int64_t image_width) {
    return raster_valid_masks_range_cuda(
        geometry_buffer, binning_buffer, gaussian_count, candidate_count, 0,
        candidate_count, image_height, image_width);
}

torch::Tensor raster_trace_records_chunk_cuda(
    const torch::Tensor& geometry_buffer, const torch::Tensor& binning_buffer,
    int64_t gaussian_count, int64_t total_candidate_count,
    int64_t candidate_start, int64_t candidate_count,
    int64_t image_height, int64_t image_width) {
    auto masks = raster_valid_masks_range_cuda(
        geometry_buffer, binning_buffer, gaussian_count, total_candidate_count,
        candidate_start, candidate_count, image_height, image_width);
    return compact_trace_records(
        binning_buffer, total_candidate_count, candidate_start, candidate_count,
        masks, false);
}


torch::Tensor voxel_trace_records_cuda(const torch::Tensor& geometry_buffer,
                                       const torch::Tensor& binning_buffer,
                                       int64_t gaussian_count, int64_t candidate_count,
                                       int64_t voxel_x, int64_t voxel_y, int64_t voxel_z) {
    auto masks = voxel_valid_masks_range_cuda(
        geometry_buffer, binning_buffer, gaussian_count, candidate_count, 0,
        candidate_count, voxel_x, voxel_y, voxel_z);
    return compact_trace_records(
        binning_buffer, candidate_count, 0, candidate_count, masks, true);
}

torch::Tensor voxel_valid_masks_cuda(const torch::Tensor& geometry_buffer,
                                     const torch::Tensor& binning_buffer,
                                     int64_t gaussian_count, int64_t candidate_count,
                                     int64_t voxel_x, int64_t voxel_y, int64_t voxel_z) {
    return voxel_valid_masks_range_cuda(
        geometry_buffer, binning_buffer, gaussian_count, candidate_count, 0,
        candidate_count, voxel_x, voxel_y, voxel_z);
}

torch::Tensor voxel_trace_records_chunk_cuda(
    const torch::Tensor& geometry_buffer, const torch::Tensor& binning_buffer,
    int64_t gaussian_count, int64_t total_candidate_count,
    int64_t candidate_start, int64_t candidate_count,
    int64_t voxel_x, int64_t voxel_y, int64_t voxel_z) {
    auto masks = voxel_valid_masks_range_cuda(
        geometry_buffer, binning_buffer, gaussian_count, total_candidate_count,
        candidate_start, candidate_count, voxel_x, voxel_y, voxel_z);
    return compact_trace_records(
        binning_buffer, total_candidate_count, candidate_start, candidate_count,
        masks, true);
}
