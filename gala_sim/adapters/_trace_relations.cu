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

torch::Tensor raster_valid_masks_cuda(const torch::Tensor& geometry_buffer,
                                      const torch::Tensor& binning_buffer,
                                      int64_t gaussian_count, int64_t candidate_count,
                                      int64_t image_height, int64_t image_width) {
    validate(geometry_buffer, gaussian_count, candidate_count);
    validate(binning_buffer, gaussian_count, candidate_count);
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
    auto point_list = obtain<std::uint32_t>(binning_buffer, binning_offset, candidate_count);
    obtain<std::uint32_t>(binning_buffer, binning_offset, candidate_count);
    auto point_keys = obtain<std::uint64_t>(binning_buffer, binning_offset, candidate_count);
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

torch::Tensor voxel_valid_masks_cuda(const torch::Tensor& geometry_buffer,
                                     const torch::Tensor& binning_buffer,
                                     int64_t gaussian_count, int64_t candidate_count,
                                     int64_t voxel_x, int64_t voxel_y, int64_t voxel_z) {
    validate(geometry_buffer, gaussian_count, candidate_count);
    validate(binning_buffer, gaussian_count, candidate_count);
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
    auto point_list = obtain<std::uint32_t>(binning_buffer, binning_offset, candidate_count);
    obtain<std::uint32_t>(binning_buffer, binning_offset, candidate_count);
    auto point_keys = obtain<std::uint64_t>(binning_buffer, binning_offset, candidate_count);
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
