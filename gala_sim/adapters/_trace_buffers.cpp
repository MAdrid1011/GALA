#include <cstdint>
#include <stdexcept>
#include <torch/extension.h>

namespace {

std::size_t align128(const torch::Tensor& buffer, std::size_t offset) {
    auto base = reinterpret_cast<std::uintptr_t>(buffer.data_ptr<std::uint8_t>());
    auto address = base + offset;
    return offset + ((address + 127u) & ~std::uintptr_t(127u)) - address;
}

std::uintptr_t aligned_pointer(const torch::Tensor& buffer, std::size_t offset) {
    auto base = reinterpret_cast<std::uintptr_t>(buffer.data_ptr<std::uint8_t>());
    return base + offset;
}

torch::Tensor copy_int32(const torch::Tensor& buffer, std::uintptr_t pointer,
                         std::vector<int64_t> shape) {
    auto options = buffer.options().dtype(torch::kInt32);
    auto view = torch::from_blob(reinterpret_cast<void*>(pointer), shape,
                                 [](void*) {}, options);
    return view.clone();
}

torch::Tensor copy_int64(const torch::Tensor& buffer, std::uintptr_t pointer,
                         std::vector<int64_t> shape) {
    auto options = buffer.options().dtype(torch::kInt64);
    auto view = torch::from_blob(reinterpret_cast<void*>(pointer), shape,
                                 [](void*) {}, options);
    return view.clone();
}

void validate_buffer(const torch::Tensor& buffer) {
    if (!buffer.defined() || !buffer.is_contiguous() || buffer.scalar_type() != torch::kUInt8
        || buffer.device().type() != torch::kCUDA) {
        throw std::invalid_argument("trace buffer must be a contiguous CUDA uint8 tensor");
    }
}

}  // namespace

torch::Tensor copy_raster_point_list(const torch::Tensor& buffer, int64_t count) {
    validate_buffer(buffer);
    if (count < 0) {
        throw std::invalid_argument("raster point-list count must be non-negative");
    }
    std::size_t offset = align128(buffer, 0);
    auto needed = offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t);
    if (needed > static_cast<std::size_t>(buffer.numel())) {
        throw std::invalid_argument("raster binning buffer is smaller than its declared layout");
    }
    return copy_int32(buffer, aligned_pointer(buffer, offset), {count});
}

torch::Tensor copy_voxel_point_list(const torch::Tensor& buffer, int64_t count) {
    validate_buffer(buffer);
    if (count < 0) {
        throw std::invalid_argument("voxel point-list count must be non-negative");
    }
    std::size_t offset = align128(buffer, 0);
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t));
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t));
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint64_t));
    auto needed = offset + static_cast<std::size_t>(count) * sizeof(std::uint64_t);
    if (needed > static_cast<std::size_t>(buffer.numel())) {
        throw std::invalid_argument("voxel binning buffer is smaller than its declared layout");
    }
    return copy_int32(buffer, aligned_pointer(buffer, 0), {count});
}

torch::Tensor copy_raster_point_keys(const torch::Tensor& buffer, int64_t count) {
    validate_buffer(buffer);
    if (count < 0) {
        throw std::invalid_argument("raster point-key count must be non-negative");
    }
    std::size_t offset = align128(buffer, 0);
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t));
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t));
    auto needed = offset + static_cast<std::size_t>(count) * sizeof(std::uint64_t);
    if (needed > static_cast<std::size_t>(buffer.numel())) {
        throw std::invalid_argument("raster binning buffer is smaller than its key layout");
    }
    return copy_int64(buffer, aligned_pointer(buffer, offset), {count});
}

torch::Tensor copy_voxel_point_keys(const torch::Tensor& buffer, int64_t count) {
    validate_buffer(buffer);
    if (count < 0) {
        throw std::invalid_argument("voxel point-key count must be non-negative");
    }
    std::size_t offset = align128(buffer, 0);
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t));
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t));
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint64_t));
    auto needed = offset + static_cast<std::size_t>(count) * sizeof(std::uint64_t);
    if (needed > static_cast<std::size_t>(buffer.numel())) {
        throw std::invalid_argument("voxel binning buffer is smaller than its key layout");
    }
    return copy_int64(buffer, aligned_pointer(buffer, offset), {count});
}

torch::Tensor copy_ranges(const torch::Tensor& buffer, int64_t count) {
    validate_buffer(buffer);
    if (count < 0) {
        throw std::invalid_argument("range count must be non-negative");
    }
    std::size_t offset = align128(buffer, 0);
    offset = align128(buffer, offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t));
    auto needed = offset + static_cast<std::size_t>(count) * sizeof(std::uint32_t) * 2u;
    if (needed > static_cast<std::size_t>(buffer.numel())) {
        throw std::invalid_argument("image buffer is smaller than its range layout");
    }
    return copy_int32(buffer, aligned_pointer(buffer, offset), {count, 2});
}

torch::Tensor raster_valid_masks_cuda(const torch::Tensor& geometry_buffer,
                                      const torch::Tensor& binning_buffer,
                                      int64_t gaussian_count, int64_t candidate_count,
                                      int64_t image_height, int64_t image_width);

torch::Tensor voxel_valid_masks_cuda(const torch::Tensor& geometry_buffer,
                                     const torch::Tensor& binning_buffer,
                                     int64_t gaussian_count, int64_t candidate_count,
                                     int64_t voxel_x, int64_t voxel_y, int64_t voxel_z);

torch::Tensor raster_trace_records_cuda(const torch::Tensor& geometry_buffer,
                                        const torch::Tensor& binning_buffer,
                                        int64_t gaussian_count, int64_t candidate_count,
                                        int64_t image_height, int64_t image_width);

torch::Tensor voxel_trace_records_cuda(const torch::Tensor& geometry_buffer,
                                       const torch::Tensor& binning_buffer,
                                       int64_t gaussian_count, int64_t candidate_count,
                                       int64_t voxel_x, int64_t voxel_y, int64_t voxel_z);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("copy_raster_point_list", &copy_raster_point_list);
    module.def("copy_voxel_point_list", &copy_voxel_point_list);
    module.def("copy_raster_point_keys", &copy_raster_point_keys);
    module.def("copy_voxel_point_keys", &copy_voxel_point_keys);
    module.def("copy_ranges", &copy_ranges);
    module.def("raster_valid_masks", &raster_valid_masks_cuda);
    module.def("voxel_valid_masks", &voxel_valid_masks_cuda);
    module.def("raster_trace_records", &raster_trace_records_cuda);
    module.def("voxel_trace_records", &voxel_trace_records_cuda);
}
