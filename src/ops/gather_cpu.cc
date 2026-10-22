#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/gather_prefix.h"

#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      const auto* indices = input.data<int32_t>();
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      if (axis == batch_dims) {
        const dim_t copy_size = data.stride(axis);
        const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
        const dim_t batch_size = data.size() / batch_stride;
        const dim_t num_indices = input.size();
        const dim_t num_indices_per_batch = num_indices / batch_size;

        const dim_t grain_size = cpu::get_minimum_batch_copies_per_thread<T>(copy_size);
        cpu::parallel_for(0, num_indices, grain_size, [&](dim_t begin, dim_t end) {
          for (dim_t i = begin; i < end; ++i) {
            const dim_t batch_index = i / num_indices_per_batch;
            const dim_t read_index = indices[i];
            primitives<Device::CPU>::copy(src + batch_index * batch_stride + read_index * copy_size,
                                          dst + i * copy_size,
                                          copy_size);
          }
        });

      } else {
        throw std::invalid_argument("Gather only supports indexing the first non batch dimension");
      }
    }

    template <Device D, typename T>
    void GatherPrefix::compute(const StorageView& data,
                               const StorageView& indices,
                               const dim_t axis,
                               StorageView& output) const {
      // Each row of the data is num_slices slices of slice_size values, and the first
      // prefix_size values of each slice are copied.
      const auto* index_data = indices.data<int32_t>();
      const dim_t row_size = data.stride(0);
      const dim_t slice_size = data.dim(axis) * data.stride(axis);
      const dim_t prefix_size = _length * data.stride(axis);
      const dim_t num_slices = row_size / slice_size;
      const dim_t num_copies = indices.size() * num_slices;
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      const dim_t grain_size = cpu::get_minimum_batch_copies_per_thread<T>(prefix_size);
      cpu::parallel_for(0, num_copies, grain_size, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t row = i / num_slices;
          const dim_t slice = i % num_slices;
          primitives<Device::CPU>::copy(src + index_data[row] * row_size + slice * slice_size,
                                        dst + row * row_size + slice * slice_size,
                                        prefix_size);
        }
      });
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::CPU, T>(const StorageView& data,            \
                                    const StorageView& input,           \
                                    const dim_t axis,                   \
                                    const dim_t batch_dims,             \
                                    StorageView& output) const;         \
    template void                                                       \
    GatherPrefix::compute<Device::CPU, T>(const StorageView& data,      \
                                          const StorageView& indices,   \
                                          const dim_t axis,             \
                                          StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
