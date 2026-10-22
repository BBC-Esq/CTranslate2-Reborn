#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/gather_prefix.h"

#include "ctranslate2/primitives.h"
#include "mps/kernels.h"
#include "mps/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      if (axis != batch_dims)
        throw std::invalid_argument("Gather only supports indexing the first non batch dimension");

      const dim_t copy_size = data.stride(axis);
      const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
      const dim_t batch_size = data.size() / batch_stride;
      const dim_t num_indices = input.size();
      const dim_t num_indices_per_batch = num_indices / batch_size;
      mps::gather(DataTypeToEnum<T>::value,
                  data.data<T>(),
                  input.data<int32_t>(),
                  output.data<T>(),
                  copy_size,
                  batch_stride,
                  num_indices,
                  num_indices_per_batch);
    }

#define DECLARE_GATHER(T) \
    template void Gather::compute<Device::MPS, T>( \
      const StorageView&, const StorageView&, dim_t, dim_t, StorageView&) const;
    DECLARE_ALL_TYPES(DECLARE_GATHER)
#undef DECLARE_GATHER

    template <Device D, typename T>
    void GatherPrefix::compute(const StorageView& data,
                               const StorageView& indices,
                               const dim_t axis,
                               StorageView& output) const {
      // Each row of the data is num_slices slices of slice_size values, and the first
      // prefix_size values of each slice are copied. The indices are read on the host.
      const StorageView host_indices = mps::to_cpu(indices);
      const auto* index_data = host_indices.data<int32_t>();
      const dim_t row_size = data.stride(0);
      const dim_t slice_size = data.dim(axis) * data.stride(axis);
      const dim_t prefix_size = _length * data.stride(axis);
      const dim_t num_slices = row_size / slice_size;
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      for (dim_t row = 0; row < host_indices.size(); ++row) {
        for (dim_t slice = 0; slice < num_slices; ++slice)
          primitives<Device::MPS>::copy(src + index_data[row] * row_size + slice * slice_size,
                                        dst + row * row_size + slice * slice_size,
                                        prefix_size);
      }
    }

#define DECLARE_GATHER_PREFIX(T) \
    template void GatherPrefix::compute<Device::MPS, T>( \
      const StorageView&, const StorageView&, dim_t, StorageView&) const;
    DECLARE_ALL_TYPES(DECLARE_GATHER_PREFIX)
#undef DECLARE_GATHER_PREFIX

  }
}
