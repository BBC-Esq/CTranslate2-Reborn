#include "ctranslate2/ops/gather_prefix.h"

#include <algorithm>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Gathers increasing indices in place on CPU, one row after the other: each row is read from
    // a row at or after it, before that row is written.
    template <typename T>
    static void gather_prefix_inplace(StorageView& data,
                                      const StorageView& indices,
                                      const dim_t axis,
                                      const dim_t length) {
      const auto* index_data = indices.data<int32_t>();
      const dim_t row_size = data.stride(0);
      const dim_t slice_size = data.dim(axis) * data.stride(axis);
      const dim_t prefix_size = length * data.stride(axis);
      if (prefix_size == 0)
        return;
      const dim_t num_slices = row_size / slice_size;
      T* rows = data.data<T>();

      for (dim_t i = 0; i < indices.size(); ++i) {
        const dim_t index = index_data[i];
        if (index == i)
          continue;
        const T* src = rows + index * row_size;
        T* dst = rows + i * row_size;
        for (dim_t s = 0; s < num_slices; ++s)
          primitives<Device::CPU>::copy(src + s * slice_size, dst + s * slice_size, prefix_size);
      }
    }

    GatherPrefix::GatherPrefix(dim_t axis, dim_t length)
      : _axis(axis)
      , _length(length) {
    }

    dim_t GatherPrefix::check_arguments(const StorageView& data, const StorageView& indices) const {
      const dim_t axis = _axis < 0 ? data.rank() + _axis : _axis;
      if (axis < 1 || axis >= data.rank())
        throw std::invalid_argument("GatherPrefix: the axis must be an inner dimension, but got "
                                    + std::to_string(_axis) + " for a tensor of rank "
                                    + std::to_string(data.rank()));
      if (_length < 0 || _length > data.dim(axis))
        throw std::invalid_argument("GatherPrefix: cannot copy " + std::to_string(_length)
                                    + " positions of a dimension of size "
                                    + std::to_string(data.dim(axis)));
      if (indices.dtype() != DataType::INT32 || indices.rank() != 1)
        throw std::invalid_argument("GatherPrefix: the indices must be a 1D int32 tensor");
      if (indices.device() != data.device())
        throw std::invalid_argument("GatherPrefix: the data and indices devices differ");
      return axis;
    }

    void GatherPrefix::operator()(StorageView& data, const StorageView& indices) const {
      PROFILE("GatherPrefix");
      const dim_t axis = check_arguments(data, indices);

      if (data.device() == Device::CPU && indices.size() <= data.dim(0)) {
        const auto* index_begin = indices.data<int32_t>();
        const auto* index_end = index_begin + indices.size();
        if (std::adjacent_find(index_begin, index_end, std::greater_equal<int32_t>()) == index_end) {
          TYPE_DISPATCH(data.dtype(), (gather_prefix_inplace<T>(data, indices, axis, _length)));
          data.resize(0, indices.size());
          return;
        }
      }

      StorageView clone(std::move(data));
      operator()(clone, indices, data);
    }

    void GatherPrefix::operator()(const StorageView& data,
                                  const StorageView& indices,
                                  StorageView& output) const {
      PROFILE("GatherPrefix");
      const dim_t axis = check_arguments(data, indices);
      if (&data == &output)
        throw std::invalid_argument("GatherPrefix: use the in-place operator to gather in place");

      Shape output_shape(data.shape());
      output_shape[0] = indices.size();
      output.resize(std::move(output_shape));

      if (output.empty() || _length == 0)
        return;

      DEVICE_AND_TYPE_DISPATCH(data.device(), data.dtype(),
                               (compute<D, T>(data, indices, axis, output)));
    }

  }
}
