#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/gather_prefix.h"

#include <type_traits>

#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "cuda/helpers.h"
#include "cuda/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // Functor mapping output index to data index when axis == batch_dims.
    template <typename T>
    class batch_gather_index_map {
    private:
      const int32_t* _indices;
      const T _num_indices_per_batch;
      const T _batch_stride;
      const T _axis_stride;
    public:
      batch_gather_index_map(const int32_t* indices,
                             const T num_indices_per_batch,
                             const T batch_stride,
                             const T axis_stride)
        : _indices(indices)
        , _num_indices_per_batch(num_indices_per_batch)
        , _batch_stride(batch_stride)
        , _axis_stride(axis_stride)
      {
      }
      __device__
      T operator()(const T i) const {
        const T inner_index = i % _axis_stride;
        const T outer_index = i / _axis_stride;
        const T batch_index = outer_index / _num_indices_per_batch;
        return batch_index * _batch_stride + _indices[outer_index] * _axis_stride + inner_index;
      }
    };

    template <typename T, typename IndexMap>
    void run_gather(const IndexMap& index_map,
                    const T* src,
                    T* dst,
                    const dim_t dst_size) {
      auto gather_ids = thrust::make_transform_iterator(thrust::counting_iterator<cuda::index_t>(0),
                                                        index_map);
      THRUST_CALL(thrust::gather, gather_ids, gather_ids + dst_size, src, dst);
    }

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      const dim_t dst_size = output.size();
      const int32_t* indices = input.data<int32_t>();
      const T* src = data.data<T>();
      T* dst = output.data<T>();

      if (axis == batch_dims) {
        const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
        const dim_t batch_size = data.size() / batch_stride;
        const dim_t num_indices_per_batch = input.size() / batch_size;
        const dim_t gather_size = data.stride(axis);
        const dim_t gather_bytes = gather_size * sizeof (T);

        if (gather_bytes % sizeof (uint4) == 0) {
          const dim_t dst_bytes = dst_size * sizeof (T);
          const dim_t batch_stride_bytes = batch_stride * sizeof (T);
          run_gather(batch_gather_index_map<cuda::index_t>(indices,
                                                           num_indices_per_batch,
                                                           batch_stride_bytes / sizeof (uint4),
                                                           gather_bytes / sizeof (uint4)),
                     reinterpret_cast<const uint4*>(src),
                     reinterpret_cast<uint4*>(dst),
                     dst_bytes / sizeof (uint4));
        } else {
          run_gather(batch_gather_index_map<cuda::index_t>(indices,
                                                           num_indices_per_batch,
                                                           batch_stride,
                                                           gather_size),
                     src, dst, dst_size);
        }

      } else {
        throw std::invalid_argument("Gather only supports indexing the first non batch dimension");
      }
    }

    // Copies the first prefix_size units of each slice of slice_size units. The output has
    // num_slices slices for each index, read from the slices of the data row at that index.
    template <typename U>
    __global__ void gather_prefix_kernel(const U* src,
                                         U* dst,
                                         const int32_t* indices,
                                         const cuda::index_t num_slices,
                                         const cuda::index_t slice_size,
                                         const cuda::index_t prefix_size,
                                         const cuda::index_t size) {
      for (cuda::index_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        const cuda::index_t slice = i / prefix_size;
        const cuda::index_t offset = i - slice * prefix_size;
        const cuda::index_t row = slice / num_slices;
        const cuda::index_t row_slice = slice - row * num_slices;
        const cuda::index_t src_slice = cuda::index_t(indices[row]) * num_slices + row_slice;
        dst[slice * slice_size + offset] = src[src_slice * slice_size + offset];
      }
    }

    template <typename U>
    static void run_gather_prefix(const void* src,
                                  void* dst,
                                  const int32_t* indices,
                                  const dim_t num_indices,
                                  const dim_t num_slices,
                                  const dim_t slice_bytes,
                                  const dim_t prefix_bytes) {
      const dim_t slice_size = slice_bytes / sizeof (U);
      const dim_t prefix_size = prefix_bytes / sizeof (U);
      const dim_t size = num_indices * num_slices * prefix_size;
      const dim_t threads = 256;
      const dim_t blocks = std::min(ceil_divide(size, threads), cuda::max_blocks);
      gather_prefix_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
        static_cast<const U*>(src),
        static_cast<U*>(dst),
        indices,
        num_slices,
        slice_size,
        prefix_size,
        size);
    }

    template <Device D, typename T>
    void GatherPrefix::compute(const StorageView& data,
                               const StorageView& indices,
                               const dim_t axis,
                               StorageView& output) const {
      const dim_t row_size = data.stride(0);
      const dim_t slice_size = data.dim(axis) * data.stride(axis);
      const dim_t num_slices = row_size / slice_size;
      const dim_t slice_bytes = slice_size * sizeof (T);
      const dim_t prefix_bytes = _length * data.stride(axis) * sizeof (T);
      const void* src = data.data<T>();
      void* dst = output.data<T>();

      const auto aligned = [](const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) % sizeof (uint4) == 0;
      };

      using Raw = std::conditional_t<sizeof (T) == 1, uint8_t,
                  std::conditional_t<sizeof (T) == 2, uint16_t,
                  std::conditional_t<sizeof (T) == 4, uint32_t, uint64_t>>>;

      if (slice_bytes % sizeof (uint4) == 0
          && prefix_bytes % sizeof (uint4) == 0
          && aligned(src)
          && aligned(dst))
        run_gather_prefix<uint4>(src, dst, indices.data<int32_t>(), indices.size(),
                                 num_slices, slice_bytes, prefix_bytes);
      else
        run_gather_prefix<Raw>(src, dst, indices.data<int32_t>(), indices.size(),
                               num_slices, slice_bytes, prefix_bytes);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gather::compute<Device::CUDA, T>(const StorageView& data,           \
                                     const StorageView& input,          \
                                     const dim_t axis,                  \
                                     const dim_t batch_dims,            \
                                     StorageView& output) const;        \
    template void                                                       \
    GatherPrefix::compute<Device::CUDA, T>(const StorageView& data,     \
                                           const StorageView& indices,  \
                                           const dim_t axis,            \
                                           StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)

  }
}
