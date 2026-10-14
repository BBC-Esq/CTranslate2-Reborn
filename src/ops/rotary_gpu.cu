#include "ctranslate2/ops/rotary.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    struct ComputeType {
      using type = T;
    };

#if !CUDA_CAN_USE_HALF
    template<>
    struct ComputeType<__half> {
      using type = float;
    };
#endif

#if !CUDA_CAN_USE_BF16_MATH
    template<>
    struct ComputeType<__nv_bfloat16> {
      using type = float;
    };
#endif

    template <typename T, bool interleave>
    __global__ void rotary_kernel(const T* x,
                                  const T* sin,
                                  const T* cos,
                                  T* y,
                                  const cuda::index_t max_time,
                                  const cuda::index_t head_size,
                                  const cuda::index_t ndims,
                                  const cuda::index_t depth,
                                  const bool transpose) {
      const auto time = transpose ? blockIdx.x % max_time : blockIdx.x / head_size;
      const auto middle = ndims / 2;

      x += blockIdx.x * depth;
      y += blockIdx.x * depth;

      sin += time * ndims;
      cos += time * ndims;

      using C = typename ComputeType<T>::type;

      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x) {
        if (i >= ndims)
          y[i] = x[i];
        else if (interleave)
          y[i] = C(x[i]) * C(cos[i]) + (i % 2 == 0 ? -C(x[i + 1]) : C(x[i - 1])) * C(sin[i]);
        else
          y[i] = C(x[i]) * C(cos[i]) + (i < middle ? -C(x[i + middle]) : C(x[i - middle])) * C(sin[i]);
      }
    }

    // The same computation with one warp per row and vector loads of 4 values: the kernel above
    // uses a block per row and loads one value per thread, which is slow for attention heads.
    constexpr int rotary_warp_size = C10_WARP_SIZE;
    constexpr int rotary_warps_per_block = 8;
    constexpr int rotary_vector_size = 4;

    template <typename T>
    struct alignas(sizeof (T) * rotary_vector_size) RotaryVector {
      T values[rotary_vector_size];
    };

    template <typename T, bool interleave>
    __global__ void rotary_warp_kernel(const T* x,
                                       const T* sin,
                                       const T* cos,
                                       T* y,
                                       const cuda::index_t rows,
                                       const cuda::index_t max_time,
                                       const cuda::index_t head_size,
                                       const cuda::index_t ndims,
                                       const cuda::index_t depth,
                                       const bool transpose) {
      const cuda::index_t row = blockIdx.x * rotary_warps_per_block + threadIdx.x / rotary_warp_size;
      if (row >= rows)
        return;

      const auto time = transpose ? row % max_time : row / head_size;
      const cuda::index_t num_vectors = depth / rotary_vector_size;
      const cuda::index_t rotated_vectors = ndims / rotary_vector_size;
      const cuda::index_t middle_vectors = rotated_vectors / 2;

      using Vector = RotaryVector<T>;
      const auto* x_vectors = reinterpret_cast<const Vector*>(x + row * depth);
      const auto* sin_vectors = reinterpret_cast<const Vector*>(sin + time * ndims);
      const auto* cos_vectors = reinterpret_cast<const Vector*>(cos + time * ndims);
      auto* y_vectors = reinterpret_cast<Vector*>(y + row * depth);

      using C = typename ComputeType<T>::type;

      for (cuda::index_t v = threadIdx.x % rotary_warp_size; v < num_vectors; v += rotary_warp_size) {
        const Vector x_vector = x_vectors[v];
        if (v >= rotated_vectors) {
          y_vectors[v] = x_vector;
          continue;
        }

        const Vector sin_vector = sin_vectors[v];
        const Vector cos_vector = cos_vectors[v];
        Vector y_vector;

        if (interleave) {
#pragma unroll
          for (int j = 0; j < rotary_vector_size; ++j)
            y_vector.values[j] = C(x_vector.values[j]) * C(cos_vector.values[j])
              + (j % 2 == 0 ? -C(x_vector.values[j + 1]) : C(x_vector.values[j - 1]))
              * C(sin_vector.values[j]);
        } else {
          const bool first_half = v < middle_vectors;
          const Vector other = x_vectors[first_half ? v + middle_vectors : v - middle_vectors];
#pragma unroll
          for (int j = 0; j < rotary_vector_size; ++j)
            y_vector.values[j] = C(x_vector.values[j]) * C(cos_vector.values[j])
              + (first_half ? -C(other.values[j]) : C(other.values[j]))
              * C(sin_vector.values[j]);
        }

        y_vectors[v] = y_vector;
      }
    }

    template <typename T>
    static bool is_rotary_vector_aligned(const T* ptr) {
      return reinterpret_cast<uintptr_t>(ptr) % sizeof (RotaryVector<T>) == 0;
    }

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output,
                         bool is_transposed) const {
      const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);
      const dim_t head_size = is_transposed ? input.dim(-3) : input.dim(-2);
      const dim_t depth = input.dim(-1);
      const dim_t ndims = _ndims == 0 ? depth : _ndims;

      const dim_t blocks = std::min(input.size() / depth, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      const auto* x = cuda::device_cast(input.data<T>());
      const auto* s = cuda::device_cast(sin.data<T>());
      const auto* c = cuda::device_cast(cos.data<T>());
      auto* y = cuda::device_cast(output.data<T>());

      using DeviceT = cuda::device_type<T>;

      if (depth % rotary_vector_size == 0
          && ndims % (2 * rotary_vector_size) == 0
          && is_rotary_vector_aligned(x)
          && is_rotary_vector_aligned(s)
          && is_rotary_vector_aligned(c)
          && is_rotary_vector_aligned(y)) {
        const dim_t rows = input.size() / depth;
        const dim_t warp_blocks = ceil_divide(rows, dim_t(rotary_warps_per_block));
        const dim_t warp_threads = rotary_warps_per_block * rotary_warp_size;
        if (_interleave)
          rotary_warp_kernel<DeviceT, true><<<warp_blocks, warp_threads, 0, cuda::get_cuda_stream()>>>(
            x, s, c, y, rows, max_time, head_size, ndims, depth, is_transposed);
        else
          rotary_warp_kernel<DeviceT, false><<<warp_blocks, warp_threads, 0, cuda::get_cuda_stream()>>>(
            x, s, c, y, rows, max_time, head_size, ndims, depth, is_transposed);
        return;
      }

      if (_interleave)
        rotary_kernel<DeviceT, true><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          x, s, c, y, max_time, head_size, ndims, depth, is_transposed);
      else
        rotary_kernel<DeviceT, false><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          x, s, c, y, max_time, head_size, ndims, depth, is_transposed);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::CUDA, T>(const StorageView&,                \
                                     const StorageView&,                \
                                     const StorageView&,                \
                                     StorageView&,                       \
                                     bool) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
