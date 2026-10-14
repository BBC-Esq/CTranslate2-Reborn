#include "ctranslate2/ops/rms_norm.h"

#ifdef CT2_USE_HIP
#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_reduce.hpp>
#define cub hipcub
#else
#include <cub/block/block_reduce.cuh>
#endif

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 512;

    template <typename T>
    __global__ void rms_norm_kernel(const T* input,
                                    const T* gamma,
                                    T* output,
                                    cuda::index_t depth,
                                    float epsilon,
                                    bool use_residual) {
      typedef cub::BlockReduce<float, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ float s_inv_rms;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      float sum_squares = 0;
      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x)
        sum_squares += float(input[i]) * float(input[i]);
      sum_squares = BlockReduce(temp_storage).Sum(sum_squares);

      if (threadIdx.x == 0)
        s_inv_rms = rsqrtf(sum_squares / depth + epsilon);

      __syncthreads();

      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x)
        if (use_residual)
          output[i] = float(input[i]) * s_inv_rms * (1 + float(gamma[i]));
        else
          output[i] = float(input[i]) * s_inv_rms * float(gamma[i]);
    }

    // Short rows, such as the per-head query and key norms of Qwen3 and Gemma 3, are normalized by
    // one warp each: one block per row would leave most of its threads idle. Each thread loads
    // vectors of 4 values.
    constexpr int warp_size = C10_WARP_SIZE;
    constexpr int warps_per_block = 8;
    constexpr int vector_size = 4;
    constexpr int max_vectors_per_thread = 2;
    constexpr dim_t warp_kernel_max_depth = warp_size * vector_size * max_vectors_per_thread;

    template <typename T>
    struct alignas(sizeof (T) * vector_size) Vector {
      T values[vector_size];
    };

    template <typename T>
    __global__ void rms_norm_warp_kernel(const T* input,
                                         const T* gamma,
                                         T* output,
                                         cuda::index_t rows,
                                         cuda::index_t depth,
                                         float epsilon,
                                         bool use_residual) {
      const cuda::index_t row = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
      if (row >= rows)
        return;

      const cuda::index_t lane = threadIdx.x % warp_size;
      const cuda::index_t num_vectors = depth / vector_size;
      const auto* x = reinterpret_cast<const Vector<T>*>(input + row * depth);
      const auto* g = reinterpret_cast<const Vector<T>*>(gamma);
      auto* y = reinterpret_cast<Vector<T>*>(output + row * depth);

      float values[max_vectors_per_thread][vector_size];
      float sum_squares = 0;

#pragma unroll
      for (int k = 0; k < max_vectors_per_thread; ++k) {
        const cuda::index_t i = lane + k * warp_size;
        if (i < num_vectors) {
          const Vector<T> vector = x[i];
#pragma unroll
          for (int j = 0; j < vector_size; ++j) {
            values[k][j] = float(vector.values[j]);
            sum_squares += values[k][j] * values[k][j];
          }
        }
      }

      sum_squares = cuda::warp_all_reduce(sum_squares, cuda::plus<float>());
      const float inv_rms = rsqrtf(sum_squares / depth + epsilon);

#pragma unroll
      for (int k = 0; k < max_vectors_per_thread; ++k) {
        const cuda::index_t i = lane + k * warp_size;
        if (i < num_vectors) {
          const Vector<T> weights = g[i];
          Vector<T> vector;
#pragma unroll
          for (int j = 0; j < vector_size; ++j) {
            const float weight = float(weights.values[j]);
            vector.values[j] = values[k][j] * inv_rms * (use_residual ? 1 + weight : weight);
          }
          y[i] = vector;
        }
      }
    }

    template <typename T>
    static bool is_vector_aligned(const T* ptr) {
      return reinterpret_cast<uintptr_t>(ptr) % sizeof (Vector<T>) == 0;
    }

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      const auto* x = cuda::device_cast(input.data<T>());
      const auto* g = cuda::device_cast(gamma.data<T>());
      auto* y = cuda::device_cast(output.data<T>());

      if (depth <= warp_kernel_max_depth
          && depth % vector_size == 0
          && is_vector_aligned(x)
          && is_vector_aligned(g)
          && is_vector_aligned(y)) {
        const dim_t blocks = ceil_divide(batch_size, dim_t(warps_per_block));
        rms_norm_warp_kernel<<<blocks, warps_per_block * warp_size, 0, cuda::get_cuda_stream()>>>(
          x, g, y, batch_size, depth, _epsilon, _use_residual);
      } else {
        rms_norm_kernel<<<batch_size, num_threads, 0, cuda::get_cuda_stream()>>>(
          x, g, y, depth, _epsilon, _use_residual);
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void RMSNorm::compute<Device::CUDA, T>(const StorageView&, \
                                                    const StorageView&, \
                                                    StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
