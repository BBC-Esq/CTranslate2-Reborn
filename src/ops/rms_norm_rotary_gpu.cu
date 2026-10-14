#include "ctranslate2/ops/rms_norm_rotary.h"

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    // One warp per head, with the arithmetic of rms_norm_warp_kernel (rms_norm_gpu.cu) followed by
    // that of rotary_warp_kernel (rotary_gpu.cu), so the results are identical to RMSNorm then
    // Rotary. The normalized head goes through shared memory to reach the lanes that rotate it.
    constexpr int warp_size = C10_WARP_SIZE;
    constexpr int warps_per_block = 8;
    constexpr int vector_size = 4;
    constexpr int max_vectors_per_thread = 2;
    constexpr dim_t max_depth = warp_size * vector_size * max_vectors_per_thread;

    template <typename T>
    struct alignas(sizeof (T) * vector_size) NormRotaryVector {
      T values[vector_size];
    };

    template <typename T, bool interleave>
    __global__ void rms_norm_rotary_kernel(T* queries,
                                           T* keys,
                                           const T* queries_gamma,
                                           const T* keys_gamma,
                                           const T* sin,
                                           const T* cos,
                                           const cuda::index_t queries_rows,
                                           const cuda::index_t keys_rows,
                                           const cuda::index_t max_time,
                                           const cuda::index_t depth,
                                           const cuda::index_t ndims,
                                           const float epsilon,
                                           const bool queries_use_residual,
                                           const bool keys_use_residual) {
      using Vector = NormRotaryVector<T>;
      __shared__ Vector normed[warps_per_block][max_depth / vector_size];

      const cuda::index_t warp = threadIdx.x / warp_size;
      const cuda::index_t lane = threadIdx.x % warp_size;
      cuda::index_t row = blockIdx.x * warps_per_block + warp;
      if (row >= queries_rows + keys_rows)
        return;

      const bool is_query = row < queries_rows;
      if (!is_query)
        row -= queries_rows;
      auto* x_vectors = reinterpret_cast<Vector*>((is_query ? queries : keys) + row * depth);
      const auto* g = reinterpret_cast<const Vector*>(is_query ? queries_gamma : keys_gamma);
      const bool use_residual = is_query ? queries_use_residual : keys_use_residual;
      const cuda::index_t num_vectors = depth / vector_size;

      float values[max_vectors_per_thread][vector_size];
      float sum_squares = 0;

#pragma unroll
      for (int k = 0; k < max_vectors_per_thread; ++k) {
        const cuda::index_t i = lane + k * warp_size;
        if (i < num_vectors) {
          const Vector vector = x_vectors[i];
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
          const Vector weights = g[i];
          Vector vector;
#pragma unroll
          for (int j = 0; j < vector_size; ++j) {
            const float weight = float(weights.values[j]);
            vector.values[j] = values[k][j] * inv_rms * (use_residual ? 1 + weight : weight);
          }
          normed[warp][i] = vector;
        }
      }

      __syncwarp();

      const cuda::index_t time = row % max_time;
      const cuda::index_t rotated_vectors = ndims / vector_size;
      const cuda::index_t middle_vectors = rotated_vectors / 2;
      const auto* sin_vectors = reinterpret_cast<const Vector*>(sin + time * ndims);
      const auto* cos_vectors = reinterpret_cast<const Vector*>(cos + time * ndims);

      using C = typename cuda::MathType<T>::type;

      for (cuda::index_t v = lane; v < num_vectors; v += warp_size) {
        const Vector x_vector = normed[warp][v];
        if (v >= rotated_vectors) {
          x_vectors[v] = x_vector;
          continue;
        }

        const Vector sin_vector = sin_vectors[v];
        const Vector cos_vector = cos_vectors[v];
        Vector y_vector;

        if (interleave) {
#pragma unroll
          for (int j = 0; j < vector_size; ++j)
            y_vector.values[j] = C(x_vector.values[j]) * C(cos_vector.values[j])
              + (j % 2 == 0 ? -C(x_vector.values[j + 1]) : C(x_vector.values[j - 1]))
              * C(sin_vector.values[j]);
        } else {
          const bool first_half = v < middle_vectors;
          const Vector other = normed[warp][first_half ? v + middle_vectors : v - middle_vectors];
#pragma unroll
          for (int j = 0; j < vector_size; ++j)
            y_vector.values[j] = C(x_vector.values[j]) * C(cos_vector.values[j])
              + (first_half ? -C(other.values[j]) : C(other.values[j]))
              * C(sin_vector.values[j]);
        }

        x_vectors[v] = y_vector;
      }
    }

    template <typename T>
    bool RMSNormRotary::compute_fused(const StorageView& queries_gamma,
                                      const StorageView& keys_gamma,
                                      const StorageView& sin,
                                      const StorageView& cos,
                                      StorageView& queries,
                                      StorageView& keys) const {
      using DeviceT = cuda::device_type<T>;
      const dim_t depth = queries.dim(-1);
      const dim_t ndims = sin.dim(1);
      const dim_t max_time = queries.dim(-2);
      const auto aligned = [](const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) % sizeof (NormRotaryVector<DeviceT>) == 0;
      };

      // The conditions under which RMSNorm and Rotary use their warp kernels.
      if (depth > max_depth
          || depth % vector_size != 0
          || ndims % (2 * vector_size) != 0
          || ndims > depth
          || keys.dim(-1) != depth
          || keys.dim(-2) != max_time
          || queries_gamma.size() != depth
          || keys_gamma.size() != depth
          || !aligned(queries.buffer())
          || !aligned(keys.buffer())
          || !aligned(queries_gamma.buffer())
          || !aligned(keys_gamma.buffer())
          || !aligned(sin.buffer())
          || !aligned(cos.buffer()))
        return false;

      const dim_t queries_rows = queries.size() / depth;
      const dim_t keys_rows = keys.size() / depth;
      const dim_t blocks = ceil_divide(queries_rows + keys_rows, dim_t(warps_per_block));
      const dim_t threads = warps_per_block * warp_size;

      auto* q = cuda::device_cast(queries.data<T>());
      auto* k = cuda::device_cast(keys.data<T>());
      const auto* qg = cuda::device_cast(queries_gamma.data<T>());
      const auto* kg = cuda::device_cast(keys_gamma.data<T>());
      const auto* s = cuda::device_cast(sin.data<T>());
      const auto* c = cuda::device_cast(cos.data<T>());

      if (_interleave)
        rms_norm_rotary_kernel<DeviceT, true><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          q, k, qg, kg, s, c, queries_rows, keys_rows, max_time, depth, ndims,
          _epsilon, _queries_use_residual, _keys_use_residual);
      else
        rms_norm_rotary_kernel<DeviceT, false><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          q, k, qg, kg, s, c, queries_rows, keys_rows, max_time, depth, ndims,
          _epsilon, _queries_use_residual, _keys_use_residual);
      return true;
    }

#define DECLARE_IMPL(T)                                                 \
    template bool                                                       \
    RMSNormRotary::compute_fused<T>(const StorageView&,                 \
                                    const StorageView&,                 \
                                    const StorageView&,                 \
                                    const StorageView&,                 \
                                    StorageView&,                       \
                                    StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
