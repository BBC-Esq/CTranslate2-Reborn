#include "ctranslate2/ops/split_qkv_heads.h"

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    // One block per (batch, time) row, which holds the query heads, then the key heads, then the
    // value heads. Values are copied in units of VecT (16 bytes when the head size allows it).
    template <typename VecT>
    __global__ void split_qkv_heads_kernel(const VecT* input,
                                           VecT* queries,
                                           VecT* keys,
                                           VecT* values,
                                           const cuda::index_t time,
                                           const cuda::index_t num_heads,
                                           const cuda::index_t num_heads_kv,
                                           const cuda::index_t head_size) {
      const cuda::index_t row = blockIdx.x;
      const cuda::index_t b = row / time;
      const cuda::index_t t = row % time;
      const cuda::index_t total_heads = num_heads + 2 * num_heads_kv;
      input += row * total_heads * head_size;

      for (cuda::index_t i = threadIdx.x; i < total_heads * head_size; i += blockDim.x) {
        const cuda::index_t h = i / head_size;
        VecT* output = queries;
        cuda::index_t heads = num_heads;
        cuda::index_t head = h;
        if (h >= num_heads + num_heads_kv) {
          output = values;
          heads = num_heads_kv;
          head = h - num_heads - num_heads_kv;
        } else if (h >= num_heads) {
          output = keys;
          heads = num_heads_kv;
          head = h - num_heads;
        }
        output[((b * heads + head) * time + t) * head_size + i % head_size] = input[i];
      }
    }

    template <typename VecT>
    static void split_qkv_heads(const void* input,
                                void* queries,
                                void* keys,
                                void* values,
                                const dim_t rows,
                                const dim_t time,
                                const dim_t num_heads,
                                const dim_t num_heads_kv,
                                const dim_t head_size) {
      const dim_t row_size = (num_heads + 2 * num_heads_kv) * head_size;
      const dim_t threads = std::min(row_size, dim_t(256));
      split_qkv_heads_kernel<<<rows, threads, 0, cuda::get_cuda_stream()>>>(
        static_cast<const VecT*>(input),
        static_cast<VecT*>(queries),
        static_cast<VecT*>(keys),
        static_cast<VecT*>(values),
        time,
        num_heads,
        num_heads_kv,
        head_size);
    }

    template <Device D, typename T>
    void SplitQKVHeads::compute(const StorageView& input,
                                StorageView& queries,
                                StorageView& keys,
                                StorageView& values) const {
      const dim_t rows = input.dim(0) * input.dim(1);
      const dim_t time = input.dim(1);
      const dim_t head_bytes = queries.dim(3) * sizeof (T);
      const auto aligned = [](const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) % sizeof (uint4) == 0;
      };

      if (head_bytes % sizeof (uint4) == 0
          && aligned(input.buffer())
          && aligned(queries.buffer())
          && aligned(keys.buffer())
          && aligned(values.buffer()))
        split_qkv_heads<uint4>(input.buffer(), queries.buffer(), keys.buffer(), values.buffer(),
                               rows, time, _num_heads, _num_heads_kv, head_bytes / sizeof (uint4));
      else
        split_qkv_heads<cuda::device_type<T>>(input.buffer(), queries.buffer(), keys.buffer(), values.buffer(),
                                              rows, time, _num_heads, _num_heads_kv, queries.dim(3));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SplitQKVHeads::compute<Device::CUDA, T>(const StorageView&,         \
                                            StorageView&,               \
                                            StorageView&,               \
                                            StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
