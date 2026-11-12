#include "ctranslate2/ops/split_qkv_heads.h"

#include <algorithm>

#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/transpose.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void SplitQKVHeads::compute(const StorageView& input,
                                StorageView& queries,
                                StorageView& keys,
                                StorageView& values) const {
      const dim_t time = input.dim(1);
      const dim_t head_dim = queries.dim(3);
      const dim_t total_heads = _num_heads + 2 * _num_heads_kv;
      const T* x = input.data<T>();
      T* q = queries.data<T>();
      T* k = keys.data<T>();
      T* v = values.data<T>();

      // Each (batch, time) row holds the query heads, then the key heads, then the value heads.
      cpu::parallel_for(0, input.dim(0) * time, 1, [&](dim_t begin, dim_t end) {
        for (dim_t row = begin; row < end; ++row) {
          const dim_t b = row / time;
          const dim_t t = row % time;
          for (dim_t h = 0; h < total_heads; ++h) {
            T* output = q;
            dim_t num_heads = _num_heads;
            dim_t head = h;
            if (h >= _num_heads + _num_heads_kv) {
              output = v;
              num_heads = _num_heads_kv;
              head = h - _num_heads - _num_heads_kv;
            } else if (h >= _num_heads) {
              output = k;
              num_heads = _num_heads_kv;
              head = h - _num_heads;
            }
            const T* src = x + (row * total_heads + h) * head_dim;
            std::copy(src, src + head_dim, output + ((b * num_heads + head) * time + t) * head_dim);
          }
        }
      });
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SplitQKVHeads::compute<Device::CPU, T>(const StorageView&,          \
                                           StorageView&,                \
                                           StorageView&,                \
                                           StorageView&) const;

    DECLARE_IMPL(float)

    // The same result with a Split and one Transpose per output, for a device without a kernel
    // for this op.
    [[maybe_unused]] static void split_qkv_heads_with_ops(const StorageView& input,
                                                          const dim_t num_heads,
                                                          const dim_t num_heads_kv,
                                                          StorageView& queries,
                                                          StorageView& keys,
                                                          StorageView& values) {
      const dim_t batch_size = input.dim(0);
      const dim_t time = input.dim(1);
      const dim_t head_dim = input.dim(2) / (num_heads + 2 * num_heads_kv);
      StorageView q(input.dtype(), input.device());
      StorageView k(input.dtype(), input.device());
      StorageView v(input.dtype(), input.device());
      const Split split_op(2, {num_heads * head_dim, num_heads_kv * head_dim, num_heads_kv * head_dim});
      split_op(input, q, k, v);
      q.reshape({batch_size, time, num_heads, head_dim});
      k.reshape({batch_size, time, num_heads_kv, head_dim});
      v.reshape({batch_size, time, num_heads_kv, head_dim});
      const Transpose transpose_op({0, 2, 1, 3});
      transpose_op(q, queries);
      transpose_op(k, keys);
      transpose_op(v, values);
    }

#ifdef CT2_WITH_MPS
    // The Metal backend has no kernel for this op. Its layers don't use it (see
    // MultiHeadAttention), but it stays correct there.
#define DECLARE_MPS_IMPL(T)                                             \
    template<>                                                          \
    void SplitQKVHeads::compute<Device::MPS, T>(const StorageView& input, \
                                                StorageView& queries,   \
                                                StorageView& keys,      \
                                                StorageView& values) const { \
      split_qkv_heads_with_ops(input, _num_heads, _num_heads_kv, queries, keys, values); \
    }

    DECLARE_MPS_IMPL(float)
    DECLARE_MPS_IMPL(float16_t)
    DECLARE_MPS_IMPL(bfloat16_t)
#endif

  }
}
