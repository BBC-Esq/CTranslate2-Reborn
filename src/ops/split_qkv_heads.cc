#include "ctranslate2/ops/split_qkv_heads.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    SplitQKVHeads::SplitQKVHeads(const dim_t num_heads, const dim_t num_heads_kv)
      : _num_heads(num_heads)
      , _num_heads_kv(num_heads_kv)
    {
    }

    void SplitQKVHeads::operator()(const StorageView& input,
                                   StorageView& queries,
                                   StorageView& keys,
                                   StorageView& values) const {
      PROFILE("SplitQKVHeads");
      if (input.rank() != 3)
        throw std::invalid_argument("SplitQKVHeads: expected an input of rank 3, got rank "
                                    + std::to_string(input.rank()));

      const dim_t batch_size = input.dim(0);
      const dim_t time = input.dim(1);
      const dim_t total_heads = _num_heads + 2 * _num_heads_kv;
      if (input.dim(2) % total_heads != 0)
        throw std::invalid_argument("SplitQKVHeads: the input depth " + std::to_string(input.dim(2))
                                    + " is not divisible by the number of heads "
                                    + std::to_string(total_heads));
      const dim_t head_dim = input.dim(2) / total_heads;

      queries.resize({batch_size, _num_heads, time, head_dim});
      keys.resize({batch_size, _num_heads_kv, time, head_dim});
      values.resize({batch_size, _num_heads_kv, time, head_dim});

      DEVICE_AND_FLOAT_DISPATCH("SplitQKVHeads", input.device(), input.dtype(),
                                (compute<D, T>(input, queries, keys, values)));
    }

  }
}
