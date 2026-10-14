#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Splits the fused projection of self-attention, [batch, time, (num_heads + 2 * num_heads_kv) * head_dim],
    // into queries [batch, num_heads, time, head_dim] and keys and values [batch, num_heads_kv, time, head_dim].
    // This is a Split followed by one transpose per output, done in a single pass.
    class SplitQKVHeads : public Op {
    public:
      SplitQKVHeads(const dim_t num_heads, const dim_t num_heads_kv);

      void operator()(const StorageView& input,
                      StorageView& queries,
                      StorageView& keys,
                      StorageView& values) const;

    private:
      const dim_t _num_heads;
      const dim_t _num_heads_kv;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   StorageView& queries,
                   StorageView& keys,
                   StorageView& values) const;
    };

  }
}
