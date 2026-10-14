#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Normalizes each query and key head with RMSNorm and then applies rotary embeddings to it, in
    // place, as the self-attention of Qwen3 and Gemma 3 does. Queries and keys are
    // [batch, heads, time, depth], the gammas are [depth] and sin and cos are [time, rotary_dim].
    // The result is the same as RMSNorm followed by Rotary; on CUDA it is computed in one pass when
    // the sizes allow it.
    class RMSNormRotary : public Op {
    public:
      RMSNormRotary(const float epsilon,
                    const bool queries_use_residual,
                    const bool keys_use_residual,
                    const bool interleave);

      void operator()(const StorageView& queries_gamma,
                      const StorageView& keys_gamma,
                      const StorageView& sin,
                      const StorageView& cos,
                      StorageView& queries,
                      StorageView& keys) const;

    private:
      const float _epsilon;
      const bool _queries_use_residual;
      const bool _keys_use_residual;
      const bool _interleave;

      // Returns false when the sizes or alignments do not allow the one-pass kernel.
      template <typename T>
      bool compute_fused(const StorageView& queries_gamma,
                         const StorageView& keys_gamma,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& queries,
                         StorageView& keys) const;
    };

  }
}
