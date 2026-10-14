#include "ctranslate2/ops/rms_norm_rotary.h"

#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/ops/rotary.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    RMSNormRotary::RMSNormRotary(const float epsilon,
                                 const bool queries_use_residual,
                                 const bool keys_use_residual,
                                 const bool interleave)
      : _epsilon(epsilon)
      , _queries_use_residual(queries_use_residual)
      , _keys_use_residual(keys_use_residual)
      , _interleave(interleave)
    {
    }

    void RMSNormRotary::operator()(const StorageView& queries_gamma,
                                   const StorageView& keys_gamma,
                                   const StorageView& sin,
                                   const StorageView& cos,
                                   StorageView& queries,
                                   StorageView& keys) const {
      PROFILE("RMSNormRotary");

#ifdef CT2_WITH_CUDA
      if (queries.device() == Device::CUDA) {
        bool fused = false;
        switch (queries.dtype()) {
        case DataType::FLOAT32:
          fused = compute_fused<float>(queries_gamma, keys_gamma, sin, cos, queries, keys);
          break;
        case DataType::FLOAT16:
          fused = compute_fused<float16_t>(queries_gamma, keys_gamma, sin, cos, queries, keys);
          break;
        case DataType::BFLOAT16:
          fused = compute_fused<bfloat16_t>(queries_gamma, keys_gamma, sin, cos, queries, keys);
          break;
        default:
          break;
        }
        if (fused)
          return;
      }
#endif

      const RMSNorm queries_norm(_epsilon, _queries_use_residual);
      const RMSNorm keys_norm(_epsilon, _keys_use_residual);
      queries_norm(queries_gamma, queries, queries);
      keys_norm(keys_gamma, keys, keys);

      const Rotary rotary_op(sin.dim(1), _interleave);
      StorageView rotated(queries.dtype(), queries.device());
      rotary_op(queries, sin, cos, rotated);
      queries = std::move(rotated);
      rotated = StorageView(keys.dtype(), keys.device());
      rotary_op(keys, sin, cos, rotated);
      keys = std::move(rotated);
    }

  }
}
