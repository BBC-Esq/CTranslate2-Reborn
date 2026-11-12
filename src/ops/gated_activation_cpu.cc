#include "ctranslate2/ops/gated_activation.h"

#include "ctranslate2/ops/mul.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void GatedActivation::compute(const StorageView& gate,
                                  const StorageView& up,
                                  StorageView& output) const {
      get_activation_op(_activation_type)(gate, output);
      Mul()(up, output, output);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GatedActivation::compute<Device::CPU, T>(const StorageView&,        \
                                             const StorageView&,        \
                                             StorageView&) const;

    DECLARE_IMPL(float)

#ifdef CT2_WITH_MPS
    // The Metal backend has no fused kernel: the implementation above runs its activation and
    // product ops. Its layers don't use this op (see FeedForwardNetwork).
#define DECLARE_MPS_IMPL(T)                                             \
    template void                                                       \
    GatedActivation::compute<Device::MPS, T>(const StorageView&,        \
                                             const StorageView&,        \
                                             StorageView&) const;

    DECLARE_MPS_IMPL(float)
    DECLARE_MPS_IMPL(float16_t)
    DECLARE_MPS_IMPL(bfloat16_t)
#endif

  }
}
