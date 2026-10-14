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

  }
}
