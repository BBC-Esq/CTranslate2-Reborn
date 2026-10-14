#include "ctranslate2/ops/gated_activation.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    GatedActivation::GatedActivation(const ActivationType activation_type)
      : _activation_type(activation_type)
    {
    }

    void GatedActivation::operator()(const StorageView& gate,
                                     const StorageView& up,
                                     StorageView& output) const {
      PROFILE("GatedActivation");
      if (gate.shape() != up.shape())
        throw std::invalid_argument("GatedActivation: the gate and up inputs should have the same shape");
      DEVICE_AND_FLOAT_DISPATCH("GatedActivation", gate.device(), gate.dtype(),
                                (compute<D, T>(gate, up, output)));
    }

  }
}
