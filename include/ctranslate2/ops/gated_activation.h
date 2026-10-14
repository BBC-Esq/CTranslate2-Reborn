#pragma once

#include "activation.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Computes up * activation(gate), the gated activation of feed-forward networks such as SwiGLU
    // and GeGLU. The result is the same as the activation op followed by Mul; on CUDA it is
    // computed in one pass.
    class GatedActivation : public BinaryOp {
    public:
      GatedActivation(const ActivationType activation_type);

      void operator()(const StorageView& gate, const StorageView& up, StorageView& output) const override;

    private:
      const ActivationType _activation_type;

      template <Device D, typename T>
      void compute(const StorageView& gate, const StorageView& up, StorageView& output) const;
    };

  }
}
