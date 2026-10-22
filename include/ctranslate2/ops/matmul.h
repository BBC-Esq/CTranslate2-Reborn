#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class MatMul : public BinaryOp {
    public:
      MatMul(bool trans_a = false, bool trans_b = false, float alpha = 1);
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const;

      // Multiplies by the first b_rows rows of each matrix of b, along its dimension -2, as if b
      // had b_rows rows. The matrices of b keep their stride, so b can be a preallocated cache
      // of which only the first positions are filled.
      void operator()(const StorageView& a,
                      const StorageView& b,
                      StorageView& c,
                      dim_t b_rows) const;

    private:
      bool _trans_a;
      bool _trans_b;
      float _alpha;

      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c, dim_t b_rows) const;
    };

  }
}
