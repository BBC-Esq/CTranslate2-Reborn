#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class SoftMax : public UnaryOp {
    public:
      // With lengths, row i is normalized over its first lengths[i] values. A positive window
      // further restricts it to the last window of those values (sliding window attention).
      SoftMax(bool log = false, dim_t window = 0);

      using UnaryOp::operator();
      void operator()(StorageView& x) const;
      void operator()(const StorageView& x, StorageView& y) const override;
      void operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const;
      void operator()(const StorageView& x, const StorageView* lengths, StorageView& y) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input, const StorageView* lengths, StorageView& output) const;

      bool _log;
      dim_t _window;
    };


    class LogSoftMax : public SoftMax {
    public:
      LogSoftMax();
    };

  }
}
