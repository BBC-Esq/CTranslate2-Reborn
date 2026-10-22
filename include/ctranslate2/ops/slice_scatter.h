#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Writes the input into the output at an index along an axis, in place:
    //
    //   output[..., index:index + input.dim(axis), ...] = input
    //
    // The other dimensions of the input and the output must match. This is the reverse of Slide,
    // and it writes the new positions into a preallocated cache without reallocating it.
    class SliceScatter : public Op {
    public:
      SliceScatter(dim_t axis, dim_t index);

      void operator()(const StorageView& input, StorageView& output) const;

    private:
      const dim_t _axis;
      const dim_t _index;

      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output, const dim_t axis) const;
    };

  }
}
