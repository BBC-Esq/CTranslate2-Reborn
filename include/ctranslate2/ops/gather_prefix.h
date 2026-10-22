#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Gathers along the first dimension like Gather, but copies only the first `length`
    // positions of `axis`. The output keeps the size of `axis`, and its other positions are left
    // uninitialized:
    //
    //   output[i, ..., 0:length, ...] = data[indices[i], ..., 0:length, ...]
    //
    // This reorders a preallocated cache when only its first positions have been written.
    class GatherPrefix : public Op {
    public:
      GatherPrefix(dim_t axis, dim_t length);

      void operator()(StorageView& data, const StorageView& indices) const;
      void operator()(const StorageView& data,
                      const StorageView& indices,
                      StorageView& output) const;

    private:
      const dim_t _axis;
      const dim_t _length;

      dim_t check_arguments(const StorageView& data, const StorageView& indices) const;

      template <Device D, typename T>
      void compute(const StorageView& data,
                   const StorageView& indices,
                   const dim_t axis,
                   StorageView& output) const;
    };

  }
}
