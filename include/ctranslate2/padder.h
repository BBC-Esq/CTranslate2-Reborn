#pragma once

#include "ops/gather.h"
#include "storage_view.h"

namespace ctranslate2 {

  // This class can be used to dynamically remove or add padding.
  // This is useful to save on computation when lengths are very different.
  class Padder {
  public:
    static inline bool allow_padding_removal(const Device device,
                                             const ComputeType compute_type) {
      return device == Device::CPU || compute_type != ComputeType::FLOAT16;
    }

    // On a GPU, removing and restoring the padding costs more than computing the padded
    // positions when they are only a few percent of the batch.
    static inline float min_padding_to_remove(const Device device) {
      return device == Device::CUDA ? 0.05f : 0.f;
    }

    // If max_time is negative, it is set to the maximum length. The padding is kept when it is
    // less than min_padding of the batch.
    Padder(const StorageView& lengths,
           const dim_t max_time = -1,
           const dim_t pad_batch_to_multiple = 1,
           const float min_padding = 0);

    // Merge batch and time dimensions and remove padding.
    void remove_padding(StorageView& x) const;

    // Split first dimension into batch and time dimensions and add padding.
    void add_padding(StorageView& x) const;

  private:
    dim_t _batch_size;
    dim_t _max_time;
    StorageView _padded_to_flat;
    StorageView _flat_to_padded;
    const ops::Gather _gather_op;
  };

}
