#include "ctranslate2/ops/slice_scatter.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    SliceScatter::SliceScatter(dim_t axis, dim_t index)
      : _axis(axis)
      , _index(index) {
    }

    void SliceScatter::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("SliceScatter");

      if (input.dtype() != output.dtype())
        throw std::invalid_argument("SliceScatter: the input and output types differ");
      if (input.device() != output.device())
        throw std::invalid_argument("SliceScatter: the input and output devices differ");
      if (input.rank() != output.rank())
        throw std::invalid_argument("SliceScatter: the input and output ranks differ");

      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      if (axis < 0 || axis >= output.rank())
        throw std::invalid_argument("SliceScatter: invalid axis " + std::to_string(_axis));

      for (dim_t i = 0; i < output.rank(); ++i) {
        if (i != axis && input.dim(i) != output.dim(i))
          throw std::invalid_argument("SliceScatter: dimension " + std::to_string(i)
                                      + " of the input (" + std::to_string(input.dim(i))
                                      + ") and the output (" + std::to_string(output.dim(i))
                                      + ") differ");
      }

      if (_index < 0 || _index + input.dim(axis) > output.dim(axis))
        throw std::invalid_argument("SliceScatter: positions " + std::to_string(_index)
                                    + " to " + std::to_string(_index + input.dim(axis))
                                    + " are out of the output range [0, "
                                    + std::to_string(output.dim(axis)) + ")");

      if (input.empty())
        return;

      DEVICE_AND_TYPE_DISPATCH(output.device(), output.dtype(),
                               (compute<D, T>(input, output, axis)));
    }

  }
}
