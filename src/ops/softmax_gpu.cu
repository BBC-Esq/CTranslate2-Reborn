#include "ctranslate2/ops/softmax.h"

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const T* x,
                               const dim_t rows,
                               const dim_t cols,
                               const int32_t* lengths,
                               const dim_t window,
                               T* y);

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      softmax_kernel(cuda::get_cuda_stream(),
                     _log,
                     input.data<T>(),
                     batch_size,
                     depth,
                     lengths ? lengths->data<int32_t>() : nullptr,
                     _window,
                     output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CUDA, T>(const StorageView& input,         \
                                      const StorageView* lengths,       \
                                      StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}

// The following CUDA kernels are adapted from:
// https://github.com/pytorch/pytorch/blob/40eff454ce5638fbff638a7f4502e29ffb9a2f0d/aten/src/ATen/native/cuda/SoftMax.cu
// which has the following license notice:

/*
  From PyTorch:

  Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
  Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
  Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
  Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
  Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
  Copyright (c) 2011-2013 NYU                      (Clement Farabet)
  Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
  Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
  Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

  From Caffe2:

  Copyright (c) 2016-present, Facebook Inc. All rights reserved.

  All contributions by Facebook:
  Copyright (c) 2016 Facebook Inc.

  All contributions by Google:
  Copyright (c) 2015 Google Inc.
  All rights reserved.

  All contributions by Yangqing Jia:
  Copyright (c) 2015 Yangqing Jia
  All rights reserved.

  All contributions from Caffe:
  Copyright(c) 2013, 2014, 2015, the respective contributors
  All rights reserved.

  All other contributions:
  Copyright(c) 2015, 2016 the respective contributors
  All rights reserved.

  Caffe2 uses a copyright model similar to Caffe: each contributor holds
  copyright over their contributions to Caffe2. The project versioning records
  all such contribution and copyright details. If a contributor wants to further
  mark their specific copyright on a particular contribution, they should
  indicate their copyright solely in the commit message of the change when it is
  committed.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
     and IDIAP Research Institute nor the names of its contributors may be
     used to endorse or promote products derived from this software without
     specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/

namespace at {
  namespace native {

    constexpr float max_float = std::numeric_limits<float>::max();

    template<typename T, typename AccumT, typename OutT>
    struct LogSoftMaxForwardEpilogue {
      __device__ __forceinline__ LogSoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : max_input(max_input),  logsum(std::log(sum)) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(static_cast<AccumT>(input) - max_input - logsum);
      }

      const AccumT max_input;
      const AccumT logsum;
    };

    template<typename T, typename AccumT, typename OutT>
    struct SoftMaxForwardEpilogue {
      __device__ __forceinline__ SoftMaxForwardEpilogue(AccumT max_input, AccumT sum)
        : max_input(max_input)
        , sum(sum) {}

      __device__ __forceinline__ OutT operator()(T input) const {
        return static_cast<OutT>(std::exp(static_cast<AccumT>(input) - max_input) / sum);
      }

      const AccumT max_input;
      const AccumT sum;
    };

    template<typename T, typename AccumT>
    struct SumExpFloat
    {
      __device__ __forceinline__ SumExpFloat(AccumT v)
        : max_k(v) {}

      __device__ __forceinline__ AccumT operator()(AccumT sum, T v) const {
        return sum + std::exp(static_cast<AccumT>(v) - max_k);
      }

      const AccumT max_k;
    };

    template<typename T>
    struct Max {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return a < b ? b : a;
      }
    };

    template <typename T, typename AccumT>
    struct MaxFloat
    {
      __device__ __forceinline__ AccumT operator()(AccumT max, T v) const {
        return ::max(max, (AccumT)v);
      }
    };

    template<typename T>
    struct Add {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
      }
    };

    template <typename scalar_t,
              typename accscalar_t,
              typename outscalar_t,
              typename index_t,
              typename length_t,
              template <typename, typename, typename> class Epilogue>
    __global__ void
    cunn_SoftMaxForward(outscalar_t *output,
                        const scalar_t *input,
                        const index_t classes,
                        const length_t *lengths,
                        const index_t window)
    {
      extern __shared__ unsigned char smem[];
      auto sdata = reinterpret_cast<accscalar_t*>(smem);
      // forward pointers to batch[blockIdx.x]
      // each block handles a sample in the mini-batch
      const index_t row = blockIdx.x;
      input += row * classes;
      output += row * classes;

      index_t size = classes;
      if (lengths)
      {
        // Directly set 0 in output for out of range positions.
        size = lengths[row];
        for (index_t i = size + threadIdx.x; i < classes; i += blockDim.x)
          output[i] = 0.f;

        // With a sliding window, the positions before the window are out of range too.
        if (window > 0 && size > window) {
          const index_t start = size - window;
          for (index_t i = threadIdx.x; i < start; i += blockDim.x)
            output[i] = 0.f;
          input += start;
          output += start;
          size = window;
        }
      }

      // find the max
      accscalar_t threadMax = ctranslate2::cuda::ilp_reduce(
        input, size, MaxFloat<scalar_t, accscalar_t>(), -max_float);
      accscalar_t max_k = ctranslate2::cuda::block_reduce(
        sdata, threadMax, Max<accscalar_t>(), -max_float);

      // reduce all values
      accscalar_t threadExp = ctranslate2::cuda::ilp_reduce(
        input, size, SumExpFloat<scalar_t, accscalar_t>(max_k), static_cast<accscalar_t>(0));
      accscalar_t sumAll = ctranslate2::cuda::block_reduce(
        sdata, threadExp, Add<accscalar_t>(), static_cast<accscalar_t>(0));

      // apply epilogue
      ctranslate2::cuda::apply_epilogue(
        input, size, Epilogue<scalar_t, accscalar_t, outscalar_t>(max_k, sumAll), output);
    }

  }
}

namespace ctranslate2 {
  namespace ops {

    // Rows of up to 1024 values, such as attention scores, are handled by one warp each: every
    // thread keeps its values in registers, so a row is read once instead of three times and the
    // reductions are warp shuffles.
    constexpr int softmax_warp_size = C10_WARP_SIZE;
    constexpr int softmax_warps_per_block = 4;
    constexpr dim_t softmax_warp_max_cols = 1024;
    constexpr float softmax_max_float = std::numeric_limits<float>::max();

    template <typename T, int max_cols, bool log_softmax>
    __global__ void softmax_warp_kernel(T* output,
                                        const T* input,
                                        const cuda::index_t rows,
                                        const cuda::index_t cols,
                                        const int32_t* lengths,
                                        const cuda::index_t window) {
      constexpr int values_per_thread = max_cols / softmax_warp_size;

      const cuda::index_t row = blockIdx.x * softmax_warps_per_block + threadIdx.x / softmax_warp_size;
      if (row >= rows)
        return;

      const cuda::index_t lane = threadIdx.x % softmax_warp_size;
      const cuda::index_t size = lengths ? lengths[row] : cols;
      // With a sliding window, only the last window positions before size are in range.
      const cuda::index_t start = lengths && window > 0 && size > window ? size - window : 0;
      input += row * cols;
      output += row * cols;

      float values[values_per_thread];
      float max_value = -softmax_max_float;
#pragma unroll
      for (int k = 0; k < values_per_thread; ++k) {
        const cuda::index_t i = lane + k * softmax_warp_size;
        values[k] = i >= start && i < size ? float(input[i]) : -softmax_max_float;
        max_value = ::max(max_value, values[k]);
      }
      max_value = cuda::warp_all_reduce(max_value, cuda::maximum<float>());

      float sum = 0;
#pragma unroll
      for (int k = 0; k < values_per_thread; ++k) {
        const cuda::index_t i = lane + k * softmax_warp_size;
        if (i >= start && i < size) {
          const float exp_value = std::exp(values[k] - max_value);
          if (!log_softmax)
            values[k] = exp_value;
          sum += exp_value;
        }
      }
      sum = cuda::warp_all_reduce(sum, cuda::plus<float>());

      const float log_sum = log_softmax ? std::log(sum) : 0.f;
#pragma unroll
      for (int k = 0; k < values_per_thread; ++k) {
        const cuda::index_t i = lane + k * softmax_warp_size;
        if (i < cols) {
          if (i < start || i >= size)
            output[i] = T(0.f);
          else if (log_softmax)
            output[i] = T(values[k] - max_value - log_sum);
          else
            output[i] = T(values[k] / sum);
        }
      }
    }

    template <typename T, bool log_softmax>
    static void softmax_warp_kernel_impl(cudaStream_t stream,
                                         const T* x,
                                         const dim_t rows,
                                         const dim_t cols,
                                         const int32_t* lengths,
                                         const dim_t window,
                                         T* y) {
      const dim3 grid(ceil_divide(rows, dim_t(softmax_warps_per_block)));
      const dim3 block(softmax_warps_per_block * softmax_warp_size);
      if (cols <= 32)
        softmax_warp_kernel<T, 32, log_softmax><<<grid, block, 0, stream>>>(y, x, rows, cols, lengths, window);
      else if (cols <= 64)
        softmax_warp_kernel<T, 64, log_softmax><<<grid, block, 0, stream>>>(y, x, rows, cols, lengths, window);
      else if (cols <= 128)
        softmax_warp_kernel<T, 128, log_softmax><<<grid, block, 0, stream>>>(y, x, rows, cols, lengths, window);
      else if (cols <= 256)
        softmax_warp_kernel<T, 256, log_softmax><<<grid, block, 0, stream>>>(y, x, rows, cols, lengths, window);
      else if (cols <= 512)
        softmax_warp_kernel<T, 512, log_softmax><<<grid, block, 0, stream>>>(y, x, rows, cols, lengths, window);
      else
        softmax_warp_kernel<T, 1024, log_softmax><<<grid, block, 0, stream>>>(y, x, rows, cols, lengths, window);
    }

    // Longer rows, up to 8192 values (such as attention scores over long prompts), are handled by
    // a block each, with the same approach: the values stay in registers, so a row is read once,
    // and the reductions are warp shuffles.
    constexpr int softmax_block_warps = 8;
    constexpr dim_t softmax_block_max_cols = softmax_block_warps * softmax_warp_size * 32;

    template <typename Reduction>
    __device__ __forceinline__ float softmax_block_all_reduce(float value,
                                                              const Reduction& reduction,
                                                              const float identity,
                                                              float* warp_values) {
      const int warp = threadIdx.x / softmax_warp_size;
      const int lane = threadIdx.x % softmax_warp_size;
      value = cuda::warp_all_reduce(value, reduction);
      if (lane == 0)
        warp_values[warp] = value;
      __syncthreads();
      value = lane < softmax_block_warps ? warp_values[lane] : identity;
      value = cuda::warp_all_reduce(value, reduction);
      __syncthreads();
      return value;
    }

    template <typename T, int values_per_thread, bool log_softmax>
    __global__ void softmax_block_kernel(T* output,
                                         const T* input,
                                         const cuda::index_t cols,
                                         const int32_t* lengths,
                                         const cuda::index_t window) {
      constexpr int threads = softmax_block_warps * softmax_warp_size;
      __shared__ float warp_values[softmax_block_warps];

      const cuda::index_t row = blockIdx.x;
      const cuda::index_t size = lengths ? lengths[row] : cols;
      // With a sliding window, only the last window positions before size are in range.
      const cuda::index_t start = lengths && window > 0 && size > window ? size - window : 0;
      input += row * cols;
      output += row * cols;

      float values[values_per_thread];
      float max_value = -softmax_max_float;
#pragma unroll
      for (int k = 0; k < values_per_thread; ++k) {
        const cuda::index_t i = threadIdx.x + k * threads;
        values[k] = i >= start && i < size ? float(input[i]) : -softmax_max_float;
        max_value = ::max(max_value, values[k]);
      }
      max_value = softmax_block_all_reduce(max_value, cuda::maximum<float>(), -softmax_max_float, warp_values);

      float sum = 0;
#pragma unroll
      for (int k = 0; k < values_per_thread; ++k) {
        const cuda::index_t i = threadIdx.x + k * threads;
        if (i >= start && i < size) {
          const float exp_value = std::exp(values[k] - max_value);
          if (!log_softmax)
            values[k] = exp_value;
          sum += exp_value;
        }
      }
      sum = softmax_block_all_reduce(sum, cuda::plus<float>(), 0.f, warp_values);

      const float log_sum = log_softmax ? std::log(sum) : 0.f;
#pragma unroll
      for (int k = 0; k < values_per_thread; ++k) {
        const cuda::index_t i = threadIdx.x + k * threads;
        if (i < cols) {
          if (i < start || i >= size)
            output[i] = T(0.f);
          else if (log_softmax)
            output[i] = T(values[k] - max_value - log_sum);
          else
            output[i] = T(values[k] / sum);
        }
      }
    }

    template <typename T, bool log_softmax>
    static void softmax_block_kernel_impl(cudaStream_t stream,
                                          const T* x,
                                          const dim_t rows,
                                          const dim_t cols,
                                          const int32_t* lengths,
                                          const dim_t window,
                                          T* y) {
      constexpr dim_t threads = softmax_block_warps * softmax_warp_size;
      if (cols <= threads * 8)
        softmax_block_kernel<T, 8, log_softmax><<<rows, threads, 0, stream>>>(y, x, cols, lengths, window);
      else if (cols <= threads * 16)
        softmax_block_kernel<T, 16, log_softmax><<<rows, threads, 0, stream>>>(y, x, cols, lengths, window);
      else
        softmax_block_kernel<T, 32, log_softmax><<<rows, threads, 0, stream>>>(y, x, cols, lengths, window);
    }

    template <typename T, template <typename, typename, typename> class Epilogue>
    static void softmax_kernel_impl(cudaStream_t stream,
                                    const T* x,
                                    const dim_t rows,
                                    const dim_t cols,
                                    const int32_t* lengths,
                                    const dim_t window,
                                    T* y) {
      const dim3 grid(rows);
      const dim3 block(cuda::get_block_size(cols));
      at::native::cunn_SoftMaxForward<T, float, T, cuda::index_t, int32_t, Epilogue>
        <<<grid, block, block.x * sizeof (float), stream>>>(y,
                                                            x,
                                                            cols,
                                                            lengths,
                                                            static_cast<cuda::index_t>(window));
    }

    template <typename T>
    static void softmax_kernel(cudaStream_t stream,
                               const bool log_softmax,
                               const T* x,
                               const dim_t rows,
                               const dim_t cols,
                               const int32_t* lengths,
                               const dim_t window,
                               T* y) {
      if (cols <= softmax_warp_max_cols) {
        if (log_softmax)
          softmax_warp_kernel_impl<cuda::device_type<T>, true>(
            stream, cuda::device_cast(x), rows, cols, lengths, window, cuda::device_cast(y));
        else
          softmax_warp_kernel_impl<cuda::device_type<T>, false>(
            stream, cuda::device_cast(x), rows, cols, lengths, window, cuda::device_cast(y));
      } else if (cols <= softmax_block_max_cols) {
        if (log_softmax)
          softmax_block_kernel_impl<cuda::device_type<T>, true>(
            stream, cuda::device_cast(x), rows, cols, lengths, window, cuda::device_cast(y));
        else
          softmax_block_kernel_impl<cuda::device_type<T>, false>(
            stream, cuda::device_cast(x), rows, cols, lengths, window, cuda::device_cast(y));
      } else if (log_softmax)
        softmax_kernel_impl<cuda::device_type<T>, at::native::LogSoftMaxForwardEpilogue>(
          stream, cuda::device_cast(x), rows, cols, lengths, window, cuda::device_cast(y));
      else
        softmax_kernel_impl<cuda::device_type<T>, at::native::SoftMaxForwardEpilogue>(
          stream, cuda::device_cast(x), rows, cols, lengths, window, cuda::device_cast(y));
    }

  }
}
