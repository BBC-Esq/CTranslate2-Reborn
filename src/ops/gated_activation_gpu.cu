#include "ctranslate2/ops/gated_activation.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    // The activation is rounded to T before the product, as when the activation op writes its
    // output and Mul reads it, so the results are identical.
    template <typename T, typename Activation>
    struct gated_activation_func {
      __device__ T operator()(const T gate, const T up) const {
        const T activated = Activation()(gate);
        return cuda::multiplies<T>()(up, activated);
      }
    };

    // Loads and stores 16 bytes at a time.
    template <typename T>
    struct alignas(16) GatedActivationVector {
      T values[16 / sizeof (T)];
    };

    template <typename T, typename Function>
    __global__ void gated_activation_kernel(const GatedActivationVector<T>* gate,
                                            const GatedActivationVector<T>* up,
                                            GatedActivationVector<T>* output,
                                            const cuda::index_t num_vectors) {
      const Function function;
      for (cuda::index_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < num_vectors;
           i += blockDim.x * gridDim.x) {
        const GatedActivationVector<T> g = gate[i];
        const GatedActivationVector<T> u = up[i];
        GatedActivationVector<T> y;
#pragma unroll
        for (int j = 0; j < int(16 / sizeof (T)); ++j)
          y.values[j] = function(g.values[j], u.values[j]);
        output[i] = y;
      }
    }

    template <typename T, template <typename> class Activation>
    static void gated_activation(const T* gate, const T* up, T* output, const dim_t size) {
      using DeviceT = cuda::device_type<T>;
      using Function = gated_activation_func<DeviceT, Activation<DeviceT>>;
      const auto aligned = [](const void* ptr) {
        return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
      };

      if ((size * sizeof (T)) % 16 == 0 && aligned(gate) && aligned(up) && aligned(output)) {
        using Vector = GatedActivationVector<DeviceT>;
        const dim_t num_vectors = size * sizeof (T) / 16;
        const dim_t threads = 256;
        const dim_t blocks = std::min(ceil_divide(num_vectors, threads), cuda::max_blocks);
        gated_activation_kernel<DeviceT, Function><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          reinterpret_cast<const Vector*>(gate),
          reinterpret_cast<const Vector*>(up),
          reinterpret_cast<Vector*>(output),
          num_vectors);
      } else {
        cuda::binary_transform(gate, up, output, size, Function());
      }
    }

    template <Device D, typename T>
    void GatedActivation::compute(const StorageView& gate,
                                  const StorageView& up,
                                  StorageView& output) const {
      output.resize_as(gate);
      const T* g = gate.data<T>();
      const T* u = up.data<T>();
      T* y = output.data<T>();
      const dim_t size = gate.size();

      switch (_activation_type) {
      case ActivationType::ReLU:
        gated_activation<T, cuda::relu_func>(g, u, y, size);
        break;
      case ActivationType::GELUTanh:
        gated_activation<T, cuda::gelu_tanh_func>(g, u, y, size);
        break;
      case ActivationType::Swish:
        gated_activation<T, cuda::swish_func>(g, u, y, size);
        break;
      case ActivationType::GELU:
        gated_activation<T, cuda::gelu_func>(g, u, y, size);
        break;
      case ActivationType::GELUSigmoid:
        gated_activation<T, cuda::gelu_sigmoid_func>(g, u, y, size);
        break;
      case ActivationType::Tanh:
        gated_activation<T, cuda::tanh_func>(g, u, y, size);
        break;
      case ActivationType::Sigmoid:
        gated_activation<T, cuda::sigmoid_func>(g, u, y, size);
        break;
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GatedActivation::compute<Device::CUDA, T>(const StorageView&,       \
                                              const StorageView&,       \
                                              StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
