#include "ctranslate2/ops/matmul.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    MatMul::MatMul(bool trans_a, bool trans_b, float alpha)
      : _trans_a(trans_a)
      , _trans_b(trans_b)
      , _alpha(alpha) {
    }

    void MatMul::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("MatMul");
      DEVICE_AND_FLOAT_DISPATCH("MatMul", a.device(), a.dtype(), (compute<D, T>(a, b, c, b.dim(-2))));
    }

    void MatMul::operator()(const StorageView& a,
                            const StorageView& b,
                            StorageView& c,
                            dim_t b_rows) const {
      PROFILE("MatMul");
      if (b_rows < 0 || b_rows > b.dim(-2))
        throw std::invalid_argument("MatMul: cannot use " + std::to_string(b_rows)
                                    + " rows of matrices with "
                                    + std::to_string(b.dim(-2)) + " rows");
      DEVICE_AND_FLOAT_DISPATCH("MatMul", a.device(), a.dtype(), (compute<D, T>(a, b, c, b_rows)));
    }

    template <Device D, typename T>
    void MatMul::compute(const StorageView& a,
                         const StorageView& b,
                         StorageView& c,
                         dim_t b_rows) const {
      dim_t m, k_a;
      if (_trans_a) {
        m = a.dim(-1);
        k_a = a.dim(-2);
      } else {
        m = a.dim(-2);
        k_a = a.dim(-1);
      }

      // The matrices of b are stored with b.dim(-2) rows, of which the first b_rows are used.
      dim_t k_b, n;
      if (_trans_b) {
        n = b_rows;
        k_b = b.dim(-1);
      } else {
        n = b.dim(-1);
        k_b = b_rows;
      }

      if (k_a != k_b)
        throw std::invalid_argument("MatMul: k dimension of inputs a and b should match");

      const dim_t k = k_a;
      const dim_t b_matrix_size = b.dim(-2) * b.dim(-1);
      const dim_t a_batch_size = a.size() / (m * k);
      const dim_t b_batch_size = b_matrix_size > 0 ? b.size() / b_matrix_size : 0;

      if (a_batch_size != b_batch_size)
        throw std::invalid_argument("MatMul: batch dimension of inputs a and b should match");

      {
        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 1] = n;
        output_shape[output_shape.size() - 2] = m;
        c.resize(std::move(output_shape));
      }

      const dim_t batch_size = a_batch_size;
      const dim_t lda = _trans_a ? m : k;
      const dim_t ldb = b.dim(-1);
      const dim_t ldc = n;
      const float beta = 0;

      if (batch_size > 1) {
        const dim_t stridea = m * k;
        const dim_t strideb = b_matrix_size;
        const dim_t stridec = m * n;
        primitives<D>::gemm_batch_strided(_trans_a, _trans_b,
                                          m, n, k,
                                          _alpha,
                                          a.data<T>(), lda, stridea,
                                          b.data<T>(), ldb, strideb,
                                          beta,
                                          c.data<T>(), ldc, stridec,
                                          batch_size);
      } else {
        primitives<D>::gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
                            _trans_a, _trans_b,
                            m, n, k,
                            _alpha,
                            a.data<T>(), lda,
                            b.data<T>(), ldb,
                            beta,
                            c.data<T>(), ldc);
      }
    }

  }
}
