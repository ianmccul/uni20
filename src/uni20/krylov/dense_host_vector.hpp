#pragma once

#include <uni20/krylov/matrix_free.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20::krylov
{

/// \brief Prototype host vector used by matrix-free Krylov tests and adapters.
template <typename Scalar> struct DenseHostVector
{
    std::vector<Scalar> values;
};

/// \brief Dense host matrix-free operator and vector operations for prototypes.
template <typename Scalar> class DenseHostVectorOps {
  public:
    DenseHostVectorOps(std::size_t dimension, std::vector<Scalar> matrix)
        : dimension_(dimension), matrix_(std::move(matrix))
    {
      if (matrix_.size() != dimension_ * dimension_)
      {
        throw std::invalid_argument("dense Krylov operator matrix has the wrong size");
      }
    }

    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t problem_dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t vector_dimension(DenseHostVector<Scalar> const& x) const noexcept
    {
      return x.values.size();
    }

    [[nodiscard]] int allocation_count() const noexcept { return allocation_count_; }
    [[nodiscard]] int copy_count() const noexcept { return copy_count_; }
    [[nodiscard]] int axpy_count() const noexcept { return axpy_count_; }
    [[nodiscard]] int scal_count() const noexcept { return scal_count_; }
    [[nodiscard]] int set_zero_count() const noexcept { return set_zero_count_; }
    [[nodiscard]] int inner_product_count() const noexcept { return inner_product_count_; }
    [[nodiscard]] int norm_count() const noexcept { return norm_count_; }
    [[nodiscard]] int matvec_count() const noexcept { return matvec_count_; }

    [[nodiscard]] DenseHostVector<Scalar> allocate_like(DenseHostVector<Scalar> const& x)
    {
      ++allocation_count_;
      return DenseHostVector<Scalar>{std::vector<Scalar>(x.values.size())};
    }

    void copy(DenseHostVector<Scalar>& dst, DenseHostVector<Scalar> const& src)
    {
      require_same_size(dst, src);
      ++copy_count_;
      dst.values = src.values;
    }

    void axpy(DenseHostVector<Scalar>& y, Scalar alpha, DenseHostVector<Scalar> const& x)
    {
      require_same_size(y, x);
      ++axpy_count_;
      for (std::size_t i = 0; i < y.values.size(); ++i)
      {
        y.values[i] += alpha * x.values[i];
      }
    }

    void scal(DenseHostVector<Scalar>& x, Scalar alpha)
    {
      ++scal_count_;
      for (auto& value : x.values)
      {
        value *= alpha;
      }
    }

    void set_zero(DenseHostVector<Scalar>& x)
    {
      ++set_zero_count_;
      for (auto& value : x.values)
      {
        value = Scalar{};
      }
    }

    [[nodiscard]] Scalar inner_product(DenseHostVector<Scalar> const& x, DenseHostVector<Scalar> const& y)
    {
      require_same_size(x, y);
      ++inner_product_count_;

      Scalar result{};
      for (std::size_t i = 0; i < x.values.size(); ++i)
      {
        result += conjugate_for_inner_product(x.values[i]) * y.values[i];
      }
      return result;
    }

    [[nodiscard]] uni20::make_real_t<Scalar> norm(DenseHostVector<Scalar> const& x)
    {
      ++norm_count_;

      uni20::make_real_t<Scalar> norm_squared{};
      for (auto const& value : x.values)
      {
        norm_squared += detail::abs_squared(value);
      }
      return norm_squared > uni20::make_real_t<Scalar>{} ? detail::adl_sqrt(norm_squared)
                                                         : uni20::make_real_t<Scalar>{};
    }

    void matvec(DenseHostVector<Scalar>& y, DenseHostVector<Scalar> const& x)
    {
      if (x.values.size() != dimension_ || y.values.size() != dimension_)
      {
        throw std::invalid_argument("dense Krylov matvec vector has the wrong size");
      }

      ++matvec_count_;
      for (std::size_t row = 0; row < dimension_; ++row)
      {
        Scalar value{};
        for (std::size_t col = 0; col < dimension_; ++col)
        {
          value += matrix_[row * dimension_ + col] * x.values[col];
        }
        y.values[row] = value;
      }
    }

  private:
    static void require_same_size(DenseHostVector<Scalar> const& lhs, DenseHostVector<Scalar> const& rhs)
    {
      if (lhs.values.size() != rhs.values.size())
      {
        throw std::invalid_argument("dense Krylov vectors have different sizes");
      }
    }

    [[nodiscard]] static Scalar conjugate_for_inner_product(Scalar const& value)
    {
      if constexpr (uni20::Complex<Scalar>)
      {
        return std::conj(value);
      }
      else
      {
        return value;
      }
    }

    std::size_t dimension_ = 0;
    std::vector<Scalar> matrix_;
    int allocation_count_ = 0;
    int copy_count_ = 0;
    int axpy_count_ = 0;
    int scal_count_ = 0;
    int set_zero_count_ = 0;
    int inner_product_count_ = 0;
    int norm_count_ = 0;
    int matvec_count_ = 0;
};

static_assert(KrylovMatrixFreeOperator<DenseHostVectorOps<float>, DenseHostVector<float>, float>);
static_assert(KrylovMatrixFreeOperator<DenseHostVectorOps<double>, DenseHostVector<double>, double>);
static_assert(KrylovMatrixFreeOperator<DenseHostVectorOps<uni20::complex<float>>,
                                       DenseHostVector<uni20::complex<float>>, uni20::complex<float>>);
static_assert(KrylovMatrixFreeOperator<DenseHostVectorOps<uni20::complex<double>>,
                                       DenseHostVector<uni20::complex<double>>, uni20::complex<double>>);

} // namespace uni20::krylov
