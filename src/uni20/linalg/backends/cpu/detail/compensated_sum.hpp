#pragma once

#include <uni20/core/math.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>

#include <cmath>

namespace uni20::linalg::backends::cpu::detail
{

/// \brief Same-precision Neumaier compensated sum for one real component.
template <uni20::Real Real> class CompensatedRealSum {
  public:
    void add(Real value)
    {
      using std::abs;

      Real const next = sum_ + value;
      if (!uni20::isfinite(next))
      {
        sum_ = next;
        correction_ = Real{};
        return;
      }
      if (abs(sum_) >= abs(value))
      {
        correction_ += (sum_ - next) + value;
      }
      else
      {
        correction_ += (value - next) + sum_;
      }
      sum_ = next;
    }

    [[nodiscard]] Real value() const { return sum_ + correction_; }

  private:
    Real sum_{};
    Real correction_{};
};

/// \brief Component-wise compensated sum that preserves the input scalar field.
template <uni20::RealOrComplex Scalar> class CompensatedSum {
  public:
    using real_type = uni20::make_real_t<Scalar>;

    void add(Scalar value)
    {
      if constexpr (uni20::Complex<Scalar>)
      {
        real_.add(value.real());
        imaginary_.add(value.imag());
      }
      else
      {
        real_.add(value);
      }
    }

    [[nodiscard]] Scalar value() const
    {
      if constexpr (uni20::Complex<Scalar>)
      {
        return Scalar{real_.value(), imaginary_.value()};
      }
      else
      {
        return real_.value();
      }
    }

  private:
    CompensatedRealSum<real_type> real_;
    CompensatedRealSum<real_type> imaginary_;
};

} // namespace uni20::linalg::backends::cpu::detail
