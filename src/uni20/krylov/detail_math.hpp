#pragma once

#include <cmath>

namespace uni20::krylov::detail
{

template <typename T> auto adl_abs(T const& value)
{
  using std::abs;
  return abs(value);
}

template <typename T> auto adl_sqrt(T const& value)
{
  using std::sqrt;
  return sqrt(value);
}

template <typename T> auto adl_pow(T const& value, T const& exponent)
{
  using std::pow;
  return pow(value, exponent);
}

template <typename T> auto adl_real(T const& value)
{
  using std::real;
  return real(value);
}

template <typename T> auto adl_imag(T const& value)
{
  using std::imag;
  return imag(value);
}

template <typename T> bool adl_isfinite(T const& value)
{
  using std::isfinite;
  return isfinite(value);
}

template <typename T> auto abs_squared(T const& value)
{
  auto const magnitude = adl_abs(value);
  return magnitude * magnitude;
}

} // namespace uni20::krylov::detail
