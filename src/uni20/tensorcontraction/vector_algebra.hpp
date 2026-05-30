#pragma once

#include <uni20/tensorcontraction/matrix_family.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace uni20::tensorcontraction
{

inline void validate_compatible_vector_shapes(MatrixFamily const& lhs, MatrixFamily const& rhs)
{
  if (lhs.blocks().size() != rhs.blocks().size())
  {
    throw std::invalid_argument("TensorContraction vector operation has mismatched block counts");
  }

  for (std::size_t i = 0; i < lhs.blocks().size(); ++i)
  {
    if (lhs.block(i) != rhs.block(i))
    {
      throw std::invalid_argument("TensorContraction vector operation has mismatched block shapes");
    }
  }
}

inline auto make_like(MatrixFamily const& source) -> MatrixFamily { return MatrixFamily(source.blocks()); }

inline void zero(MatrixFamily& x) { x.fill(0.0); }

inline void copy(MatrixFamily const& source, MatrixFamily& target) { target.assign(source); }

inline double dot(MatrixFamily const& lhs, MatrixFamily const& rhs)
{
  validate_compatible_vector_shapes(lhs, rhs);

  double result = 0.0;
  for (std::size_t block = 0; block < lhs.blocks().size(); ++block)
  {
    auto const lhs_values = lhs.values(block);
    auto const rhs_values = rhs.values(block);
    for (std::size_t i = 0; i < lhs_values.size(); ++i)
    {
      result += lhs_values[i] * rhs_values[i];
    }
  }
  return result;
}

inline double norm2(MatrixFamily const& x) { return dot(x, x); }

inline double norm(MatrixFamily const& x) { return std::sqrt(norm2(x)); }

inline void scale(MatrixFamily& x, double alpha)
{
  for (std::size_t block = 0; block < x.blocks().size(); ++block)
  {
    auto values = x.values(block);
    for (double& value : values)
    {
      value *= alpha;
    }
  }
}

inline void axpy(double alpha, MatrixFamily const& x, MatrixFamily& y)
{
  validate_compatible_vector_shapes(x, y);

  for (std::size_t block = 0; block < x.blocks().size(); ++block)
  {
    auto const x_values = x.values(block);
    auto y_values = y.values(block);
    for (std::size_t i = 0; i < x_values.size(); ++i)
    {
      y_values[i] += alpha * x_values[i];
    }
  }
}

inline double normalize(MatrixFamily& x)
{
  double const x_norm = norm(x);
  if (x_norm == 0.0)
  {
    throw std::invalid_argument("TensorContraction cannot normalize a zero vector");
  }
  scale(x, 1.0 / x_norm);
  return x_norm;
}

} // namespace uni20::tensorcontraction
