#pragma once

#include <uni20/tensorcontraction/matrix_family.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>

namespace tensor
{
class Arranger;
}

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

inline void validate_compatible_gemm_shapes(MatrixFamily const& lhs, MatrixFamily const& rhs,
                                            MatrixFamily const& result)
{
  if (lhs.size() != rhs.size() || lhs.size() != result.size())
  {
    throw std::invalid_argument("TensorContraction batched GEMM has mismatched block counts");
  }

  for (std::size_t block = 0; block < lhs.size(); ++block)
  {
    auto const lhs_block = lhs.block(block);
    auto const rhs_block = rhs.block(block);
    auto const result_block = result.block(block);
    if (lhs_block.cols != rhs_block.rows || result_block.rows != lhs_block.rows || result_block.cols != rhs_block.cols)
    {
      throw std::invalid_argument("TensorContraction batched GEMM has incompatible block shapes");
    }
  }
}

inline void validate_compatible_selected_gemm_shapes(MatrixFamily const& lhs, MatrixFamily const& rhs,
                                                     MatrixFamily const& result,
                                                     std::span<std::size_t const> lhs_block_for_result,
                                                     std::span<std::size_t const> rhs_block_for_result)
{
  if (lhs_block_for_result.size() != result.size() || rhs_block_for_result.size() != result.size())
  {
    throw std::invalid_argument("TensorContraction selected GEMM has mismatched selector counts");
  }

  for (std::size_t block = 0; block < result.size(); ++block)
  {
    auto const lhs_index = lhs_block_for_result[block];
    auto const rhs_index = rhs_block_for_result[block];
    if (lhs_index >= lhs.size() || rhs_index >= rhs.size())
    {
      throw std::invalid_argument("TensorContraction selected GEMM selector is out of range");
    }

    auto const lhs_block = lhs.block(lhs_index);
    auto const rhs_block = rhs.block(rhs_index);
    auto const result_block = result.block(block);
    if (lhs_block.cols != rhs_block.rows || result_block.rows != lhs_block.rows || result_block.cols != rhs_block.cols)
    {
      throw std::invalid_argument("TensorContraction selected GEMM has incompatible block shapes");
    }
  }
}

inline void validate_compatible_sparse_selected_gemm_shapes(MatrixFamily const& lhs, MatrixFamily const& rhs,
                                                            MatrixFamily const& result,
                                                            std::span<std::size_t const> lhs_block_for_product,
                                                            std::span<std::size_t const> rhs_block_for_product,
                                                            std::span<std::size_t const> result_block_for_product)
{
  if (lhs_block_for_product.size() != rhs_block_for_product.size() ||
      lhs_block_for_product.size() != result_block_for_product.size())
  {
    throw std::invalid_argument("TensorContraction sparse selected GEMM has mismatched selector counts");
  }

  for (std::size_t product = 0; product < result_block_for_product.size(); ++product)
  {
    auto const lhs_index = lhs_block_for_product[product];
    auto const rhs_index = rhs_block_for_product[product];
    auto const result_index = result_block_for_product[product];
    if (lhs_index >= lhs.size() || rhs_index >= rhs.size() || result_index >= result.size())
    {
      throw std::invalid_argument("TensorContraction sparse selected GEMM selector is out of range");
    }

    auto const lhs_block = lhs.block(lhs_index);
    auto const rhs_block = rhs.block(rhs_index);
    auto const result_block = result.block(result_index);
    if (lhs_block.cols != rhs_block.rows || result_block.rows != lhs_block.rows || result_block.cols != rhs_block.cols)
    {
      throw std::invalid_argument("TensorContraction sparse selected GEMM has incompatible block shapes");
    }
  }
}

inline void gemm_each(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result)
{
  validate_compatible_gemm_shapes(lhs, rhs, result);

  for (std::size_t block = 0; block < lhs.size(); ++block)
  {
    auto const lhs_block = lhs.block(block);
    auto const rhs_block = rhs.block(block);
    auto const result_block = result.block(block);
    auto const lhs_values = lhs.values(block);
    auto const rhs_values = rhs.values(block);
    auto result_values = result.values(block);
    std::fill(result_values.begin(), result_values.end(), 0.0);

    for (std::size_t row = 0; row < lhs_block.rows; ++row)
    {
      for (std::size_t inner = 0; inner < lhs_block.cols; ++inner)
      {
        auto const lhs_value = lhs_values[row * lhs_block.cols + inner];
        for (std::size_t col = 0; col < rhs_block.cols; ++col)
        {
          result_values[row * result_block.cols + col] += lhs_value * rhs_values[inner * rhs_block.cols + col];
        }
      }
    }
  }
}

inline void gemm_selected(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result,
                          std::span<std::size_t const> lhs_block_for_result,
                          std::span<std::size_t const> rhs_block_for_result)
{
  validate_compatible_selected_gemm_shapes(lhs, rhs, result, lhs_block_for_result, rhs_block_for_result);

  for (std::size_t block = 0; block < result.size(); ++block)
  {
    auto const lhs_index = lhs_block_for_result[block];
    auto const rhs_index = rhs_block_for_result[block];
    auto const lhs_block = lhs.block(lhs_index);
    auto const rhs_block = rhs.block(rhs_index);
    auto const result_block = result.block(block);
    auto const lhs_values = lhs.values(lhs_index);
    auto const rhs_values = rhs.values(rhs_index);
    auto result_values = result.values(block);
    std::fill(result_values.begin(), result_values.end(), 0.0);

    for (std::size_t row = 0; row < lhs_block.rows; ++row)
    {
      for (std::size_t inner = 0; inner < lhs_block.cols; ++inner)
      {
        auto const lhs_value = lhs_values[row * lhs_block.cols + inner];
        for (std::size_t col = 0; col < rhs_block.cols; ++col)
        {
          result_values[row * result_block.cols + col] += lhs_value * rhs_values[inner * rhs_block.cols + col];
        }
      }
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

class VectorAlgebraEngine {
  public:
    VectorAlgebraEngine();
    VectorAlgebraEngine(VectorAlgebraEngine const&) = delete;
    VectorAlgebraEngine& operator=(VectorAlgebraEngine const&) = delete;
    VectorAlgebraEngine(VectorAlgebraEngine&&) noexcept;
    VectorAlgebraEngine& operator=(VectorAlgebraEngine&&) noexcept;
    ~VectorAlgebraEngine();

    [[nodiscard]] double dot(MatrixFamily const& lhs, MatrixFamily const& rhs);
    [[nodiscard]] double norm2(MatrixFamily const& x);
    [[nodiscard]] double norm(MatrixFamily const& x);
    void set_host_synchronization(bool enabled);
    void localize(MatrixFamily& x);
    void upload(MatrixFamily& x);
    void synchronize(MatrixFamily& x);
    void release(MatrixFamily const& x) noexcept;
    void zero(MatrixFamily& x);
    void copy(MatrixFamily const& source, MatrixFamily& target);
    void scale(MatrixFamily& x, double alpha);
    void axpy(double alpha, MatrixFamily const& x, MatrixFamily& y);
    void gemm_each(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result);
    void gemm_each_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result);
    void gemm_selected_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result,
                                   std::span<std::size_t const> lhs_block_for_result,
                                   std::span<std::size_t const> rhs_block_for_result);
    void gemm_sparse_selected_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result,
                                          std::span<std::size_t const> lhs_block_for_product,
                                          std::span<std::size_t const> rhs_block_for_product,
                                          std::span<std::size_t const> result_block_for_product);
    [[nodiscard]] double normalize(MatrixFamily& x);

    [[nodiscard]] bool uses_host_backend() const;
    [[nodiscard]] tensor::Arranger& resident_arranger();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace uni20::tensorcontraction
