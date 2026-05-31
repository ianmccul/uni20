#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

struct VectorAlgebraEngine::Impl
{
    tensor::Swapper swapper;
    tensor::Arranger arranger;

    Impl() : swapper(), arranger(swapper) {}
};

VectorAlgebraEngine::VectorAlgebraEngine() : impl_(std::make_unique<Impl>()) {}

VectorAlgebraEngine::VectorAlgebraEngine(VectorAlgebraEngine&&) noexcept = default;
VectorAlgebraEngine& VectorAlgebraEngine::operator=(VectorAlgebraEngine&&) noexcept = default;
VectorAlgebraEngine::~VectorAlgebraEngine() = default;

double VectorAlgebraEngine::dot(MatrixFamily const& lhs, MatrixFamily const& rhs)
{
  validate_compatible_vector_shapes(lhs, rhs);

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  std::vector<double> block_results(lhs_matrices.size(), 0.0);
  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    impl_->arranger.compileInnerProductForLinearAlgebra(lhs_matrices[block], rhs_matrices[block],
                                                        &block_results[block]);
  }
  impl_->arranger.doLinearAlgebra();

  double result = 0.0;
  for (double value : block_results)
  {
    result += value;
  }
  return result;
}

double VectorAlgebraEngine::norm2(MatrixFamily const& x) { return this->dot(x, x); }

double VectorAlgebraEngine::norm(MatrixFamily const& x) { return std::sqrt(this->norm2(x)); }

void VectorAlgebraEngine::zero(MatrixFamily& x)
{
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger.compileZeroForLinearAlgebra(matrix);
  }
  impl_->arranger.doLinearAlgebra();
}

void VectorAlgebraEngine::copy(MatrixFamily const& source, MatrixFamily& target)
{
  validate_compatible_vector_shapes(source, target);

  double one = 1.0;
  auto const& source_matrices = raw_matrices(source);
  auto const& target_matrices = raw_matrices(target);
  for (std::size_t block = 0; block < source_matrices.size(); ++block)
  {
    impl_->arranger.compileZeroForLinearAlgebra(target_matrices[block]);
    impl_->arranger.compileAddAccuForLinearAlgebra(target_matrices[block], source_matrices[block], &one);
  }
  impl_->arranger.doLinearAlgebra();
}

void VectorAlgebraEngine::scale(MatrixFamily& x, double alpha)
{
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger.compileScalarMulForLinearAlgebra(matrix, &alpha);
  }
  impl_->arranger.doLinearAlgebra();
}

void VectorAlgebraEngine::axpy(double alpha, MatrixFamily const& x, MatrixFamily& y)
{
  validate_compatible_vector_shapes(x, y);

  auto const& x_matrices = raw_matrices(x);
  auto const& y_matrices = raw_matrices(y);
  for (std::size_t block = 0; block < x_matrices.size(); ++block)
  {
    impl_->arranger.compileAddAccuForLinearAlgebra(y_matrices[block], x_matrices[block], &alpha);
  }
  impl_->arranger.doLinearAlgebra();
}

double VectorAlgebraEngine::normalize(MatrixFamily& x)
{
  double const x_norm = this->norm(x);
  if (x_norm == 0.0)
  {
    throw std::invalid_argument("TensorContraction cannot normalize a zero vector");
  }
  this->scale(x, 1.0 / x_norm);
  return x_norm;
}

} // namespace uni20::tensorcontraction
