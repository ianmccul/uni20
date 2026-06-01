#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20::tensorcontraction
{

struct VectorAlgebraEngine::Impl
{
    std::unique_ptr<tensor::Swapper> swapper;
    std::unique_ptr<tensor::Arranger> arranger;
    bool sync_host = true;

    Impl();

    [[nodiscard]] bool host_backend() const { return arranger == nullptr; }

    void localize(MatrixFamily& x, bool upload_from_host)
    {
      // MatrixFamily remains the public host-visible container.  The
      // TensorContraction runtime uses pre-store buffers as the current GPU
      // resident representation for Lanczos-local vector algebra.
      if (arranger != nullptr)
      {
        arranger->localizeForLinearAlgebra(raw_matrices(x), upload_from_host);
      }
    }
};

namespace
{

double broadcast_from_rank_zero(double value)
{
  // The current TensorContraction runtime is lockstep MPI.  Scalar Krylov
  // decisions must be identical on every rank, so rank 0 is the authority for
  // reductions exposed back to the algorithm layer.
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0)
  {
    return value;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size > 1)
  {
    MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  return value;
}

bool use_host_vector_algebra_backend()
{
  auto const* backend = std::getenv("UNI20_TENSORCONTRACTION_BACKEND");
  if (backend == nullptr)
  {
    return false;
  }
  return std::string(backend) == "host" || std::string(backend) == "cpu";
}

} // namespace

VectorAlgebraEngine::Impl::Impl()
{
  if (use_host_vector_algebra_backend())
  {
    return;
  }
  swapper = std::make_unique<tensor::Swapper>();
  arranger = std::make_unique<tensor::Arranger>(*swapper);
}

VectorAlgebraEngine::VectorAlgebraEngine() : impl_(std::make_unique<Impl>()) {}

VectorAlgebraEngine::VectorAlgebraEngine(VectorAlgebraEngine&&) noexcept = default;
VectorAlgebraEngine& VectorAlgebraEngine::operator=(VectorAlgebraEngine&&) noexcept = default;
VectorAlgebraEngine::~VectorAlgebraEngine() = default;

double VectorAlgebraEngine::dot(MatrixFamily const& lhs, MatrixFamily const& rhs)
{
  validate_compatible_vector_shapes(lhs, rhs);
  if (impl_->host_backend())
  {
    return uni20::tensorcontraction::dot(lhs, rhs);
  }

  if (!impl_->sync_host)
  {
    // Resident mode assumes the latest vector data is already in pre-store
    // buffers.  localize(..., false) only guarantees GPU allocation; it does
    // not overwrite resident data from stale host storage.
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), false);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), true);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), true);
  }

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  std::vector<double> block_results(lhs_matrices.size(), 0.0);
  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    impl_->arranger->compileInnerProductForLinearAlgebra(lhs_matrices[block], rhs_matrices[block],
                                                         &block_results[block]);
  }
  impl_->arranger->doLinearAlgebra();

  double result = 0.0;
  for (double value : block_results)
  {
    result += value;
  }
  return broadcast_from_rank_zero(result);
}

double VectorAlgebraEngine::norm2(MatrixFamily const& x) { return this->dot(x, x); }

double VectorAlgebraEngine::norm(MatrixFamily const& x) { return std::sqrt(this->norm2(x)); }

void VectorAlgebraEngine::set_host_synchronization(bool enabled) { impl_->sync_host = enabled; }

void VectorAlgebraEngine::localize(MatrixFamily& x) { impl_->localize(x, false); }

void VectorAlgebraEngine::upload(MatrixFamily& x) { impl_->localize(x, true); }

void VectorAlgebraEngine::synchronize(MatrixFamily& x)
{
  if (impl_->host_backend())
  {
    return;
  }
  // Explicit escape hatch for boundaries that still consume host storage, such
  // as the current EffectiveHamiltonianOperator adapter.
  impl_->arranger->synchronizeLinearAlgebraToHost(raw_matrices(x));
}

void VectorAlgebraEngine::zero(MatrixFamily& x)
{
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::zero(x);
    return;
  }
  if (!impl_->sync_host)
  {
    impl_->localize(x, false);
  }
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger->compileZeroForLinearAlgebra(matrix, impl_->sync_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::copy(MatrixFamily const& source, MatrixFamily& target)
{
  validate_compatible_vector_shapes(source, target);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::copy(source, target);
    return;
  }

  if (!impl_->sync_host)
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(source), false);
    impl_->localize(target, false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(source), true);
  }

  double one = 1.0;
  auto const& source_matrices = raw_matrices(source);
  auto const& target_matrices = raw_matrices(target);
  for (std::size_t block = 0; block < source_matrices.size(); ++block)
  {
    impl_->arranger->compileZeroForLinearAlgebra(target_matrices[block], impl_->sync_host);
    impl_->arranger->compileAddAccuForLinearAlgebra(target_matrices[block], source_matrices[block], &one,
                                                    impl_->sync_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::scale(MatrixFamily& x, double alpha)
{
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::scale(x, alpha);
    return;
  }
  if (!impl_->sync_host)
  {
    impl_->localize(x, false);
  }
  else
  {
    impl_->localize(x, true);
  }
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger->compileScalarMulForLinearAlgebra(matrix, &alpha, impl_->sync_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::axpy(double alpha, MatrixFamily const& x, MatrixFamily& y)
{
  validate_compatible_vector_shapes(x, y);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::axpy(alpha, x, y);
    return;
  }

  if (!impl_->sync_host)
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(x), false);
    impl_->localize(y, false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(x), true);
    impl_->localize(y, true);
  }

  auto const& x_matrices = raw_matrices(x);
  auto const& y_matrices = raw_matrices(y);
  for (std::size_t block = 0; block < x_matrices.size(); ++block)
  {
    impl_->arranger->compileAddAccuForLinearAlgebra(y_matrices[block], x_matrices[block], &alpha, impl_->sync_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_each(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result)
{
  validate_compatible_gemm_shapes(lhs, rhs, result);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::gemm_each(lhs, rhs, result);
    return;
  }

  if (!impl_->sync_host)
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), false);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), false);
    impl_->localize(result, false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), true);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), true);
  }

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  auto const& result_matrices = raw_matrices(result);
  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[block], lhs_matrices[block], rhs_matrices[block],
                                                   impl_->sync_host);
  }
  impl_->arranger->doLinearAlgebra();
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
