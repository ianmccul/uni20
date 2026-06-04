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
    bool upload_from_host = true;
    bool sync_results_to_host = false;

    Impl();

    [[nodiscard]] bool host_backend() const { return arranger == nullptr; }

    void localize(MatrixFamily& x, bool upload_from_host, bool refresh_existing)
    {
      // MatrixFamily remains the public host-visible container.  The
      // TensorContraction runtime uses pre-store buffers as the current GPU
      // resident representation for Lanczos-local vector algebra.
      if (arranger != nullptr)
      {
        arranger->localizeForLinearAlgebra(raw_matrices(x), upload_from_host, refresh_existing);
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

  if (!impl_->upload_from_host)
  {
    // Resident mode assumes the latest vector data is already in pre-store
    // buffers.  localize(..., false) only guarantees GPU allocation; it does
    // not overwrite resident data from stale host storage.
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), false);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), true, false);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), true, false);
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

void VectorAlgebraEngine::set_host_synchronization(bool enabled)
{
  impl_->upload_from_host = enabled;
  impl_->sync_results_to_host = enabled;
}

void VectorAlgebraEngine::localize(MatrixFamily& x) { impl_->localize(x, false, false); }

void VectorAlgebraEngine::upload(MatrixFamily& x) { impl_->localize(x, true, true); }

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
  impl_->localize(x, false, false);
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger->compileZeroForLinearAlgebra(matrix, impl_->sync_results_to_host);
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

  if (!impl_->upload_from_host)
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(source), false);
    impl_->localize(target, false, false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(source), true, false);
    impl_->localize(target, false, false);
  }

  double one = 1.0;
  auto const& source_matrices = raw_matrices(source);
  auto const& target_matrices = raw_matrices(target);
  for (std::size_t block = 0; block < source_matrices.size(); ++block)
  {
    impl_->arranger->compileZeroForLinearAlgebra(target_matrices[block], impl_->sync_results_to_host);
    impl_->arranger->compileAddAccuForLinearAlgebra(target_matrices[block], source_matrices[block], &one,
                                                    impl_->sync_results_to_host);
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
  if (!impl_->upload_from_host)
  {
    impl_->localize(x, false, false);
  }
  else
  {
    impl_->localize(x, true, false);
  }
  for (auto const& matrix : raw_matrices(x))
  {
    impl_->arranger->compileScalarMulForLinearAlgebra(matrix, &alpha, impl_->sync_results_to_host);
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

  if (!impl_->upload_from_host)
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(x), false);
    impl_->localize(y, false, false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(x), true, false);
    impl_->localize(y, true, false);
  }

  auto const& x_matrices = raw_matrices(x);
  auto const& y_matrices = raw_matrices(y);
  for (std::size_t block = 0; block < x_matrices.size(); ++block)
  {
    impl_->arranger->compileAddAccuForLinearAlgebra(y_matrices[block], x_matrices[block], &alpha,
                                                    impl_->sync_results_to_host);
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

  if (!impl_->upload_from_host)
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), false);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), false);
    impl_->localize(result, false, false);
  }
  else
  {
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), true, false);
    impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), true, false);
    impl_->localize(result, false, false);
  }

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  auto const& result_matrices = raw_matrices(result);
  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[block], lhs_matrices[block], rhs_matrices[block],
                                                   impl_->sync_results_to_host);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_each_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs, MatrixFamily& result)
{
  validate_compatible_gemm_shapes(lhs, rhs, result);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::gemm_each(lhs, rhs, result);
    return;
  }

  impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), true);
  impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), true);
  impl_->localize(result, false, false);

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  auto const& result_matrices = raw_matrices(result);
  for (std::size_t block = 0; block < lhs_matrices.size(); ++block)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[block], lhs_matrices[block], rhs_matrices[block],
                                                   false);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_selected_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs,
                                                    MatrixFamily& result,
                                                    std::span<std::size_t const> lhs_block_for_result,
                                                    std::span<std::size_t const> rhs_block_for_result)
{
  validate_compatible_selected_gemm_shapes(lhs, rhs, result, lhs_block_for_result, rhs_block_for_result);
  if (impl_->host_backend())
  {
    uni20::tensorcontraction::gemm_selected(lhs, rhs, result, lhs_block_for_result, rhs_block_for_result);
    return;
  }

  impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), true);
  impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), true);
  impl_->localize(result, false, false);

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  auto const& result_matrices = raw_matrices(result);
  for (std::size_t block = 0; block < result_matrices.size(); ++block)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[block], lhs_matrices[lhs_block_for_result[block]],
                                                   rhs_matrices[rhs_block_for_result[block]], false);
  }
  impl_->arranger->doLinearAlgebra();
}

void VectorAlgebraEngine::gemm_sparse_selected_to_resident(MatrixFamily const& lhs, MatrixFamily const& rhs,
                                                           MatrixFamily& result,
                                                           std::span<std::size_t const> lhs_block_for_product,
                                                           std::span<std::size_t const> rhs_block_for_product,
                                                           std::span<std::size_t const> result_block_for_product)
{
  validate_compatible_sparse_selected_gemm_shapes(lhs, rhs, result, lhs_block_for_product, rhs_block_for_product,
                                                  result_block_for_product);

  std::vector<bool> result_seen(result.size(), false);
  for (std::size_t result_index : result_block_for_product)
  {
    if (result_seen[result_index])
    {
      throw std::invalid_argument(
          "TensorContraction sparse selected GEMM does not yet support duplicate result blocks");
    }
    result_seen[result_index] = true;
  }

  if (impl_->host_backend())
  {
    uni20::tensorcontraction::zero(result);
    for (std::size_t product = 0; product < result_block_for_product.size(); ++product)
    {
      auto const lhs_index = lhs_block_for_product[product];
      auto const rhs_index = rhs_block_for_product[product];
      auto const result_index = result_block_for_product[product];
      auto const lhs_block = lhs.block(lhs_index);
      auto const rhs_block = rhs.block(rhs_index);
      auto const result_block = result.block(result_index);
      auto const lhs_values = lhs.values(lhs_index);
      auto const rhs_values = rhs.values(rhs_index);
      auto result_values = result.values(result_index);
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
    return;
  }

  impl_->arranger->localizeForLinearAlgebra(raw_matrices(lhs), true);
  impl_->arranger->localizeForLinearAlgebra(raw_matrices(rhs), true);
  impl_->localize(result, false, false);

  auto const& result_matrices = raw_matrices(result);
  for (auto const& matrix : result_matrices)
  {
    impl_->arranger->compileZeroForLinearAlgebra(matrix, false);
  }

  auto const& lhs_matrices = raw_matrices(lhs);
  auto const& rhs_matrices = raw_matrices(rhs);
  for (std::size_t product = 0; product < result_block_for_product.size(); ++product)
  {
    impl_->arranger->compileMatMulForLinearAlgebra(result_matrices[result_block_for_product[product]],
                                                   lhs_matrices[lhs_block_for_product[product]],
                                                   rhs_matrices[rhs_block_for_product[product]], false);
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

bool VectorAlgebraEngine::uses_host_backend() const { return impl_->host_backend(); }

tensor::Arranger& VectorAlgebraEngine::resident_arranger()
{
  if (impl_->arranger == nullptr)
  {
    throw std::logic_error("TensorContraction resident runtime is not available on the host backend");
  }
  return *impl_->arranger;
}

} // namespace uni20::tensorcontraction
