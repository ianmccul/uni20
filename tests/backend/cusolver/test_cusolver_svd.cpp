#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/backend/cusolver/execution.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

namespace
{

class CusolverSvdTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess) GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    int device_count_ = 0;
};

struct CudaSvdOperands
{
    uni20::CudaMatrix<double> matrix;
    uni20::CudaTensor<double, 1> values;
    uni20::CudaMatrix<double> left;
    uni20::CudaMatrix<double> right_adjoint;
};

[[nodiscard]] CudaSvdOperands make_operands(uni20::cuda::DeviceResources& resources, double scale)
{
  uni20::ColumnMajorTensor<double, 2> host_matrix(3, 2);
  host_matrix[0, 0] = 3.0 * scale;
  host_matrix[1, 0] = 0.0;
  host_matrix[2, 0] = 0.0;
  host_matrix[0, 1] = 0.0;
  host_matrix[1, 1] = 2.0 * scale;
  host_matrix[2, 1] = 0.0;

  CudaSvdOperands result{.matrix = uni20::CudaMatrix<double>(resources, 3, 2),
                         .values = uni20::CudaTensor<double, 1>(resources, 2),
                         .left = uni20::CudaMatrix<double>(resources, 3, 2),
                         .right_adjoint = uni20::CudaMatrix<double>(resources, 2, 2)};
  uni20::copy(result.matrix, host_matrix);
  return result;
}

void expect_reconstruction(CudaSvdOperands const& operands, double scale)
{
  uni20::ColumnMajorTensor<double, 1> values(2);
  uni20::ColumnMajorTensor<double, 2> left(3, 2);
  uni20::ColumnMajorTensor<double, 2> right_adjoint(2, 2);
  uni20::copy(values, operands.values);
  uni20::copy(left, operands.left);
  uni20::copy(right_adjoint, operands.right_adjoint);

  EXPECT_NEAR(values[0], 3.0 * scale, 1.0e-12);
  EXPECT_NEAR(values[1], 2.0 * scale, 1.0e-12);
  for (uni20::index_type row = 0; row < 3; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      double reconstructed = 0.0;
      for (uni20::index_type inner = 0; inner < 2; ++inner)
        reconstructed += left[row, inner] * values[inner] * right_adjoint[inner, column];
      double const expected = row == column ? (row == 0 ? 3.0 : 2.0) * scale : 0.0;
      EXPECT_NEAR(reconstructed, expected, 1.0e-11);
    }
  }
}

TEST_F(CusolverSvdTest, DestructiveTensorSvdReconstructsTallMatrix)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  auto operands = make_operands(resources, 1.0);

  uni20::linalg::singular_value_decomposition(uni20::linalg::CusolverBackend{}, operands.values, operands.left,
                                              operands.right_adjoint, operands.matrix);

  expect_reconstruction(operands, 1.0);
}

TEST_F(CusolverSvdTest, CanonicalExecutionPoolDefaultsToTwoHandlesAndSupportsConfiguration)
{
  uni20::cuda::DeviceResources capped_resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  EXPECT_EQ(uni20::cusolver::execution_pool(capped_resources).handle_count(), 1);

  uni20::cuda::DeviceResources default_resources({.device = uni20::cuda::Device::get(0), .stream_count = 4});
  auto& default_pool = uni20::cusolver::execution_pool(default_resources);
  EXPECT_EQ(default_pool.handle_count(), 2);

  uni20::cuda::DeviceResources configured_resources({.device = uni20::cuda::Device::get(0), .stream_count = 4});
  auto& configured = uni20::cusolver::execution_pool(configured_resources, 3);
  auto& reused = uni20::cusolver::execution_pool(configured_resources);
  EXPECT_EQ(configured.handle_count(), 3);
  EXPECT_EQ(&reused, &configured);
}

TEST_F(CusolverSvdTest, SchedulerBatchRunsIndependentProviderCalls)
{
  constexpr std::size_t sector_count = 4;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = sector_count});
  auto& resources = runtime.device_resources(0);
  uni20::async::TbbScheduler scheduler{sector_count};
  uni20::async::ScopedScheduler scoped_scheduler(&scheduler);

  std::vector<CudaSvdOperands> sectors;
  sectors.reserve(sector_count);
  for (std::size_t sector = 0; sector < sector_count; ++sector)
    sectors.push_back(make_operands(resources, static_cast<double>(sector + 1)));

  uni20::async::execute_batch(sector_count, [&](std::size_t sector) {
    auto& operands = sectors[sector];
    uni20::linalg::singular_value_decomposition(uni20::linalg::CusolverBackend{}, operands.values, operands.left,
                                                operands.right_adjoint, operands.matrix);
  });

  auto& pool = uni20::cusolver::execution_pool(resources);
  EXPECT_EQ(pool.handle_count(), 2);
  for (std::size_t sector = 0; sector < sector_count; ++sector)
    expect_reconstruction(sectors[sector], static_cast<double>(sector + 1));
}

TEST_F(CusolverSvdTest, ParallelPackedBlockSvdDispatchesIndependentChargeSectors)
{
  using host_storage = uni20::PackedCompleteBlockStorage<>;
  using cuda_storage = uni20::ParallelPackedCompleteBlockStorage<uni20::CudaStorage>;
  using domain_type = uni20::Domain<uni20::BlockSpace>;
  using codomain_type = uni20::Codomain<uni20::BlockSpace>;
  using host_tensor = uni20::BlockTensor<double, domain_type, codomain_type, host_storage>;
  using cuda_tensor = uni20::BlockTensor<double, domain_type, codomain_type, cuda_storage>;

  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 4});
  uni20::async::TbbScheduler scheduler{4};
  uni20::async::ScopedScheduler scoped_scheduler(&scheduler);
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const domain_space(symmetry, {{q0, 2}, {q1, 2}}, "domain");
  uni20::BlockSpace const codomain_space(symmetry, {{q0, 3}, {q1, 4}}, "codomain");

  host_tensor host(symmetry, domain_type{domain_space}, codomain_type{codomain_space});
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
  {
    auto block = host.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = 0.0;
    double const scale = static_cast<double>(ordinal + 1);
    block[0, 0] = 3.0 * scale;
    block[1, 1] = 2.0 * scale;
  }

  cuda_tensor device(symmetry, domain_type{domain_space}, codomain_type{codomain_space});
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
  {
    auto device_block = device.block_by_ordinal(ordinal);
    auto host_block = host.block_by_ordinal(ordinal);
    uni20::copy(device_block, host_block);
  }
  auto decomposition = uni20::block_svd(std::as_const(device));

  ASSERT_EQ(decomposition.sectors().size(), 2U);
  ASSERT_EQ(decomposition.spectrum().size(), 4U);
  EXPECT_NEAR(decomposition.spectrum()[0].singular_value, 6.0, 1.0e-12);
  EXPECT_NEAR(decomposition.spectrum()[1].singular_value, 4.0, 1.0e-12);
  EXPECT_NEAR(decomposition.spectrum()[2].singular_value, 3.0, 1.0e-12);
  EXPECT_NEAR(decomposition.spectrum()[3].singular_value, 2.0, 1.0e-12);
}

} // namespace
