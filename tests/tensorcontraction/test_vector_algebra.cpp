#include <uni20/tensorcontraction/vector_algebra.hpp>

#include "Arranger.hpp"
#include "Swapper.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace utc = uni20::tensorcontraction;

namespace
{

auto make_vector() -> utc::MatrixFamily
{
  std::array blocks{utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{1, 3}};
  utc::MatrixFamily x(blocks);
  x.assign(0, std::array{1.0, -2.0, 3.0, -4.0});
  x.assign(1, std::array{5.0, -6.0, 7.0});
  return x;
}

auto select_matrices(utc::MatrixFamily& family,
                     std::initializer_list<std::size_t> blocks) -> std::vector<tensor::Matrix>
{
  auto const& matrices = utc::raw_matrices(family);
  std::vector<tensor::Matrix> selected;
  selected.reserve(blocks.size());
  for (std::size_t block : blocks)
  {
    selected.push_back(matrices[block]);
  }
  return selected;
}

class EnvGuard {
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_;

  public:
    explicit EnvGuard(std::initializer_list<char const*> names)
    {
      saved_.reserve(names.size());
      for (auto const* name : names)
      {
        if (auto const* value = std::getenv(name); value != nullptr)
        {
          saved_.push_back({name, std::string(value)});
        }
        else
        {
          saved_.push_back({name, std::nullopt});
        }
      }
    }

    EnvGuard(EnvGuard const&) = delete;
    EnvGuard& operator=(EnvGuard const&) = delete;

    ~EnvGuard()
    {
      for (auto const& [name, value] : saved_)
      {
        if (value.has_value())
        {
          setenv(name.c_str(), value->c_str(), 1);
        }
        else
        {
          unsetenv(name.c_str());
        }
      }
    }
};

} // namespace

TEST(TensorContractionVectorAlgebraTest, ComputesDotAndNorm)
{
  auto x = make_vector();
  auto y = make_vector();
  y.assign(0, std::array{2.0, 3.0, -1.0, 0.5});
  y.assign(1, std::array{4.0, -2.0, 1.0});

  EXPECT_DOUBLE_EQ(utc::dot(x, y), 30.0);
  EXPECT_DOUBLE_EQ(utc::norm2(x), 140.0);
  EXPECT_DOUBLE_EQ(utc::norm(x), std::sqrt(140.0));
}

TEST(TensorContractionVectorAlgebraTest, ScalesAndAxpy)
{
  auto x = make_vector();
  auto y = utc::make_like(x);
  y.fill(1.0);

  utc::scale(x, 0.5);
  EXPECT_DOUBLE_EQ(x.values(0)[0], 0.5);
  EXPECT_DOUBLE_EQ(x.values(0)[3], -2.0);
  EXPECT_DOUBLE_EQ(x.values(1)[2], 3.5);

  utc::axpy(2.0, x, y);
  EXPECT_DOUBLE_EQ(y.values(0)[0], 2.0);
  EXPECT_DOUBLE_EQ(y.values(0)[1], -1.0);
  EXPECT_DOUBLE_EQ(y.values(0)[2], 4.0);
  EXPECT_DOUBLE_EQ(y.values(0)[3], -3.0);
  EXPECT_DOUBLE_EQ(y.values(1)[0], 6.0);
  EXPECT_DOUBLE_EQ(y.values(1)[1], -5.0);
  EXPECT_DOUBLE_EQ(y.values(1)[2], 8.0);
}

TEST(TensorContractionVectorAlgebraTest, CopiesAndZeros)
{
  auto x = make_vector();
  auto y = utc::make_like(x);

  utc::copy(x, y);
  EXPECT_DOUBLE_EQ(y.values(0)[2], 3.0);
  EXPECT_DOUBLE_EQ(y.values(1)[1], -6.0);

  utc::zero(y);
  for (std::size_t block = 0; block < y.blocks().size(); ++block)
  {
    for (double value : y.values(block))
    {
      EXPECT_DOUBLE_EQ(value, 0.0);
    }
  }
}

TEST(TensorContractionVectorAlgebraTest, MultipliesBlockPairs)
{
  std::array lhs_blocks{utc::MatrixFamily::Block{2, 3}, utc::MatrixFamily::Block{1, 2}};
  std::array rhs_blocks{utc::MatrixFamily::Block{3, 2}, utc::MatrixFamily::Block{2, 1}};
  std::array result_blocks{utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{1, 1}};
  utc::MatrixFamily lhs(lhs_blocks);
  utc::MatrixFamily rhs(rhs_blocks);
  utc::MatrixFamily result(result_blocks);
  lhs.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  rhs.assign(0, std::array{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
  lhs.assign(1, std::array{2.0, -3.0});
  rhs.assign(1, std::array{5.0, 7.0});

  utc::gemm_each(lhs, rhs, result);

  EXPECT_DOUBLE_EQ(result.values(0)[0], 58.0);
  EXPECT_DOUBLE_EQ(result.values(0)[1], 64.0);
  EXPECT_DOUBLE_EQ(result.values(0)[2], 139.0);
  EXPECT_DOUBLE_EQ(result.values(0)[3], 154.0);
  EXPECT_DOUBLE_EQ(result.values(1)[0], -11.0);
}

TEST(TensorContractionVectorAlgebraTest, MultipliesSelectedBlockPairs)
{
  std::array lhs_blocks{utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{2, 2}};
  std::array rhs_blocks{utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1}};
  std::array result_blocks{utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1},
                           utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1}};
  std::array<std::size_t, 4> lhs_selector{0, 0, 1, 1};
  std::array<std::size_t, 4> rhs_selector{0, 1, 0, 1};
  utc::MatrixFamily lhs(lhs_blocks);
  utc::MatrixFamily rhs(rhs_blocks);
  utc::MatrixFamily result(result_blocks);
  lhs.assign(0, std::array{1.0, 2.0, 3.0, 4.0});
  lhs.assign(1, std::array{5.0, 6.0, 7.0, 8.0});
  rhs.assign(0, std::array{9.0, 10.0});
  rhs.assign(1, std::array{11.0, 12.0});

  utc::gemm_selected(lhs, rhs, result, lhs_selector, rhs_selector);

  EXPECT_DOUBLE_EQ(result.values(0)[0], 29.0);
  EXPECT_DOUBLE_EQ(result.values(0)[1], 67.0);
  EXPECT_DOUBLE_EQ(result.values(1)[0], 35.0);
  EXPECT_DOUBLE_EQ(result.values(1)[1], 81.0);
  EXPECT_DOUBLE_EQ(result.values(2)[0], 105.0);
  EXPECT_DOUBLE_EQ(result.values(2)[1], 143.0);
  EXPECT_DOUBLE_EQ(result.values(3)[0], 127.0);
  EXPECT_DOUBLE_EQ(result.values(3)[1], 173.0);
}

TEST(TensorContractionVectorAlgebraTest, NormalizesAndReturnsOriginalNorm)
{
  auto x = make_vector();

  double const original_norm = utc::normalize(x);

  EXPECT_DOUBLE_EQ(original_norm, std::sqrt(140.0));
  EXPECT_NEAR(utc::norm(x), 1.0, 1.0e-14);
}

TEST(TensorContractionVectorAlgebraTest, EngineRunsOperationsThroughTensorContraction)
{
  utc::VectorAlgebraEngine engine;
  auto x = make_vector();
  auto y = make_vector();
  y.assign(0, std::array{2.0, 3.0, -1.0, 0.5});
  y.assign(1, std::array{4.0, -2.0, 1.0});

  EXPECT_DOUBLE_EQ(engine.dot(x, y), 30.0);
  EXPECT_DOUBLE_EQ(engine.norm2(x), 140.0);
  EXPECT_DOUBLE_EQ(engine.norm(x), std::sqrt(140.0));

  engine.scale(x, 0.5);
  engine.synchronize(x);
  EXPECT_DOUBLE_EQ(x.values(0)[0], 0.5);
  EXPECT_DOUBLE_EQ(x.values(0)[3], -2.0);
  EXPECT_DOUBLE_EQ(x.values(1)[2], 3.5);

  engine.zero(y);
  engine.synchronize(y);
  for (std::size_t block = 0; block < y.blocks().size(); ++block)
  {
    for (double value : y.values(block))
    {
      EXPECT_DOUBLE_EQ(value, 0.0);
    }
  }

  engine.axpy(2.0, x, y);
  engine.synchronize(y);
  EXPECT_DOUBLE_EQ(y.values(0)[0], 1.0);
  EXPECT_DOUBLE_EQ(y.values(0)[1], -2.0);
  EXPECT_DOUBLE_EQ(y.values(0)[2], 3.0);
  EXPECT_DOUBLE_EQ(y.values(0)[3], -4.0);
  EXPECT_DOUBLE_EQ(y.values(1)[0], 5.0);
  EXPECT_DOUBLE_EQ(y.values(1)[1], -6.0);
  EXPECT_DOUBLE_EQ(y.values(1)[2], 7.0);

  auto z = utc::make_like(y);
  engine.copy(y, z);
  engine.synchronize(z);
  EXPECT_DOUBLE_EQ(z.values(0)[2], 3.0);
  EXPECT_DOUBLE_EQ(z.values(1)[1], -6.0);

  double const original_norm = engine.normalize(z);
  EXPECT_DOUBLE_EQ(original_norm, std::sqrt(140.0));
  EXPECT_NEAR(engine.norm(z), 1.0, 1.0e-14);

  std::array lhs_blocks{utc::MatrixFamily::Block{2, 2}};
  std::array rhs_blocks{utc::MatrixFamily::Block{2, 1}};
  std::array result_blocks{utc::MatrixFamily::Block{2, 1}};
  utc::MatrixFamily lhs(lhs_blocks);
  utc::MatrixFamily rhs(rhs_blocks);
  utc::MatrixFamily result(result_blocks);
  lhs.assign(0, std::array{1.0, 2.0, 3.0, 4.0});
  rhs.assign(0, std::array{5.0, 6.0});

  engine.gemm_each(lhs, rhs, result);
  engine.synchronize(result);

  EXPECT_DOUBLE_EQ(result.values(0)[0], 17.0);
  EXPECT_DOUBLE_EQ(result.values(0)[1], 39.0);
}

TEST(TensorContractionVectorAlgebraTest, EngineDotAccumulatesManyBlocksIntoDevicePartials)
{
  std::vector<utc::MatrixFamily::Block> blocks(16, utc::MatrixFamily::Block{1, 1});
  utc::MatrixFamily x(blocks);
  utc::MatrixFamily y(blocks);
  double expected = 0.0;
  for (std::size_t block = 0; block < blocks.size(); ++block)
  {
    double const x_value = static_cast<double>(block + 1);
    double const y_value = static_cast<double>(2 * block + 3);
    x.assign(block, std::array{x_value});
    y.assign(block, std::array{y_value});
    expected += x_value * y_value;
  }

  utc::VectorAlgebraEngine engine;
  EXPECT_DOUBLE_EQ(engine.dot(x, y), expected);
  EXPECT_DOUBLE_EQ(engine.dot(x, y), expected);

  y.fill(0.0);
  engine.upload(y);
  EXPECT_DOUBLE_EQ(engine.dot(x, y), 0.0);
}

TEST(TensorContractionVectorAlgebraTest, EngineRunsResidentSlabOperationsAcrossTwoLocalGpus)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2)
  {
    GTEST_SKIP() << "requires at least two visible CUDA devices";
  }

  EnvGuard env_guard({"UNI20_TENSORCONTRACTION_BACKEND", "UNI20_TENSORCONTRACTION_DEVICES",
                      "UNI20_TENSORCONTRACTION_MULTI_GPU_MIN_BYTES", "UNI20_TENSORCONTRACTION_MULTI_GPU_MIN_FLOPS"});
  unsetenv("UNI20_TENSORCONTRACTION_BACKEND");
  setenv("UNI20_TENSORCONTRACTION_DEVICES", "2", 1);
  setenv("UNI20_TENSORCONTRACTION_MULTI_GPU_MIN_BYTES", "0", 1);
  setenv("UNI20_TENSORCONTRACTION_MULTI_GPU_MIN_FLOPS", "0", 1);

  std::vector<utc::MatrixFamily::Block> blocks(8, utc::MatrixFamily::Block{2, 2});
  utc::MatrixFamily x(blocks);
  utc::MatrixFamily y(blocks);
  utc::MatrixFamily z(blocks);
  double expected_dot = 0.0;
  std::vector<double> expected_z(x.coalesced_values().size());
  for (std::size_t index = 0; index < x.coalesced_values().size(); ++index)
  {
    double const x_value = static_cast<double>(index + 1);
    double const y_value = static_cast<double>(3 * index + 2);
    x.coalesced_values()[index] = x_value;
    y.coalesced_values()[index] = y_value;
    expected_dot += x_value * y_value;
    expected_z[index] = y_value - 0.5 * x_value;
  }

  utc::VectorAlgebraEngine engine;
  engine.upload(x);
  engine.upload(y);
  engine.set_host_synchronization(false);

  EXPECT_DOUBLE_EQ(engine.dot(x, y), expected_dot);
  engine.scale(x, 2.0);
  engine.axpy(-0.25, x, y);
  engine.copy(y, z);
  engine.synchronize(z);

  for (std::size_t index = 0; index < z.coalesced_values().size(); ++index)
  {
    EXPECT_DOUBLE_EQ(z.coalesced_values()[index], expected_z[index]);
  }
}

TEST(TensorContractionVectorAlgebraTest, EngineHandlesNonContiguousLogicalBlocksInResidentSlabs)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1)
  {
    GTEST_SKIP() << "requires at least one visible CUDA device";
  }

  EnvGuard env_guard({"UNI20_TENSORCONTRACTION_BACKEND", "UNI20_TENSORCONTRACTION_DEVICES"});
  unsetenv("UNI20_TENSORCONTRACTION_BACKEND");
  setenv("UNI20_TENSORCONTRACTION_DEVICES", "1", 1);

  std::vector<utc::MatrixFamily::Block> blocks{utc::MatrixFamily::Block{1, 2}, utc::MatrixFamily::Block{2, 1},
                                               utc::MatrixFamily::Block{1, 3}, utc::MatrixFamily::Block{3, 1},
                                               utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{1, 1},
                                               utc::MatrixFamily::Block{2, 3}, utc::MatrixFamily::Block{3, 2}};
  utc::MatrixFamily x(blocks);
  utc::MatrixFamily y(blocks);
  utc::MatrixFamily z(blocks);

  double expected_dot = 0.0;
  std::vector<double> expected_z(x.coalesced_values().size());
  for (std::size_t index = 0; index < x.coalesced_values().size(); ++index)
  {
    double const x_value = static_cast<double>(index + 1);
    double const y_value = static_cast<double>(2 * index + 5);
    x.coalesced_values()[index] = x_value;
    y.coalesced_values()[index] = y_value;
    expected_dot += x_value * y_value;
    expected_z[index] = y_value - 0.25 * x_value;
  }

  utc::VectorAlgebraEngine engine;
  if (engine.uses_host_backend())
  {
    GTEST_SKIP() << "requires the TensorContraction resident CUDA backend";
  }
  engine.upload(x);
  engine.upload(y);
  engine.set_host_synchronization(false);

  auto& swapper = engine.resident_arranger().residentSwapper();
  auto even_blocks = select_matrices(x, {0, 2, 4, 6});
  auto odd_blocks = select_matrices(x, {1, 3, 5, 7});
  swapper.ensurePreStoreCoalescedOnDevice(even_blocks, 0, /*preserveExistingContent=*/true);
  swapper.ensurePreStoreCoalescedOnDevice(odd_blocks, 0, /*preserveExistingContent=*/true);

  ASSERT_TRUE(swapper.preStoreBuffersAreCoalesced(even_blocks, 0));
  ASSERT_TRUE(swapper.preStoreBuffersAreCoalesced(odd_blocks, 0));
  EXPECT_FALSE(swapper.preStoreBuffersAreCoalesced(utc::raw_matrices(x), 0));

  EXPECT_NEAR(engine.dot(x, y), expected_dot, 1.0e-12);
  engine.axpy(-0.25, x, y);
  engine.copy(y, z);
  engine.synchronize(z);

  for (std::size_t index = 0; index < z.coalesced_values().size(); ++index)
  {
    EXPECT_DOUBLE_EQ(z.coalesced_values()[index], expected_z[index]);
  }
}

TEST(TensorContractionVectorAlgebraTest, EngineCanKeepMutationsResidentUntilExplicitSync)
{
  utc::VectorAlgebraEngine engine;
  auto x = make_vector();

  engine.upload(x);
  engine.set_host_synchronization(false);
  engine.scale(x, 2.0);

  EXPECT_DOUBLE_EQ(x.values(0)[0], 1.0);
  EXPECT_DOUBLE_EQ(x.values(1)[2], 7.0);

  engine.synchronize(x);
  EXPECT_DOUBLE_EQ(x.values(0)[0], 2.0);
  EXPECT_DOUBLE_EQ(x.values(1)[2], 14.0);
}

TEST(TensorContractionVectorAlgebraTest, EngineReleaseDropsResidentStorage)
{
  utc::VectorAlgebraEngine engine;
  if (engine.uses_host_backend())
  {
    GTEST_SKIP() << "requires the TensorContraction resident CUDA backend";
  }

  auto x = make_vector();
  engine.upload(x);
  engine.set_host_synchronization(false);
  engine.scale(x, 2.0);

  auto& swapper = engine.resident_arranger().residentSwapper();
  for (auto const& matrix : utc::raw_matrices(x))
  {
    auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(matrix);
    EXPECT_GE(device_id, 0);
    EXPECT_NE(buffer, nullptr);
  }

  engine.release(x);
  for (auto const& matrix : utc::raw_matrices(x))
  {
    auto [device_id, buffer] = swapper.getPreStoreBufferOrNone(matrix);
    EXPECT_EQ(device_id, -1);
    EXPECT_EQ(buffer, nullptr);
  }

  engine.upload(x);
  engine.scale(x, 3.0);
  engine.synchronize(x);
  EXPECT_DOUBLE_EQ(x.values(0)[0], 3.0);
  EXPECT_DOUBLE_EQ(x.values(0)[1], -6.0);
  EXPECT_DOUBLE_EQ(x.values(1)[2], 21.0);
}

TEST(TensorContractionVectorAlgebraTest, EngineCanBuildGemmOutputResidentFromHostInputs)
{
  utc::VectorAlgebraEngine engine;
  std::array lhs_blocks{utc::MatrixFamily::Block{2, 2}};
  std::array rhs_blocks{utc::MatrixFamily::Block{2, 1}};
  std::array result_blocks{utc::MatrixFamily::Block{2, 1}};
  utc::MatrixFamily lhs(lhs_blocks);
  utc::MatrixFamily rhs(rhs_blocks);
  utc::MatrixFamily result(result_blocks);
  lhs.assign(0, std::array{1.0, 2.0, 3.0, 4.0});
  rhs.assign(0, std::array{5.0, 6.0});

  engine.gemm_each_to_resident(lhs, rhs, result);

  if (!engine.uses_host_backend())
  {
    EXPECT_DOUBLE_EQ(result.values(0)[0], 0.0);
    EXPECT_DOUBLE_EQ(result.values(0)[1], 0.0);
  }
  engine.synchronize(result);
  EXPECT_DOUBLE_EQ(result.values(0)[0], 17.0);
  EXPECT_DOUBLE_EQ(result.values(0)[1], 39.0);
}

TEST(TensorContractionVectorAlgebraTest, EngineCanBuildSelectedGemmOutputResidentFromHostInputs)
{
  utc::VectorAlgebraEngine engine;
  std::array lhs_blocks{utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{2, 2}};
  std::array rhs_blocks{utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1}};
  std::array result_blocks{utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1},
                           utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1}};
  std::array<std::size_t, 4> lhs_selector{0, 0, 1, 1};
  std::array<std::size_t, 4> rhs_selector{0, 1, 0, 1};
  utc::MatrixFamily lhs(lhs_blocks);
  utc::MatrixFamily rhs(rhs_blocks);
  utc::MatrixFamily result(result_blocks);
  lhs.assign(0, std::array{1.0, 2.0, 3.0, 4.0});
  lhs.assign(1, std::array{5.0, 6.0, 7.0, 8.0});
  rhs.assign(0, std::array{9.0, 10.0});
  rhs.assign(1, std::array{11.0, 12.0});

  engine.gemm_selected_to_resident(lhs, rhs, result, lhs_selector, rhs_selector);

  if (!engine.uses_host_backend())
  {
    EXPECT_DOUBLE_EQ(result.values(0)[0], 0.0);
    EXPECT_DOUBLE_EQ(result.values(3)[1], 0.0);
  }
  engine.synchronize(result);
  EXPECT_DOUBLE_EQ(result.values(0)[0], 29.0);
  EXPECT_DOUBLE_EQ(result.values(0)[1], 67.0);
  EXPECT_DOUBLE_EQ(result.values(1)[0], 35.0);
  EXPECT_DOUBLE_EQ(result.values(1)[1], 81.0);
  EXPECT_DOUBLE_EQ(result.values(2)[0], 105.0);
  EXPECT_DOUBLE_EQ(result.values(2)[1], 143.0);
  EXPECT_DOUBLE_EQ(result.values(3)[0], 127.0);
  EXPECT_DOUBLE_EQ(result.values(3)[1], 173.0);
}

TEST(TensorContractionVectorAlgebraTest, EngineCanBuildSparseSelectedGemmOutputResidentFromHostInputs)
{
  utc::VectorAlgebraEngine engine;
  std::array lhs_blocks{utc::MatrixFamily::Block{2, 2}, utc::MatrixFamily::Block{2, 2}};
  std::array rhs_blocks{utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1}};
  std::array result_blocks{utc::MatrixFamily::Block{2, 1}, utc::MatrixFamily::Block{2, 1},
                           utc::MatrixFamily::Block{2, 1}};
  std::array<std::size_t, 2> lhs_selector{0, 1};
  std::array<std::size_t, 2> rhs_selector{1, 0};
  std::array<std::size_t, 2> result_selector{0, 2};
  utc::MatrixFamily lhs(lhs_blocks);
  utc::MatrixFamily rhs(rhs_blocks);
  utc::MatrixFamily result(result_blocks);
  lhs.assign(0, std::array{1.0, 2.0, 3.0, 4.0});
  lhs.assign(1, std::array{5.0, 6.0, 7.0, 8.0});
  rhs.assign(0, std::array{9.0, 10.0});
  rhs.assign(1, std::array{11.0, 12.0});
  result.fill(-1.0);

  engine.gemm_sparse_selected_to_resident(lhs, rhs, result, lhs_selector, rhs_selector, result_selector);

  if (!engine.uses_host_backend())
  {
    EXPECT_DOUBLE_EQ(result.values(0)[0], -1.0);
    EXPECT_DOUBLE_EQ(result.values(2)[1], -1.0);
  }
  engine.synchronize(result);
  EXPECT_DOUBLE_EQ(result.values(0)[0], 35.0);
  EXPECT_DOUBLE_EQ(result.values(0)[1], 81.0);
  EXPECT_DOUBLE_EQ(result.values(1)[0], 0.0);
  EXPECT_DOUBLE_EQ(result.values(1)[1], 0.0);
  EXPECT_DOUBLE_EQ(result.values(2)[0], 105.0);
  EXPECT_DOUBLE_EQ(result.values(2)[1], 143.0);
}

TEST(TensorContractionVectorAlgebraTest, RejectsZeroNormalizeAndShapeMismatches)
{
  auto x = make_vector();
  auto zero = utc::make_like(x);
  std::array wrong_blocks{utc::MatrixFamily::Block{7, 1}};
  utc::MatrixFamily wrong(wrong_blocks);
  std::array<std::size_t, 1> selector{0};

  EXPECT_THROW(utc::normalize(zero), std::invalid_argument);
  EXPECT_THROW(utc::dot(x, wrong), std::invalid_argument);
  EXPECT_THROW(utc::axpy(1.0, x, wrong), std::invalid_argument);
  EXPECT_THROW(utc::copy(x, wrong), std::invalid_argument);
  EXPECT_THROW(utc::gemm_each(x, x, wrong), std::invalid_argument);
  EXPECT_THROW(utc::gemm_selected(x, x, wrong, selector, selector), std::invalid_argument);
}
