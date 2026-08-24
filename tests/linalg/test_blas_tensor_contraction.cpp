#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using uni20::linalg::BlasBackend;
using uni20::linalg::CpuReferenceBackend;
using direct_blas_backend = uni20::linalg::DirectGemmContractionBackend<BlasBackend>;
using looped_blas_backend = uni20::linalg::LoopedGemmContractionBackend<BlasBackend>;
using host_backends = uni20::linalg::backend_list<direct_blas_backend, looped_blas_backend, CpuReferenceBackend>;

[[nodiscard]] auto selected_backend(std::vector<uni20::linalg::dispatch_diagnostics::event> const& events,
                                    std::string_view operation) -> std::optional<std::string_view>
{
  for (auto const& event : events)
  {
    if (event.operation == operation && event.selected_backend()) return *event.selected_backend();
  }
  return std::nullopt;
}

[[nodiscard]] auto operation_count(std::vector<uni20::linalg::dispatch_diagnostics::event> const& events,
                                   std::string_view operation) -> std::size_t
{
  return static_cast<std::size_t>(
      std::ranges::count(events, operation, &uni20::linalg::dispatch_diagnostics::event::operation));
}

template <class BackendSelector, class OutputTensor, class LhsTensor, class RhsTensor, std::size_t ContractedRank>
[[nodiscard]] bool
try_normalized_contract(BackendSelector&& selector, OutputTensor& output, uni20::tensor_element_t<OutputTensor> alpha,
                        LhsTensor const& lhs, RhsTensor const& rhs,
                        std::array<std::pair<std::size_t, std::size_t>, ContractedRank> const& requested_axes,
                        uni20::tensor_element_t<OutputTensor> beta)
{
  constexpr std::size_t lhs_rank = uni20::tensor_mdspec_t<LhsTensor>::rank();
  constexpr std::size_t rhs_rank = uni20::tensor_mdspec_t<RhsTensor>::rank();
  auto operation = uni20::linalg::contract_op<lhs_rank, rhs_rank, ContractedRank>{
      .axes = uni20::linalg::make_contraction_axes<lhs_rank, rhs_rank>(requested_axes)};
  auto output_descriptor = uni20::mdspec_of(output);
  auto lhs_descriptor = uni20::mdspec_of(lhs);
  auto rhs_descriptor = uni20::mdspec_of(rhs);
  return uni20::linalg::try_dispatch_kernel(std::forward<BackendSelector>(selector), std::move(operation),
                                            output_descriptor, alpha, lhs_descriptor, rhs_descriptor, beta);
}

} // namespace

TEST(BlasTensorContractionTest, DefaultSelectorUsesDirectBackendAndOneBlasGemm)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) { events.push_back(event); });
  uni20::RowMajorTensor<double, 3> lhs(2, 3, 4);
  uni20::RowMajorTensor<double, 2> rhs(4, 5);
  uni20::RowMajorTensor<double, 3> output(2, 3, 5);

  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type k = 0; k < 4; ++k)
        lhs[i, j, k] = static_cast<double>(1 + 13 * i + 4 * j + k);
  for (uni20::index_type k = 0; k < 4; ++k)
    for (uni20::index_type n = 0; n < 5; ++n)
      rhs[k, n] = static_cast<double>(2 + 5 * k + n);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type n = 0; n < 5; ++n)
        output[i, j, n] = 3.0;

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{2, 0}}};
  uni20::linalg::contract(output, 2.0, lhs, rhs, axes, 0.5);

  for (uni20::index_type i = 0; i < 2; ++i)
  {
    for (uni20::index_type j = 0; j < 3; ++j)
    {
      for (uni20::index_type n = 0; n < 5; ++n)
      {
        double expected = 1.5;
        for (uni20::index_type k = 0; k < 4; ++k)
          expected += 2.0 * lhs[i, j, k] * rhs[k, n];
        EXPECT_DOUBLE_EQ((output[i, j, n]), expected);
      }
    }
  }

  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(selected_backend(events, "contract"), "direct_gemm_contraction");
  EXPECT_EQ(selected_backend(events, "gemm"), "blas");
}

TEST(BlasTensorContractionTest, PreservesSupportedConjugatingAccessorSemantics)
{
  using scalar_type = uni20::complex<double>;
  uni20::RowMajorTensor<scalar_type, 2> lhs(2, 3);
  uni20::RowMajorTensor<scalar_type, 2> rhs(3, 2);
  uni20::ColumnMajorTensor<scalar_type, 2> output(2, 2);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 3; ++k)
      lhs[i, k] = scalar_type{static_cast<double>(1 + i + k), static_cast<double>(2 * i - k)};
  for (uni20::index_type k = 0; k < 3; ++k)
    for (uni20::index_type n = 0; n < 2; ++n)
      rhs[k, n] = scalar_type{static_cast<double>(2 + k), static_cast<double>(n - k)};
  auto conjugated_lhs = uni20::conj(lhs);

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};
  uni20::linalg::contract(direct_blas_backend{BlasBackend{}}, output, scalar_type{1.0}, conjugated_lhs, rhs, axes,
                          scalar_type{});

  for (uni20::index_type i = 0; i < 2; ++i)
  {
    for (uni20::index_type n = 0; n < 2; ++n)
    {
      scalar_type expected{};
      for (uni20::index_type k = 0; k < 3; ++k)
        expected += uni20::conj(lhs[i, k]) * rhs[k, n];
      EXPECT_EQ((output[i, n]), expected);
    }
  }
}

TEST(BlasTensorContractionTest, SupportsFloatAndComplexFloatScalars)
{
  auto run = []<class Scalar>() {
    uni20::RowMajorTensor<Scalar, 2> lhs(2, 2);
    uni20::RowMajorTensor<Scalar, 2> rhs(2, 2);
    uni20::RowMajorTensor<Scalar, 2> output(2, 2);
    lhs[0, 0] = Scalar{1};
    lhs[0, 1] = Scalar{2};
    lhs[1, 0] = Scalar{3};
    lhs[1, 1] = Scalar{4};
    rhs[0, 0] = Scalar{1};
    rhs[0, 1] = Scalar{};
    rhs[1, 0] = Scalar{};
    rhs[1, 1] = Scalar{1};
    std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};

    uni20::linalg::contract(direct_blas_backend{BlasBackend{}}, output, Scalar{1}, lhs, rhs, axes, Scalar{});

    EXPECT_EQ((output[0, 0]), Scalar{1});
    EXPECT_EQ((output[0, 1]), Scalar{2});
    EXPECT_EQ((output[1, 0]), Scalar{3});
    EXPECT_EQ((output[1, 1]), Scalar{4});
  };

  run.template operator()<float>();
  run.template operator()<uni20::complex<float>>();
}

TEST(BlasTensorContractionTest, AcquiresDeferredHostDescriptors)
{
  uni20::test::DeferredHostTensor<double, 2> lhs(2, 3);
  uni20::test::DeferredHostTensor<double, 2> rhs(3, 2);
  uni20::test::DeferredHostTensor<double, 2> output(2, 2);
  lhs.storage() = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
  rhs.storage() = {7.0, 9.0, 11.0, 8.0, 10.0, 12.0};

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};
  uni20::linalg::contract(direct_blas_backend{BlasBackend{}}, output, 1.0, lhs, rhs, axes, 0.0);

  EXPECT_EQ(output.storage(), (std::vector<double>{58.0, 139.0, 64.0, 154.0}));
}

TEST(BlasTensorContractionTest, HandlesLogicalUnitKDimensionForOuterProducts)
{
  uni20::Tensor<double, 1> lhs(2);
  uni20::Tensor<double, 1> rhs(3);
  uni20::Tensor<double, 2> outer(2, 3);
  lhs[0] = 2.0;
  lhs[1] = -1.0;
  rhs[0] = 3.0;
  rhs[1] = 4.0;
  rhs[2] = -2.0;
  std::array<std::pair<std::size_t, std::size_t>, 0> const no_axes{};

  uni20::linalg::contract(direct_blas_backend{BlasBackend{}}, outer, 1.0, lhs, rhs, no_axes, 0.0);

  EXPECT_DOUBLE_EQ((outer[0, 0]), 6.0);
  EXPECT_DOUBLE_EQ((outer[0, 1]), 8.0);
  EXPECT_DOUBLE_EQ((outer[0, 2]), -4.0);
  EXPECT_DOUBLE_EQ((outer[1, 0]), -3.0);
  EXPECT_DOUBLE_EQ((outer[1, 1]), -4.0);
  EXPECT_DOUBLE_EQ((outer[1, 2]), 2.0);
}

TEST(BlasTensorContractionTest, HandlesLogicalUnitMDimensionWhenLhsIsFullyContracted)
{
  uni20::Tensor<double, 1> lhs(3);
  uni20::RowMajorTensor<double, 2> rhs(3, 2);
  uni20::Tensor<double, 1> output(2);
  lhs[0] = 2.0;
  lhs[1] = -1.0;
  lhs[2] = 3.0;
  rhs[0, 0] = 1.0;
  rhs[0, 1] = 4.0;
  rhs[1, 0] = 2.0;
  rhs[1, 1] = -2.0;
  rhs[2, 0] = 5.0;
  rhs[2, 1] = 1.0;
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{0, 0}}};

  uni20::linalg::contract(direct_blas_backend{BlasBackend{}}, output, 1.0, lhs, rhs, axes, 0.0);

  EXPECT_DOUBLE_EQ(output[0], 15.0);
  EXPECT_DOUBLE_EQ(output[1], 13.0);
}

TEST(BlasTensorContractionTest, ProjectsFullContractionToOneByOneGemm)
{
  uni20::Tensor<double, 1> lhs(3);
  uni20::Tensor<double, 1> rhs(3);
  uni20::ScalarTensor<double> output;
  lhs[0] = 2.0;
  lhs[1] = -1.0;
  lhs[2] = 3.0;
  rhs[0] = 4.0;
  rhs[1] = 5.0;
  rhs[2] = -2.0;
  output[] = 7.0;
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{0, 0}}};

  uni20::linalg::contract(direct_blas_backend{BlasBackend{}}, output, 2.0, lhs, rhs, axes, 0.5);

  EXPECT_DOUBLE_EQ(output[], -2.5);
}

TEST(BlasTensorContractionTest, ZeroContractedExtentFallsThroughWithoutLosingBetaUpdate)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) { events.push_back(event); });
  uni20::Tensor<double, 2> lhs(2, 0);
  uni20::Tensor<double, 2> rhs(0, 3);
  uni20::Tensor<double, 2> output(2, 3);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      output[row, column] = 7.0;
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};

  uni20::linalg::contract(
      host_backends{direct_blas_backend{BlasBackend{}}, looped_blas_backend{BlasBackend{}}, CpuReferenceBackend{}},
      output, 3.0, lhs, rhs, axes, 2.0);

  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((output[row, column]), 14.0);
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(selected_backend(events, "contract"), "cpu_reference");
}

TEST(BlasTensorContractionTest, NonmergeableGroupDeclinesWithoutModifyingOutput)
{
  using strided_tensor = uni20::StridedTensor<double, 3>;
  strided_tensor lhs(strided_tensor::extents_type{2, 2, 3}, std::array<uni20::index_type, 3>{10, 3, 1});
  uni20::RowMajorTensor<double, 2> rhs(3, 4);
  uni20::RowMajorTensor<double, 3> output(2, 2, 4);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type k = 0; k < 3; ++k)
        lhs[i, j, k] = static_cast<double>(1 + 7 * i + 3 * j + k);
  for (uni20::index_type k = 0; k < 3; ++k)
    for (uni20::index_type n = 0; n < 4; ++n)
      rhs[k, n] = static_cast<double>(2 + 4 * k + n);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type n = 0; n < 4; ++n)
        output[i, j, n] = -7.0;

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{2, 0}}};
  EXPECT_FALSE(try_normalized_contract(direct_blas_backend{BlasBackend{}}, output, 1.0, lhs, rhs, axes, 0.0));
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type n = 0; n < 4; ++n)
        EXPECT_DOUBLE_EQ((output[i, j, n]), -7.0);
}

TEST(BlasTensorContractionTest, ResidualMGroupUsesLoopedGemm)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) { events.push_back(event); });
  using strided_tensor = uni20::StridedTensor<double, 3>;
  strided_tensor lhs(strided_tensor::extents_type{2, 2, 3}, std::array<uni20::index_type, 3>{10, 3, 1});
  uni20::RowMajorTensor<double, 2> rhs(3, 2);
  uni20::RowMajorTensor<double, 3> output(2, 2, 2);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type k = 0; k < 3; ++k)
        lhs[i, j, k] = static_cast<double>(1 + 7 * i + 3 * j + k);
  for (uni20::index_type k = 0; k < 3; ++k)
    for (uni20::index_type n = 0; n < 2; ++n)
      rhs[k, n] = static_cast<double>(2 + 2 * k + n);

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{2, 0}}};
  uni20::linalg::contract(output, 1.0, lhs, rhs, axes, 0.0);

  for (uni20::index_type i = 0; i < 2; ++i)
  {
    for (uni20::index_type j = 0; j < 2; ++j)
    {
      for (uni20::index_type n = 0; n < 2; ++n)
      {
        double expected = 0.0;
        for (uni20::index_type k = 0; k < 3; ++k)
          expected += lhs[i, j, k] * rhs[k, n];
        EXPECT_DOUBLE_EQ((output[i, j, n]), expected);
      }
    }
  }

  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(selected_backend(events, "contract"), "looped_gemm_contraction");
  EXPECT_EQ(operation_count(events, "gemm"), 2);
}

TEST(BlasTensorContractionTest, ResidualNGroupUsesOriginalBetaForEverySlice)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) { events.push_back(event); });
  uni20::RowMajorTensor<double, 2> lhs(2, 3);
  using strided_tensor = uni20::StridedTensor<double, 3>;
  strided_tensor rhs(strided_tensor::extents_type{3, 2, 2}, std::array<uni20::index_type, 3>{1, 10, 3});
  uni20::RowMajorTensor<double, 3> output(2, 2, 2);

  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 3; ++k)
      lhs[i, k] = static_cast<double>(1 + 3 * i + k);
  for (uni20::index_type k = 0; k < 3; ++k)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type n = 0; n < 2; ++n)
        rhs[k, j, n] = static_cast<double>(2 + k + 3 * j + n);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type n = 0; n < 2; ++n)
        output[i, j, n] = 5.0;

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};
  uni20::linalg::contract(output, 2.0, lhs, rhs, axes, 0.5);

  for (uni20::index_type i = 0; i < 2; ++i)
  {
    for (uni20::index_type j = 0; j < 2; ++j)
    {
      for (uni20::index_type n = 0; n < 2; ++n)
      {
        double expected = 2.5;
        for (uni20::index_type k = 0; k < 3; ++k)
          expected += 2.0 * lhs[i, k] * rhs[k, j, n];
        EXPECT_DOUBLE_EQ((output[i, j, n]), expected);
      }
    }
  }

  ASSERT_EQ(events.size(), 3);
  EXPECT_EQ(selected_backend(events, "contract"), "looped_gemm_contraction");
  EXPECT_EQ(operation_count(events, "gemm"), 2);
}

TEST(BlasTensorContractionTest, LoopedGemmOffsetsDeferredHostDescriptors)
{
  using extents_type = stdex::dextents<uni20::index_type, 3>;
  using lhs_mapping_type = stdex::layout_stride::mapping<extents_type>;
  using output_mapping_type = stdex::layout_right::mapping<extents_type>;
  using matrix_extents_type = stdex::dextents<uni20::index_type, 2>;
  using matrix_mapping_type = stdex::layout_right::mapping<matrix_extents_type>;
  using mutable_descriptor = uni20::test::HostStorageDescriptor<std::vector<double>>;
  using const_descriptor = uni20::test::HostStorageDescriptor<std::vector<double> const>;
  using output_mdspec =
      uni20::mdspec<double, extents_type, stdex::layout_right, stdex::default_accessor<double>, mutable_descriptor>;
  using lhs_mdspec = uni20::mdspec<double const, extents_type, stdex::layout_stride,
                                   stdex::default_accessor<double const>, const_descriptor>;
  using rhs_mdspec = uni20::mdspec<double const, matrix_extents_type, stdex::layout_right,
                                   stdex::default_accessor<double const>, const_descriptor>;

  std::vector<double> lhs_storage(16);
  std::vector<double> rhs_storage(12);
  std::vector<double> output_storage(16, -3.0);
  auto const lhs_mapping = lhs_mapping_type{extents_type{2, 2, 3}, std::array<uni20::index_type, 3>{10, 3, 1}};
  auto const output_mapping = output_mapping_type{extents_type{2, 2, 2}};
  auto const rhs_mapping = matrix_mapping_type{matrix_extents_type{3, 2}};
  stdex::mdspan<double, extents_type, stdex::layout_stride> lhs_values(lhs_storage.data(), lhs_mapping);
  stdex::mdspan<double, matrix_extents_type, stdex::layout_right> rhs_values(rhs_storage.data(), rhs_mapping);
  stdex::mdspan<double, extents_type, stdex::layout_right> output_values(output_storage.data(), output_mapping);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type k = 0; k < 3; ++k)
        lhs_values[i, j, k] = static_cast<double>(1 + 7 * i + 3 * j + k);
  for (uni20::index_type k = 0; k < 3; ++k)
    for (uni20::index_type n = 0; n < 2; ++n)
      rhs_values[k, n] = static_cast<double>(2 + 2 * k + n);

  output_mdspec output{mutable_descriptor{&output_storage}, output_mapping, stdex::default_accessor<double>{}};
  lhs_mdspec lhs{const_descriptor{&lhs_storage}, lhs_mapping, stdex::default_accessor<double const>{}};
  rhs_mdspec rhs{const_descriptor{&rhs_storage}, rhs_mapping, stdex::default_accessor<double const>{}};
  auto const axes =
      uni20::linalg::make_contraction_axes<3, 2>(std::array<std::pair<std::size_t, std::size_t>, 1>{{{2, 0}}});
  auto operation = uni20::linalg::contract_op<3, 2, 1>{.axes = axes};
  EXPECT_TRUE(
      uni20::linalg::try_dispatch_kernel(looped_blas_backend{BlasBackend{}}, operation, output, 1.0, lhs, rhs, 0.0));

  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      for (uni20::index_type n = 0; n < 2; ++n)
      {
        double expected = 0.0;
        for (uni20::index_type k = 0; k < 3; ++k)
          expected += lhs_values[i, j, k] * rhs_values[k, n];
        EXPECT_DOUBLE_EQ((output_values[i, j, n]), expected);
      }
}
