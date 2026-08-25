#include <uni20/core/types.hpp>
#include <uni20/linalg/cpu/contract.hpp>
#include <uni20/linalg/ops/contract.hpp>
#include <uni20/mdspan/diagonal_accessor.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

} // namespace

TEST(ContractionAxesTest, CanonicalizesPairsAndPreservesSurvivingOrder)
{
  auto const axes =
      uni20::linalg::make_contraction_axes<3, 3>(std::array<std::pair<std::size_t, std::size_t>, 2>{{{2, 0}, {1, 2}}});

  EXPECT_EQ(axes.lhs_contracted, (std::array<std::size_t, 2>{1, 2}));
  EXPECT_EQ(axes.rhs_contracted, (std::array<std::size_t, 2>{2, 0}));
  EXPECT_EQ(axes.lhs_surviving, (std::array<std::size_t, 1>{0}));
  EXPECT_EQ(axes.rhs_surviving, (std::array<std::size_t, 1>{1}));
  EXPECT_TRUE(uni20::linalg::contraction_axes_are_valid(axes));
}

TEST(ContractionAxesTest, RejectsRepeatedAndOutOfRangeAxes)
{
  ErrorModeGuard const error_mode;
  EXPECT_THROW((static_cast<void>(uni20::linalg::make_contraction_axes<2, 2>(
                   std::array<std::pair<std::size_t, std::size_t>, 2>{{{0, 0}, {0, 1}}}))),
               std::runtime_error);
  EXPECT_THROW((static_cast<void>(uni20::linalg::make_contraction_axes<2, 2>(
                   std::array<std::pair<std::size_t, std::size_t>, 2>{{{0, 1}, {1, 1}}}))),
               std::runtime_error);
  EXPECT_THROW((static_cast<void>(uni20::linalg::make_contraction_axes<2, 2>(
                   std::array<std::pair<std::size_t, std::size_t>, 1>{{{2, 0}}}))),
               std::runtime_error);
}

TEST(CpuTensorContractionTest, ContractsMultipleNonAdjacentAxes)
{
  uni20::RowMajorTensor<double, 3> lhs(2, 3, 2);
  uni20::ColumnMajorTensor<double, 3> rhs(2, 4, 3);
  uni20::RowMajorTensor<double, 2> output(2, 4);

  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type k = 0; k < 2; ++k)
        lhs[i, j, k] = static_cast<double>(1 + 7 * i + 2 * j + k);

  for (uni20::index_type k = 0; k < 2; ++k)
    for (uni20::index_type n = 0; n < 4; ++n)
      for (uni20::index_type j = 0; j < 3; ++j)
        rhs[k, n, j] = static_cast<double>(2 + 5 * k + 3 * n + j);

  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type n = 0; n < 4; ++n)
      output[i, n] = 4.0;

  std::array<std::pair<std::size_t, std::size_t>, 2> const axes{{{2, 0}, {1, 2}}};
  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, 2.0, lhs, rhs, axes, 0.5);

  for (uni20::index_type i = 0; i < 2; ++i)
  {
    for (uni20::index_type n = 0; n < 4; ++n)
    {
      double expected = 2.0;
      for (uni20::index_type j = 0; j < 3; ++j)
        for (uni20::index_type k = 0; k < 2; ++k)
          expected += 2.0 * static_cast<double>(lhs[i, j, k]) * static_cast<double>(rhs[k, n, j]);
      EXPECT_DOUBLE_EQ((output[i, n]), expected);
    }
  }
}

TEST(CpuTensorContractionTest, ZeroContractedAxesProduceOuterProduct)
{
  uni20::Tensor<double, 1> lhs(2);
  uni20::Tensor<double, 1> rhs(3);
  uni20::Tensor<double, 2> output(2, 3);
  lhs[0] = 2.0;
  lhs[1] = -1.0;
  rhs[0] = 3.0;
  rhs[1] = 4.0;
  rhs[2] = -2.0;

  std::array<std::pair<std::size_t, std::size_t>, 0> const axes{};
  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, 1.0, lhs, rhs, axes, 0.0);

  EXPECT_DOUBLE_EQ((output[0, 0]), 6.0);
  EXPECT_DOUBLE_EQ((output[0, 1]), 8.0);
  EXPECT_DOUBLE_EQ((output[0, 2]), -4.0);
  EXPECT_DOUBLE_EQ((output[1, 0]), -3.0);
  EXPECT_DOUBLE_EQ((output[1, 1]), -4.0);
  EXPECT_DOUBLE_EQ((output[1, 2]), 2.0);
}

TEST(CpuTensorContractionTest, FullContractionProducesBilinearScalar)
{
  using scalar_type = uni20::complex<double>;
  uni20::Tensor<scalar_type, 2> lhs(2, 2);
  uni20::Tensor<scalar_type, 2> rhs(2, 2);
  uni20::ScalarTensor<scalar_type> output;
  lhs[0, 0] = {1.0, 1.0};
  lhs[0, 1] = {2.0, 0.0};
  lhs[1, 0] = {0.0, 1.0};
  lhs[1, 1] = {-1.0, 0.0};
  rhs[0, 0] = {1.0, 0.0};
  rhs[0, 1] = {0.0, 1.0};
  rhs[1, 0] = {2.0, 0.0};
  rhs[1, 1] = {3.0, 0.0};

  std::array<std::pair<std::size_t, std::size_t>, 2> const axes{{{0, 0}, {1, 1}}};
  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, scalar_type{1.0}, lhs, rhs, axes,
                          scalar_type{});

  scalar_type expected{};
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 2; ++j)
      expected += lhs[i, j] * rhs[i, j];
  EXPECT_EQ(output[], expected);
}

TEST(CpuTensorContractionTest, RespectsConjugatingInputAccessor)
{
  using scalar_type = uni20::complex<double>;
  uni20::Tensor<scalar_type, 1> lhs(2);
  uni20::Tensor<scalar_type, 1> rhs(2);
  uni20::ScalarTensor<scalar_type> output;
  lhs[0] = {1.0, 2.0};
  lhs[1] = {3.0, -1.0};
  rhs[0] = {2.0, 0.0};
  rhs[1] = {0.0, 1.0};
  auto conjugated_lhs = uni20::conj(lhs);

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{0, 0}}};
  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, scalar_type{1.0}, conjugated_lhs, rhs, axes,
                          scalar_type{});

  EXPECT_EQ(output[], uni20::conj(lhs[0]) * rhs[0] + uni20::conj(lhs[1]) * rhs[1]);
}

TEST(CpuTensorContractionTest, DeferredTensorOperandsUseHostLeases)
{
  uni20::test::DeferredHostTensor<double, 2> lhs(2, 3);
  uni20::test::DeferredHostTensor<double, 2> rhs(3, 2);
  uni20::test::DeferredHostTensor<double, 2> output(2, 2);
  lhs.storage() = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
  rhs.storage() = {7.0, 9.0, 11.0, 8.0, 10.0, 12.0};

  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};
  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, 1.0, lhs, rhs, axes, 0.0);

  EXPECT_EQ(output.storage(), (std::vector<double>{58.0, 139.0, 64.0, 154.0}));
}

TEST(CpuTensorContractionTest, ContractsRankTwoDiagonalComponentsOnEitherSide)
{
  using component_extents_type = stdex::dextents<uni20::index_type, 1>;
  using component_mapping_type = stdex::layout_stride::mapping<component_extents_type>;
  using component_mdspan_type = stdex::mdspan<double, component_extents_type, stdex::layout_stride>;
  using matrix_extents_type = stdex::dextents<uni20::index_type, 2>;

  std::array<double, 5> component_storage{2.0, 0.0, 3.0, 0.0, 4.0};
  auto const component_mapping =
      component_mapping_type{component_extents_type{3}, std::array<uni20::index_type, 1>{-2}};
  component_mdspan_type components{component_storage.data() + 4, component_mapping};

  uni20::Tensor<double, 2> lhs(2, 3);
  lhs[0, 0] = 1.0;
  lhs[0, 1] = 2.0;
  lhs[0, 2] = 3.0;
  lhs[1, 0] = 5.0;
  lhs[1, 1] = 7.0;
  lhs[1, 2] = 11.0;
  auto rhs_diagonal = uni20::make_diagonal_mdspan(matrix_extents_type{3, 4}, components);
  uni20::Tensor<double, 2> right_output(2, 4);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 4; ++column)
      right_output[row, column] = 8.0;

  auto lhs_span = lhs.mdspan();
  auto right_output_span = right_output.mdspan();
  auto const right_axes =
      uni20::linalg::make_contraction_axes<2, 2>(std::array<std::pair<std::size_t, std::size_t>, 1>{{{1, 0}}});
  uni20::linalg::cpu::contract(right_output_span, 2.0, lhs_span, rhs_diagonal, 0.5, right_axes);

  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 4; ++column)
    {
      auto const product = column < 3 ? static_cast<double>(lhs[row, column]) * components[column] : 0.0;
      EXPECT_DOUBLE_EQ((right_output[row, column]), 4.0 + 2.0 * product);
    }
  }

  auto lhs_diagonal = uni20::make_diagonal_mdspan(matrix_extents_type{4, 3}, components);
  uni20::Tensor<double, 2> rhs(3, 2);
  rhs[0, 0] = 2.0;
  rhs[0, 1] = 3.0;
  rhs[1, 0] = 5.0;
  rhs[1, 1] = 7.0;
  rhs[2, 0] = 11.0;
  rhs[2, 1] = 13.0;
  uni20::Tensor<double, 2> left_output(4, 2);
  auto rhs_span = rhs.mdspan();
  auto left_output_span = left_output.mdspan();
  auto const left_axes =
      uni20::linalg::make_contraction_axes<2, 2>(std::array<std::pair<std::size_t, std::size_t>, 1>{{{1, 0}}});
  uni20::linalg::cpu::contract(left_output_span, 1.0, lhs_diagonal, rhs_span, 0.0, left_axes);

  for (uni20::index_type row = 0; row < 4; ++row)
    for (uni20::index_type column = 0; column < 2; ++column)
      EXPECT_DOUBLE_EQ((left_output[row, column]), (row < 3 ? components[row] * rhs[row, column] : 0.0));
}

TEST(CpuTensorContractionTest, EmptyContractedExtentProducesScaledZeroProduct)
{
  uni20::Tensor<double, 2> lhs(2, 0);
  uni20::Tensor<double, 2> rhs(0, 3);
  uni20::Tensor<double, 2> output(2, 3);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      output[row, column] = 7.0;
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};

  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, 3.0, lhs, rhs, axes, 2.0);

  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((output[row, column]), 14.0);
}

TEST(CpuTensorContractionTest, NoUnitStrideOperandFallsThroughToCpuReferenceBackend)
{
  namespace diagnostics = uni20::linalg::dispatch_diagnostics;
  std::vector<diagnostics::event> events;
  diagnostics::scoped_sink capture([&](diagnostics::event const& event) { events.push_back(event); });
  using strided_matrix = uni20::StridedTensor<double, 2>;
  strided_matrix lhs(strided_matrix::extents_type{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  uni20::Tensor<double, 2> rhs(2, 1);
  uni20::Tensor<double, 2> output(2, 1);
  lhs[0, 0] = 2.0;
  lhs[0, 1] = 3.0;
  lhs[1, 0] = 6.0;
  lhs[1, 1] = 7.0;
  rhs[0, 0] = 4.0;
  rhs[1, 0] = 5.0;
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};

  uni20::linalg::contract(output, 1.0, lhs, rhs, axes, 0.0);

  EXPECT_DOUBLE_EQ((output[0, 0]), 23.0);
  EXPECT_DOUBLE_EQ((output[1, 0]), 59.0);
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events[0].operation, "contract");
  ASSERT_TRUE(events[0].selected_backend().has_value());
  EXPECT_EQ(*events[0].selected_backend(), "cpu_reference");
}

TEST(CpuTensorContractionTest, ZeroCoefficientsSkipUnusedReads)
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  uni20::Tensor<double, 1> lhs(1);
  uni20::Tensor<double, 1> rhs(1);
  uni20::ScalarTensor<double> output;
  lhs[0] = nan;
  rhs[0] = nan;
  output[] = nan;
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{0, 0}}};

  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, 0.0, lhs, rhs, axes, 0.0);

  EXPECT_DOUBLE_EQ(output[], 0.0);
}

TEST(CpuTensorContractionTest, RejectsMismatchedPairedAndOutputExtents)
{
  uni20::Tensor<double, 2> lhs(2, 3);
  uni20::Tensor<double, 2> mismatched_rhs(4, 2);
  uni20::Tensor<double, 2> rhs(3, 2);
  uni20::Tensor<double, 2> wrong_output(1, 1);
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};
  ErrorModeGuard const error_mode;

  EXPECT_THROW(
      uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, wrong_output, 1.0, lhs, mismatched_rhs, axes, 0.0),
      std::runtime_error);
  EXPECT_THROW(uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, wrong_output, 1.0, lhs, rhs, axes, 0.0),
               std::runtime_error);
  EXPECT_EQ(wrong_output.extent(0), 1);
  EXPECT_EQ(wrong_output.extent(1), 1);
}

TEST(CpuTensorContractionDispatchTest, ProbesTheNormalizedDescriptorTypes)
{
  uni20::Tensor<double, 2> lhs(2, 3);
  uni20::Tensor<double, 2> rhs(3, 2);
  uni20::Tensor<double, 2> output(2, 2);
  auto axes = uni20::linalg::make_contraction_axes<2, 2>(std::array<std::pair<std::size_t, std::size_t>, 1>{{{1, 0}}});
  auto operation = uni20::linalg::contract_op<2, 2, 1>{.axes = axes};
  auto output_descriptor = uni20::mdspec_of(output);
  auto lhs_descriptor = uni20::mdspec_of(std::as_const(lhs));
  auto rhs_descriptor = uni20::mdspec_of(std::as_const(rhs));

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, operation, output_descriptor,
                                                 1.0, lhs_descriptor, rhs_descriptor, 0.0),
            uni20::linalg::KernelTypeAcceptance::yes);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, operation, output_descriptor,
                                                 1.0, lhs, rhs, 0.0),
            uni20::linalg::KernelTypeAcceptance::no);
}
