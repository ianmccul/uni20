#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/linalg/ops/gemv.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using backend_selector_type = uni20::linalg::backend_list<uni20::linalg::CpuReferenceBackend>;

struct NonConvertibleReference
{
    double const* data = nullptr;
};

struct NonConvertibleReadAccessor
{
    using element_type = double const;
    using data_handle_type = double const*;
    using reference = NonConvertibleReference;
    using offset_policy = NonConvertibleReadAccessor;

    [[nodiscard]] auto access(data_handle_type data, std::size_t offset) const noexcept -> reference
    {
      return reference{data + offset};
    }

    [[nodiscard]] auto offset(data_handle_type data, std::size_t offset) const noexcept -> data_handle_type
    {
      return data + offset;
    }
};

class NonConvertibleReadMatrixView {
  public:
    using extents_type = extents_2d;
    using index_type = typename extents_type::index_type;
    using mdspan_type = stdex::mdspan<double const, extents_type, stdex::layout_left, NonConvertibleReadAccessor>;

    NonConvertibleReadMatrixView(double const* data, index_type rows, index_type cols) : span_(data, rows, cols) {}

    [[nodiscard]] auto mdspan() const noexcept -> mdspan_type { return span_; }

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{uni20::linalg::CpuReferenceBackend{}};
    }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return span_.extents(); }

    [[nodiscard]] auto extent(std::size_t axis) const noexcept -> index_type { return span_.extent(axis); }

  private:
    mdspan_type span_;
};

class NonConvertibleReadVectorView {
  public:
    using extents_type = stdex::dextents<uni20::index_type, 1>;
    using index_type = typename extents_type::index_type;
    using mdspan_type = stdex::mdspan<double const, extents_type, stdex::layout_left, NonConvertibleReadAccessor>;

    NonConvertibleReadVectorView(double const* data, index_type size) : span_(data, size) {}

    [[nodiscard]] auto mdspan() const noexcept -> mdspan_type { return span_; }

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{uni20::linalg::CpuReferenceBackend{}};
    }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return span_.extents(); }

    [[nodiscard]] auto extent(std::size_t axis) const noexcept -> index_type { return span_.extent(axis); }

  private:
    mdspan_type span_;
};

static_assert(uni20::TensorView<NonConvertibleReadMatrixView>);
static_assert(uni20::RankedDeviceTensorView<NonConvertibleReadMatrixView, 2>);
static_assert(std::same_as<uni20::tensor_element_t<NonConvertibleReadMatrixView>, double>);
static_assert(uni20::TensorView<NonConvertibleReadVectorView>);
static_assert(uni20::RankedDeviceTensorView<NonConvertibleReadVectorView, 1>);
} // namespace

TEST(CpuGemmDispatchTest, TensorProbeUsesResolvedMdspanAcceptance)
{
  uni20::DenseMatrix<double> output(2, 2);
  std::array<double, 4> lhs_storage{};
  NonConvertibleReadMatrixView lhs(lhs_storage.data(), 2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);

  auto output_span = output.mdspan();
  auto lhs_span = lhs.mdspan();
  auto rhs_span = rhs.mdspan();
  auto const mdspan_acceptance = uni20::linalg::probe_dispatch_kernel(
      uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemm_op{}, output_span, 1.0, lhs_span, rhs_span, 0.0);
  auto const tensor_acceptance = uni20::linalg::probe_dispatch_kernel(
      uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemm_op{}, output, 1.0, lhs, rhs, 0.0);

  EXPECT_EQ(mdspan_acceptance, uni20::linalg::KernelTypeAcceptance::no);
  EXPECT_EQ(tensor_acceptance, mdspan_acceptance);
}

TEST(CpuGemvDispatchTest, TensorProbeUsesResolvedMdspanAcceptance)
{
  uni20::Tensor<double, 1> output(2);
  uni20::DenseMatrix<double> matrix(2, 2);
  std::array<double, 2> input_storage{};
  NonConvertibleReadVectorView input(input_storage.data(), 2);

  auto output_span = output.mdspan();
  auto matrix_span = matrix.mdspan();
  auto input_span = input.mdspan();
  auto const mdspan_acceptance = uni20::linalg::probe_dispatch_kernel(
      uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemv_op{}, output_span, 1.0, matrix_span, input_span, 0.0);
  auto const tensor_acceptance = uni20::linalg::probe_dispatch_kernel(
      uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemv_op{}, output, 1.0, matrix, input, 0.0);

  EXPECT_EQ(mdspan_acceptance, uni20::linalg::KernelTypeAcceptance::no);
  EXPECT_EQ(tensor_acceptance, mdspan_acceptance);
}
