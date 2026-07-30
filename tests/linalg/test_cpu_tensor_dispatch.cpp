#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/linalg/ops/gemv.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using extents_1d = stdex::dextents<uni20::index_type, 1>;
using backend_selector_type = uni20::linalg::backend_list<uni20::linalg::CpuReferenceBackend>;
using dense_mdspan = stdex::mdspan<double, extents_2d, stdex::layout_left>;
using dense_const_mdspan = stdex::mdspan<double const, extents_2d, stdex::layout_left>;
using dense_vector_mdspan = stdex::mdspan<double, extents_1d, stdex::layout_left>;
using dense_const_vector_mdspan = stdex::mdspan<double const, extents_1d, stdex::layout_left>;

template <class Backend, class OutputSpan, class InputSpan>
concept HasDirectGemmTryKernel =
    requires(Backend backend, OutputSpan& output, InputSpan& lhs, InputSpan& rhs, double scalar) {
      uni20::linalg::try_kernel(backend, uni20::linalg::gemm_op{}, output, scalar, lhs, rhs, scalar);
    };

static_assert(HasDirectGemmTryKernel<uni20::linalg::CpuReferenceBackend, dense_mdspan, dense_const_mdspan>);

template <class Backend, class OutputSpan, class MatrixSpan, class InputSpan>
concept HasDirectGemvTryKernel =
    requires(Backend backend, OutputSpan& output, MatrixSpan& matrix, InputSpan& input, double scalar) {
      uni20::linalg::try_kernel(backend, uni20::linalg::gemv_op{}, output, scalar, matrix, input, scalar);
    };

static_assert(HasDirectGemvTryKernel<uni20::linalg::CpuReferenceBackend, dense_vector_mdspan, dense_const_mdspan,
                                     dense_const_vector_mdspan>);

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

TEST(CpuGemmDispatchTest, NormalizedMdspansAreKernelDispatchOperands)
{
  uni20::DenseMatrix<double> output(2, 2);
  uni20::DenseMatrix<double> lhs(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  auto output_span = output.mdspan();
  auto lhs_span = std::as_const(lhs).mdspan();
  auto rhs_span = std::as_const(rhs).mdspan();

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemm_op{},
                                                 output_span, 1.0, lhs_span, rhs_span, 0.0),
            uni20::linalg::KernelTypeAcceptance::yes);
}

TEST(CpuGemmDispatchTest, DescriptorProbeRejectsIncompatibleResolvedMdspan)
{
  uni20::DenseMatrix<double> output(2, 2);
  std::array<double, 4> lhs_storage{};
  NonConvertibleReadMatrixView lhs(lhs_storage.data(), 2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);

  auto output_span = uni20::device_mdspan_of(output);
  auto lhs_span = uni20::device_mdspan_of(std::as_const(lhs));
  auto rhs_span = uni20::device_mdspan_of(std::as_const(rhs));
  auto const descriptor_acceptance = uni20::linalg::probe_dispatch_kernel(
      uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemm_op{}, output_span, 1.0, lhs_span, rhs_span, 0.0);

  EXPECT_EQ(descriptor_acceptance, uni20::linalg::KernelTypeAcceptance::no);
}

TEST(CpuAssignProductDispatchTest, RetainsTensorOutputAndNormalizesFixedInputs)
{
  uni20::DenseMatrix<double> output(2, 2);
  uni20::DenseMatrix<double> lhs(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  auto lhs_span = uni20::device_mdspan_of(std::as_const(lhs));
  auto rhs_span = uni20::device_mdspan_of(std::as_const(rhs));

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{},
                                                 uni20::linalg::assign_product_op{}, output, 1.0, lhs_span, rhs_span),
            uni20::linalg::KernelTypeAcceptance::yes);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{},
                                                 uni20::linalg::assign_product_op{}, output, 1.0, lhs, rhs),
            uni20::linalg::KernelTypeAcceptance::no);
}

TEST(CpuGemvDispatchTest, NormalizedMdspansAreKernelDispatchOperands)
{
  uni20::Tensor<double, 1> output(2);
  uni20::DenseMatrix<double> matrix(2, 2);
  uni20::Tensor<double, 1> input(2);
  auto output_span = output.mdspan();
  auto matrix_span = std::as_const(matrix).mdspan();
  auto input_span = std::as_const(input).mdspan();

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemv_op{},
                                                 output_span, 1.0, matrix_span, input_span, 0.0),
            uni20::linalg::KernelTypeAcceptance::yes);
}

TEST(CpuGemvDispatchTest, DescriptorProbeRejectsIncompatibleResolvedMdspan)
{
  uni20::Tensor<double, 1> output(2);
  uni20::DenseMatrix<double> matrix(2, 2);
  std::array<double, 2> input_storage{};
  NonConvertibleReadVectorView input(input_storage.data(), 2);

  auto output_span = uni20::device_mdspan_of(output);
  auto matrix_span = uni20::device_mdspan_of(std::as_const(matrix));
  auto input_span = uni20::device_mdspan_of(std::as_const(input));
  auto const descriptor_acceptance = uni20::linalg::probe_dispatch_kernel(
      uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemv_op{}, output_span, 1.0, matrix_span, input_span, 0.0);

  EXPECT_EQ(descriptor_acceptance, uni20::linalg::KernelTypeAcceptance::no);
}
