#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/gemv.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/mdspan/transform_view.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using extents_1d = stdex::dextents<uni20::index_type, 1>;
using extents_2d = stdex::dextents<uni20::index_type, 2>;

struct DoubleValue
{
    template <class Scalar> constexpr auto operator()(Scalar value) const { return Scalar{2} * value; }
};

using host_backend_selector =
    uni20::linalg::backend_list<uni20::linalg::BlasBackend, uni20::linalg::CpuReferenceBackend>;

template <class BackendSelector, uni20::MutableRankedTensorView<1> OutputTensor, class Scalar,
          uni20::RankedTensorView<2> MatrixTensor, uni20::RankedTensorView<1> InputTensor>
[[nodiscard]] auto probe_normalized_gemv(BackendSelector&& selector, OutputTensor& output, Scalar alpha,
                                         MatrixTensor const& matrix, InputTensor const& input, Scalar beta)
{
  auto output_span = uni20::mdspec_of(output);
  auto matrix_span = uni20::mdspec_of(matrix);
  auto input_span = uni20::mdspec_of(input);
  return uni20::linalg::probe_dispatch_kernel(std::forward<BackendSelector>(selector), uni20::linalg::gemv_op{},
                                              output_span, alpha, matrix_span, input_span, beta);
}

template <class BackendSelector, uni20::MutableRankedTensorView<1> OutputTensor, class Scalar,
          uni20::RankedTensorView<2> MatrixTensor, uni20::RankedTensorView<1> InputTensor>
[[nodiscard]] bool try_normalized_gemv(BackendSelector&& selector, OutputTensor& output, Scalar alpha,
                                       MatrixTensor const& matrix, InputTensor const& input, Scalar beta)
{
  auto output_span = uni20::mdspec_of(output);
  auto matrix_span = uni20::mdspec_of(matrix);
  auto input_span = uni20::mdspec_of(input);
  return uni20::linalg::try_dispatch_kernel(std::forward<BackendSelector>(selector), uni20::linalg::gemv_op{},
                                            output_span, alpha, matrix_span, input_span, beta);
}

template <class Scalar> class ValueTransformMatrixView {
  public:
    using value_type = std::remove_cv_t<Scalar>;
    using extents_type = extents_2d;
    using index_type = typename extents_type::index_type;
    using base_mdspan_type = stdex::mdspan<value_type const, extents_type, stdex::layout_left>;
    using mdspan_type = decltype(uni20::transform_view(DoubleValue{}, std::declval<base_mdspan_type const&>()));

    ValueTransformMatrixView(value_type const* data, index_type rows, index_type cols)
        : span_(uni20::transform_view(DoubleValue{}, base_mdspan_type{data, rows, cols}))
    {}

    [[nodiscard]] auto mdspan() const noexcept -> mdspan_type { return span_; }
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> host_backend_selector
    {
      return host_backend_selector{uni20::linalg::BlasBackend{}, uni20::linalg::CpuReferenceBackend{}};
    }
    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return span_.extents(); }
    [[nodiscard]] auto extent(std::size_t axis) const noexcept -> index_type { return span_.extent(axis); }

  private:
    mdspan_type span_;
};

template <class Matrix>
void fill_matrix(Matrix&& matrix, std::initializer_list<typename std::remove_reference_t<Matrix>::value_type> values)
{
  auto value = values.begin();
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.extent(1); ++col)
    {
      matrix[row, col] = *value;
      ++value;
    }
  }
}

template <class Vector>
void fill_vector(Vector&& vector, std::initializer_list<typename std::remove_reference_t<Vector>::value_type> values)
{
  auto value = values.begin();
  for (uni20::index_type index = 0; index < vector.extent(0); ++index)
  {
    vector[index] = *value;
    ++value;
  }
}
} // namespace

TEST(LinalgGemvDispatchTest, TypeProbeSeparatesDirectBlasAndAccessorRespectingCpu)
{
  std::vector<double> matrix_storage(4);
  ValueTransformMatrixView<double> matrix(matrix_storage.data(), 2, 2);
  uni20::Tensor<double, 1> input(2);
  uni20::Tensor<double, 1> output(2);

  EXPECT_EQ(probe_normalized_gemv(uni20::linalg::BlasBackend{}, output, 1.0, matrix, input, 0.0),
            uni20::linalg::KernelTypeAcceptance::no);
  EXPECT_EQ(probe_normalized_gemv(uni20::linalg::CpuReferenceBackend{}, output, 1.0, matrix, input, 0.0),
            uni20::linalg::KernelTypeAcceptance::yes);
}

TEST(LinalgGemvDispatchTest, FallsBackForConjugatingInputVector)
{
  using Scalar = uni20::complex<double>;
  uni20::Tensor<Scalar, 2> matrix(typename uni20::Tensor<Scalar, 2>::extents_type{2, 2});
  uni20::Tensor<Scalar, 1> input(typename uni20::Tensor<Scalar, 1>::extents_type{2});
  uni20::Tensor<Scalar, 1> output(typename uni20::Tensor<Scalar, 1>::extents_type{2});
  fill_matrix(matrix, {Scalar{1.0, 1.0}, Scalar{2.0, 0.0}, Scalar{3.0, 0.0}, Scalar{4.0, -1.0}});
  fill_vector(input, {Scalar{1.0, 1.0}, Scalar{2.0, -1.0}});
  fill_vector(output, {Scalar{7.0, 0.0}, Scalar{7.0, 0.0}});

  auto selector = uni20::linalg::backend_list{uni20::linalg::BlasBackend{}, uni20::linalg::CpuReferenceBackend{}};
  auto conjugated_input = uni20::conj(input);
  EXPECT_TRUE(try_normalized_gemv(selector, output, Scalar{1.0}, matrix, conjugated_input, Scalar{}));
  EXPECT_EQ(output[0], (Scalar{6.0, 2.0}));
  EXPECT_EQ(output[1], (Scalar{12.0, -1.0}));
}

TEST(LinalgGemvDispatchTest, BlasOnlyDeclinePreservesOutput)
{
  using Scalar = uni20::complex<double>;
  uni20::Tensor<Scalar, 2> matrix(typename uni20::Tensor<Scalar, 2>::extents_type{2, 2});
  uni20::Tensor<Scalar, 1> input(typename uni20::Tensor<Scalar, 1>::extents_type{2});
  uni20::Tensor<Scalar, 1> output(typename uni20::Tensor<Scalar, 1>::extents_type{2});
  fill_matrix(matrix, {Scalar{1.0}, Scalar{1.0}, Scalar{1.0}, Scalar{1.0}});
  fill_vector(input, {Scalar{1.0, 1.0}, Scalar{1.0, 1.0}});
  fill_vector(output, {Scalar{7.0}, Scalar{7.0}});

  auto conjugated_input = uni20::conj(input);
  EXPECT_FALSE(
      try_normalized_gemv(uni20::linalg::BlasBackend{}, output, Scalar{1.0}, matrix, conjugated_input, Scalar{}));
  EXPECT_EQ(output[0], Scalar{7.0});
  EXPECT_EQ(output[1], Scalar{7.0});
}

TEST(LinalgGemvDispatchTest, NormalizedMdspansAreKernelDispatchOperands)
{
  std::vector<double> matrix_storage(4);
  std::vector<double> input_storage(2);
  std::vector<double> output_storage(2);
  stdex::mdspan<double const, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<double const, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> output(output_storage.data(), 2);
  auto selector = host_backend_selector{uni20::linalg::BlasBackend{}, uni20::linalg::CpuReferenceBackend{}};

  auto candidates =
      uni20::linalg::kernel_type_candidates(selector, uni20::linalg::gemv_op{}, output, 1.0, matrix, input, 0.0);
  static_assert(std::same_as<decltype(candidates), host_backend_selector>);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(selector, uni20::linalg::gemv_op{}, output, 1.0, matrix, input, 0.0),
            uni20::linalg::KernelTypeAcceptance::yes);
}

TEST(LinalgGemvDispatchTest, TensorOperandsUseStorageDefaultSelector)
{
  using matrix_type = uni20::Tensor<double, 2>;
  using vector_type = uni20::Tensor<double, 1>;

  matrix_type matrix(matrix_type::extents_type{2, 3});
  vector_type input(vector_type::extents_type{3});
  vector_type output(vector_type::extents_type{2});
  fill_matrix(matrix, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_vector(input, {1.0, 2.0, 3.0});
  fill_vector(output, {10.0, 20.0});

  uni20::linalg::gemv(output, 1.0, matrix, input, 1.0);
  EXPECT_DOUBLE_EQ(output[0], 24.0);
  EXPECT_DOUBLE_EQ(output[1], 52.0);
}
