#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/gemv.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <initializer_list>
#include <type_traits>
#include <vector>

namespace
{
using extents_1d = stdex::dextents<uni20::index_type, 1>;
using extents_2d = stdex::dextents<uni20::index_type, 2>;

template <class Scalar> struct ValueTransformAccessor
{
    using element_type = Scalar;
    using data_handle_type = Scalar*;
    using reference = Scalar;
    using offset_policy = ValueTransformAccessor;

    constexpr data_handle_type offset(data_handle_type ptr, std::size_t offset) const { return ptr + offset; }
    constexpr reference access(data_handle_type ptr, std::size_t offset) const { return Scalar{2} * ptr[offset]; }
};

template <class Scalar>
using transformed_matrix = stdex::mdspan<Scalar, extents_2d, stdex::layout_left, ValueTransformAccessor<Scalar>>;

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
  std::vector<double> input_storage(2);
  std::vector<double> output_storage(2);

  transformed_matrix<double> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> output(output_storage.data(), 2);

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::BlasBackend{}, uni20::linalg::gemv_op{}, output, 1.0,
                                                 matrix, input, 0.0),
            uni20::linalg::KernelTypeAcceptance::no);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemv_op{}, output,
                                                 1.0, matrix, input, 0.0),
            uni20::linalg::KernelTypeAcceptance::yes);
}

TEST(LinalgGemvDispatchTest, FallsBackForConjugatingInputVector)
{
  using Scalar = uni20::complex<double>;
  std::vector<Scalar> matrix_storage(4);
  std::vector<Scalar> input_storage{Scalar{1.0, 1.0}, Scalar{2.0, -1.0}};
  std::vector<Scalar> output_storage(2, Scalar{7.0, 0.0});

  stdex::mdspan<Scalar, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> output(output_storage.data(), 2);
  fill_matrix(matrix, {Scalar{1.0, 1.0}, Scalar{2.0, 0.0}, Scalar{3.0, 0.0}, Scalar{4.0, -1.0}});

  auto selector = uni20::linalg::backend_list{uni20::linalg::BlasBackend{}, uni20::linalg::CpuReferenceBackend{}};
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(selector, uni20::linalg::gemv_op{}, output, Scalar{1.0}, matrix,
                                                 uni20::conj(input), Scalar{}));
  EXPECT_EQ(output[0], (Scalar{6.0, 2.0}));
  EXPECT_EQ(output[1], (Scalar{12.0, -1.0}));
}

TEST(LinalgGemvDispatchTest, BlasOnlyDeclinePreservesOutput)
{
  using Scalar = uni20::complex<double>;
  std::vector<Scalar> matrix_storage(4, Scalar{1.0});
  std::vector<Scalar> input_storage(2, Scalar{1.0, 1.0});
  std::vector<Scalar> output_storage(2, Scalar{7.0});

  stdex::mdspan<Scalar, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> output(output_storage.data(), 2);

  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(uni20::linalg::BlasBackend{}, uni20::linalg::gemv_op{}, output,
                                                  Scalar{1.0}, matrix, uni20::conj(input), Scalar{}));
  EXPECT_EQ(output[0], Scalar{7.0});
  EXPECT_EQ(output[1], Scalar{7.0});
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
