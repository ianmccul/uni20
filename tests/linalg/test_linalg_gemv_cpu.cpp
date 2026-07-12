#include <uni20/common/mdspan.hpp>
#include <uni20/linalg/ops/gemv.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
using extents_1d = stdex::dextents<uni20::index_type, 1>;
using extents_2d = stdex::dextents<uni20::index_type, 2>;

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

TEST(CpuGemvBackendTest, SkipsProductReadsWhenAlphaIsZero)
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> matrix_storage(4, nan);
  std::vector<double> input_storage(2, nan);
  std::vector<double> output_storage{2.0, -3.0};

  stdex::mdspan<double, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> output(output_storage.data(), 2);

  uni20::linalg::dispatch_kernel(uni20::linalg::CpuReferenceBackend{}, uni20::linalg::gemv_op{}, output, 0.0, matrix,
                                 input, 2.0);
  EXPECT_DOUBLE_EQ(output[0], 4.0);
  EXPECT_DOUBLE_EQ(output[1], -6.0);
}

TEST(CpuGemvBackendTest, TensorOperandsAcceptExplicitSelector)
{
  using matrix_type = uni20::Tensor<double, 2>;
  using vector_type = uni20::Tensor<double, 1>;

  matrix_type matrix(matrix_type::extents_type{2, 2});
  vector_type input(vector_type::extents_type{2});
  vector_type output(vector_type::extents_type{2});
  fill_matrix(matrix, {1.0, 2.0, 3.0, 4.0});
  fill_vector(input, {2.0, 3.0});

  uni20::linalg::gemv(uni20::linalg::CpuReferenceBackend{}, output, 1.0, matrix, input, 0.0);
  EXPECT_DOUBLE_EQ(output[0], 8.0);
  EXPECT_DOUBLE_EQ(output[1], 18.0);
}

#if !UNI20_BACKEND_BLAS
TEST(CpuGemvBackendTest, TensorOperandsUseCpuStorageDefaultWithoutBlas)
{
  using matrix_type = uni20::Tensor<double, 2>;
  using vector_type = uni20::Tensor<double, 1>;

  matrix_type matrix(matrix_type::extents_type{2, 2});
  vector_type input(vector_type::extents_type{2});
  vector_type output(vector_type::extents_type{2});
  fill_matrix(matrix, {1.0, 2.0, 3.0, 4.0});
  fill_vector(input, {2.0, 3.0});

  uni20::linalg::gemv(output, 1.0, matrix, input, 0.0);
  EXPECT_DOUBLE_EQ(output[0], 8.0);
  EXPECT_DOUBLE_EQ(output[1], 18.0);
}
#endif
