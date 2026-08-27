#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{
using generated_matrix = decltype(uni20::ones<double>(2, 3));

struct RecordingGenerator
{
    std::vector<std::array<uni20::index_type, 2>>* visits;

    template <class Indices> [[nodiscard]] int operator()(Indices const& indices) const
    {
      visits->push_back(indices);
      return static_cast<int>(indices[0] * 10 + indices[1]);
    }
};

static_assert(uni20::RankedImmediateTensorView<generated_matrix, 2>);
static_assert(!uni20::RankedStridedImmediateTensorView<generated_matrix, 2>);
static_assert(!uni20::MutableImmediateTensorView<generated_matrix>);
static_assert(!uni20::OwningTensor<generated_matrix>);
static_assert(std::same_as<typename generated_matrix::storage_policy, uni20::GeneratedStorage>);
static_assert(std::same_as<typename generated_matrix::layout_type, uni20::GeneratedLayout>);
} // namespace

TEST(TensorGeneratedTest, ConstantFactoriesGenerateValuesWithoutDenseStorage)
{
  auto zeros = uni20::zeros<double>(2, 3);
  auto ones = uni20::ones<int>(2, 3);
  auto full = uni20::full(uni20::complex<double>{2.0, -3.0}, 2, 3);

  EXPECT_DOUBLE_EQ((zeros[1, 2]), 0.0);
  EXPECT_EQ((ones[0, 1]), 1);
  EXPECT_EQ((full[1, 0]), (uni20::complex<double>{2.0, -3.0}));
}

TEST(TensorGeneratedTest, EyeUsesAllEqualIndexDefinitionAtEveryRank)
{
  auto scalar = uni20::eye<double>();
  auto vector = uni20::eye<double>(4);
  auto matrix = uni20::eye<double>(2, 3);
  auto rank_three = uni20::eye<double>(2, 3, 4);

  EXPECT_DOUBLE_EQ(scalar.operator[](), 1.0);
  EXPECT_DOUBLE_EQ(vector[3], 1.0);
  EXPECT_DOUBLE_EQ((matrix[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((matrix[1, 1]), 1.0);
  EXPECT_DOUBLE_EQ((matrix[0, 2]), 0.0);
  EXPECT_DOUBLE_EQ((rank_three[1, 1, 1]), 1.0);
  EXPECT_DOUBLE_EQ((rank_three[1, 1, 2]), 0.0);
}

TEST(TensorGeneratedTest, GeneratedStorageDefersBackendSelectionToConcreteStorage)
{
  uni20::Tensor<double, 2> output(2, 2);
  auto generated = uni20::ones<double>(2, 2);

  auto selector = uni20::linalg::select_backend(uni20::linalg::copy_op{}, output, generated);
  auto generated_first = uni20::linalg::select_backend(uni20::linalg::copy_op{}, generated, output);
  static_assert(std::same_as<decltype(selector), typename uni20::HostStorage::backend_selector_type>);
  static_assert(std::same_as<decltype(generated_first), typename uni20::HostStorage::backend_selector_type>);
}

TEST(TensorGeneratedTest, GeneratedAdaptorUsesStorageFallbackWhenNoConcreteOperandExists)
{
  auto generated = uni20::ones<uni20::complex<double>>(2, 2);
  auto conjugated = uni20::conj(generated);

  auto selector = uni20::linalg::select_backend(uni20::linalg::copy_op{}, conjugated);
  static_assert(std::same_as<decltype(selector), typename uni20::GeneratedStorage::backend_selector_type>);
}

TEST(TensorGeneratedTest, MakeTensorMaterializesGeneratedValues)
{
  auto materialized = uni20::make_tensor(uni20::full(2.5, 2, 3));

  static_assert(uni20::OwningTensor<decltype(materialized)>);
  static_assert(std::same_as<typename decltype(materialized)::layout_type, uni20::ColumnMajor>);
  EXPECT_EQ(materialized.extent(0), 2);
  EXPECT_EQ(materialized.extent(1), 3);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type col = 0; col < 3; ++col)
      EXPECT_DOUBLE_EQ((materialized[row, col]), 2.5);
}

TEST(TensorGeneratedTest, CtadDefaultsLayoutNeutralInputToColumnMajorAlias)
{
  auto inferred = uni20::Tensor(uni20::eye<int>(2, 3));
  auto named = uni20::ColumnMajorTensor(uni20::eye<int>(2, 3));

  using expected_type = uni20::ColumnMajorTensor<int, 2>;
  static_assert(std::same_as<decltype(inferred), expected_type>);
  static_assert(std::same_as<decltype(named), expected_type>);
  EXPECT_EQ((inferred[0, 0]), 1);
  EXPECT_EQ((inferred[1, 2]), 0);
  EXPECT_EQ((named[1, 1]), 1);
}

TEST(TensorGeneratedTest, MakeTensorAcceptsExplicitGeneratedDestinationLayout)
{
  auto materialized = uni20::make_tensor<uni20::RowMajor>(uni20::eye<int>(2, 3));

  static_assert(std::same_as<typename decltype(materialized)::layout_type, uni20::RowMajor>);
  EXPECT_EQ(materialized.mapping().stride(0), 3);
  EXPECT_EQ(materialized.mapping().stride(1), 1);
  EXPECT_EQ((materialized[0, 0]), 1);
  EXPECT_EQ((materialized[1, 1]), 1);
  EXPECT_EQ((materialized[1, 2]), 0);
}

TEST(TensorGeneratedTest, DefaultMaterializationTraversesColumnMajorDestinationOrder)
{
  using extents_type = stdex::dextents<uni20::index_type, 2>;
  std::vector<std::array<uni20::index_type, 2>> visits;
  uni20::GeneratedTensor<int, extents_type, RecordingGenerator> generated(extents_type{2, 3},
                                                                          RecordingGenerator{&visits});

  auto materialized = uni20::make_tensor(generated);

  std::vector<std::array<uni20::index_type, 2>> const expected{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}};
  EXPECT_EQ(visits, expected);
  EXPECT_EQ((materialized[1, 2]), 12);
}

TEST(TensorGeneratedTest, MatrixProductAcceptsGeneratedEye)
{
  uni20::DenseMatrix<double> input(2, 2);
  uni20::DenseMatrix<double> output;
  input[0, 0] = 1.0;
  input[0, 1] = 2.0;
  input[1, 0] = 3.0;
  input[1, 1] = 4.0;

  uni20::linalg::assign_product(output, uni20::eye<double>(2, 2), input);

  EXPECT_DOUBLE_EQ((output[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((output[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((output[1, 0]), 3.0);
  EXPECT_DOUBLE_EQ((output[1, 1]), 4.0);
}

TEST(TensorGeneratedTest, EighMaterializesGeneratedInputWorkspace)
{
  auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(uni20::ones<double>(3, 3));

  EXPECT_NEAR(eigenvalues[0], 0.0, 1e-12);
  EXPECT_NEAR(eigenvalues[1], 0.0, 1e-12);
  EXPECT_NEAR(eigenvalues[2], 3.0, 1e-12);
  EXPECT_EQ(eigenvectors.rows(), 3);
  EXPECT_EQ(eigenvectors.cols(), 3);
}
