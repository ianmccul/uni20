#include <uni20/core/types.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace
{
using complex_type = uni20::complex<double>;
using complex_matrix = uni20::DenseMatrix<complex_type>;
using conjugated_matrix = decltype(uni20::conj(std::declval<complex_matrix&>()));

template <class Tensor>
concept ConjugatesRvalueTensor = requires(Tensor&& tensor) { uni20::conj(std::move(tensor)); };

static_assert(uni20::TensorView<conjugated_matrix>);
static_assert(!uni20::OwningTensor<conjugated_matrix>);
static_assert(!uni20::OwningTensor<uni20::ConstTensorView<complex_matrix>>);
static_assert(!uni20::MutableTensorView<conjugated_matrix>);
static_assert(std::same_as<typename conjugated_matrix::storage_policy, uni20::VectorStorage>);
static_assert(!ConjugatesRvalueTensor<complex_matrix>);
} // namespace

TEST(TensorConjugateTest, ComplexTensorProducesLazyReadOnlyView)
{
  complex_matrix matrix(2, 1);
  matrix[0, 0] = complex_type{1.0, 2.0};
  matrix[1, 0] = complex_type{-3.0, 4.0};

  auto view = uni20::conj(matrix);
  auto span = view.mdspan();

  static_assert(uni20::mdspan_needs_conjugation_v<decltype(span)>);
  EXPECT_EQ((span[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((span[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((matrix[0, 0]), (complex_type{1.0, 2.0}));
}

TEST(TensorConjugateTest, DoubleConjugationReturnsConstBaseTensor)
{
  complex_matrix matrix(1, 1);
  matrix[0, 0] = complex_type{2.0, 3.0};

  auto roundtrip = uni20::conj(uni20::conj(matrix));
  auto triple = uni20::conj(uni20::conj(uni20::conj(matrix)));

  static_assert(std::same_as<decltype(roundtrip), uni20::ConstTensorView<complex_matrix>>);
  static_assert(std::same_as<decltype(triple), uni20::ConjugatedTensorView<complex_matrix>>);
  EXPECT_EQ(std::addressof(roundtrip.base()), std::addressof(matrix));
  static_assert(!uni20::mdspan_needs_conjugation_v<decltype(roundtrip.mdspan())>);
  EXPECT_EQ((triple.mdspan()[0, 0]), (complex_type{2.0, -3.0}));
}

TEST(TensorConjugateTest, RealTensorReturnsConstIdentityView)
{
  uni20::DenseMatrix<double> matrix(1, 1);
  matrix[0, 0] = 3.5;

  decltype(auto) view = uni20::conj(matrix);

  static_assert(std::same_as<decltype(view), uni20::DenseMatrix<double> const&>);
  static_assert(!uni20::MutableTensorView<decltype(view)>);
  EXPECT_EQ(std::addressof(view), std::addressof(matrix));
  EXPECT_EQ((view[0, 0]), 3.5);
}
