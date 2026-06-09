#include <uni20/core/scalar_traits.hpp>
#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
// Traits Tests
// ----------------------------------------------------------------------------
TEST(TraitsTest, IsInteger)
{
  EXPECT_TRUE(uni20::is_integer_v<int>);
  EXPECT_TRUE(uni20::is_integer_v<unsigned int>);
  EXPECT_FALSE(uni20::is_integer_v<bool>);
  EXPECT_FALSE(uni20::is_integer_v<float>);
}

TEST(TraitsTest, IsReal)
{
  EXPECT_TRUE(uni20::is_real_v<float>);
  EXPECT_TRUE(uni20::is_real_v<double>);
  EXPECT_TRUE(uni20::is_real_v<long double>);
  EXPECT_FALSE(uni20::is_real_v<std::complex<float>>);
}

TEST(TraitsTest, IsComplex)
{
  EXPECT_FALSE(uni20::is_complex_v<float>);
  EXPECT_TRUE(uni20::is_complex_v<std::complex<float>>);
  EXPECT_TRUE(uni20::is_complex_v<std::complex<double>>);
}

// ----------------------------------------------------------------------------
// make_real Tests
// ----------------------------------------------------------------------------
TEST(MakeRealTest, RealType)
{
  // For a real type, make_real_t<T> should simply be T.
  static_assert(std::is_same_v<uni20::make_real_t<float>, float>);
  static_assert(std::is_same_v<uni20::make_real_t<double>, double>);
}

TEST(MakeRealTest, ComplexType)
{
  // For a complex type, make_real_t<T> should extract the underlying type.
  static_assert(std::is_same_v<uni20::make_real_t<std::complex<float>>, float>);
  static_assert(std::is_same_v<uni20::make_real_t<std::complex<double>>, double>);
}

// ----------------------------------------------------------------------------
// make_complex Tests
// ----------------------------------------------------------------------------
TEST(MakeComplexTest, RealType)
{
  // For a floating-point type, make_complex_t<T> should be std::complex<T>.
  static_assert(std::is_same_v<uni20::make_complex_t<float>, std::complex<float>>);
  static_assert(std::is_same_v<uni20::make_complex_t<double>, std::complex<double>>);
}

TEST(MakeComplexTest, ComplexType)
{
  // For a type already recognized as complex, make_complex_t<T> should be T.
  static_assert(std::is_same_v<uni20::make_complex_t<std::complex<float>>, std::complex<float>>);
  static_assert(std::is_same_v<uni20::make_complex_t<std::complex<double>>, std::complex<double>>);
}

// ----------------------------------------------------------------------------
// scalar_t Tests
// ----------------------------------------------------------------------------
TEST(ScalarTypeTest, DirectScalar)
{
  // For a scalar type, scalar_t<T>::type should be T.
  static_assert(std::is_same_v<uni20::scalar_t<float>, float>);
  static_assert(std::is_same_v<uni20::scalar_t<std::complex<float>>, std::complex<float>>);
}

TEST(ScalarTypeTest, NestedContainer)
{
  // For a container type (e.g., std::vector) that has a nested value_type,
  // scalar_t should recursively unwrap to the underlying type.
  using NestedContainer = std::vector<std::vector<double>>;
  static_assert(std::is_same_v<uni20::scalar_t<NestedContainer>, double>);
}

TEST(ScalarTypeTest, NonScalarContainer)
{
  // char is not a scalar type
  EXPECT_FALSE(uni20::has_scalar_v<std::vector<char>>);
  EXPECT_FALSE(uni20::has_scalar_v<std::vector<std::vector<char>>>);
}

TEST(ScalarTraitTest, HasScalarVariants)
{
  using RealVec = std::vector<double>;
  using ComplexMat = std::vector<std::vector<std::complex<float>>>;
  using NonScalar = std::vector<std::string>;

  static_assert(uni20::has_scalar_v<RealVec>);
  static_assert(uni20::has_real_scalar_v<RealVec>);
  static_assert(!uni20::has_complex_scalar_v<RealVec>);

  static_assert(uni20::has_scalar_v<ComplexMat>);
  static_assert(!uni20::has_real_scalar_v<ComplexMat>);
  static_assert(uni20::has_complex_scalar_v<ComplexMat>);

  static_assert(!uni20::has_scalar_v<NonScalar>);
}

TEST(ScalarTraitTest, MakeRealTandMakeComplexT)
{
  static_assert(std::is_same_v<uni20::make_real_t<uni20::cfloat>, float>);
  static_assert(std::is_same_v<uni20::make_real_t<float>, float>);

  static_assert(std::is_same_v<uni20::make_complex_t<float>, uni20::cfloat>);
  static_assert(std::is_same_v<uni20::make_complex_t<uni20::cfloat>, uni20::cfloat>);
}

TEST(ScalarTraitTest, HasRealOrComplex)
{
  using T1 = std::vector<float>;
  using T2 = std::vector<std::complex<double>>;
  using T3 = std::vector<std::vector<char>>;
  using T4 = std::vector<std::vector<int>>;

  static_assert(uni20::has_real_or_complex_scalar_v<T1>);
  static_assert(uni20::has_real_or_complex_scalar_v<T2>);
  static_assert(!uni20::has_real_or_complex_scalar_v<T3>);
  static_assert(!uni20::has_real_or_complex_scalar_v<T4>);
}
