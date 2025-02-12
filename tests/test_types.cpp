#include "types.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

// ----------------------------------------------------------------------------
// Traits Tests
// ----------------------------------------------------------------------------
TEST(TraitsTest, IsReal)
{
  EXPECT_TRUE(uni20::is_real<float>);
  EXPECT_TRUE(uni20::is_real<double>);
  EXPECT_TRUE(uni20::is_real<long double>);
  EXPECT_FALSE(uni20::is_real<std::complex<float>>);
}

TEST(TraitsTest, IsComplex)
{
  EXPECT_FALSE(uni20::is_complex<float>);
  EXPECT_TRUE(uni20::is_complex<std::complex<float>>);
  EXPECT_TRUE(uni20::is_complex<std::complex<double>>);
}

// ----------------------------------------------------------------------------
// Concepts Tests (using static_asserts for compile-time checks)
// ----------------------------------------------------------------------------
TEST(ConceptTest, RealConcept)
{
  static_assert(uni20::real<float>);
  static_assert(uni20::real<double>);
  static_assert(!uni20::real<std::complex<float>>);
}

TEST(ConceptTest, ComplexConcept)
{
  static_assert(uni20::complex<std::complex<float>>);
  static_assert(uni20::complex<std::complex<double>>);
  static_assert(!uni20::complex<float>);
}

TEST(ConceptTest, ScalarConcept)
{
  static_assert(uni20::scalar<float>);
  static_assert(uni20::scalar<std::complex<float>>);
  // Note: integral types are not considered scalar in our definition.
  static_assert(!uni20::scalar<int>);
}

TEST(ConceptTest, BlasRealConcept)
{
  static_assert(uni20::blas_real<float>);
  static_assert(uni20::blas_real<double>);
  static_assert(!uni20::blas_real<long double>);
}

TEST(ConceptTest, BlasComplexConcept)
{
  static_assert(uni20::blas_complex<uni20::cfloat>);
  static_assert(uni20::blas_complex<uni20::cdouble>);
  static_assert(!uni20::blas_complex<std::complex<long double>>);
}

TEST(ConceptTest, BlasScalarConcept)
{
  static_assert(uni20::blas_scalar<float>);
  static_assert(uni20::blas_scalar<uni20::cfloat>);
  static_assert(!uni20::blas_scalar<int>);
}

// ----------------------------------------------------------------------------
// make_real Tests
// ----------------------------------------------------------------------------
TEST(MakeRealTest, RealType)
{
  // For a real type, make_real_type<T> should simply be T.
  static_assert(std::is_same_v<uni20::make_real_type<float>, float>);
  static_assert(std::is_same_v<uni20::make_real_type<double>, double>);
}

TEST(MakeRealTest, ComplexType)
{
  // For a complex type, make_real_type<T> should extract the underlying type.
  static_assert(std::is_same_v<uni20::make_real_type<std::complex<float>>, float>);
  static_assert(std::is_same_v<uni20::make_real_type<std::complex<double>>, double>);
}

// ----------------------------------------------------------------------------
// make_complex Tests
// ----------------------------------------------------------------------------
TEST(MakeComplexTest, RealType)
{
  // For a floating-point type, make_complex_type<T> should be std::complex<T>.
  static_assert(std::is_same_v<uni20::make_complex_type<float>, std::complex<float>>);
  static_assert(std::is_same_v<uni20::make_complex_type<double>, std::complex<double>>);
}

TEST(MakeComplexTest, ComplexType)
{
  // For a type already recognized as complex, make_complex_type<T> should be T.
  static_assert(std::is_same_v<uni20::make_complex_type<std::complex<float>>, std::complex<float>>);
  static_assert(std::is_same_v<uni20::make_complex_type<std::complex<double>>, std::complex<double>>);
}

// ----------------------------------------------------------------------------
// scalar_type Tests
// ----------------------------------------------------------------------------
TEST(ScalarTypeTest, DirectScalar)
{
  // For a scalar type, scalar_type<T>::type should be T.
  static_assert(std::is_same_v<uni20::scalar_type<float>::type, float>);
  static_assert(std::is_same_v<uni20::scalar_type<std::complex<float>>::type, std::complex<float>>);
}

TEST(ScalarTypeTest, NestedContainer)
{
  // For a container type (e.g., std::vector) that has a nested value_type,
  // scalar_type should recursively unwrap to the underlying type.
  using NestedContainer = std::vector<std::vector<double>>;
  static_assert(std::is_same_v<uni20::scalar_type<NestedContainer>::type, double>);
}
