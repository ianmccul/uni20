#include "types.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <vector>

// ----------------------------------------------------------------------------
// Traits Tests
// ----------------------------------------------------------------------------
TEST(TraitsTest, IsInteger)
{
  EXPECT_TRUE(uni20::is_integer<int>);
  EXPECT_TRUE(uni20::is_integer<unsigned int>);
  EXPECT_FALSE(uni20::is_integer<bool>);
  EXPECT_FALSE(uni20::is_integer<float>);
}

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
// Concepts Tests
// ----------------------------------------------------------------------------
TEST(ConceptTest, IntegerConcept)
{
  static_assert(uni20::integer<int>);
  static_assert(uni20::integer<unsigned int>);
  static_assert(!uni20::integer<bool>);
  static_assert(!uni20::integer<float>);
}

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

TEST(ConceptTest, NumericConcept)
{
  static_assert(uni20::numeric<int>);
  static_assert(uni20::numeric<float>);
  static_assert(uni20::numeric<std::complex<double>>);
  static_assert(!uni20::numeric<bool>);
  static_assert(!uni20::numeric<std::string>);
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

TEST(ScalarTypeTest, NonScalarContainer)
{
  // int is a numeric type, but not a scalar type
  EXPECT_FALSE(uni20::has_scalar_type<std::vector<int>>);
  EXPECT_FALSE(uni20::has_scalar_type<std::vector<std::vector<int>>>);
}

//---------------------------------------------------------------------
// numeric_type Tests
//---------------------------------------------------------------------

// Direct types: numeric_type should simply return the type itself.
TEST(NumericTypeTest, NumericTypeDirect)
{
  static_assert(std::is_same_v<uni20::numeric_type<int>::type, int>);
  static_assert(std::is_same_v<uni20::numeric_type<float>::type, float>);
  static_assert(std::is_same_v<uni20::numeric_type<std::complex<double>>::type, std::complex<double>>);
}

// Nested containers: numeric_type should recursively unwrap to a numeric type.
TEST(NumericTest, NumericTypeNestedContainer)
{
  using VecVecInt = std::vector<std::vector<int>>;
  using VecVecFloat = std::vector<std::vector<float>>;

  static_assert(std::is_same_v<uni20::numeric_type<VecVecInt>::type, int>,
                "numeric_type<vector<vector<int>>> should be int");
  static_assert(std::is_same_v<uni20::numeric_type<VecVecFloat>::type, float>,
                "numeric_type<vector<vector<float>>> should be float");
}

//---------------------------------------------------------------------
// Negative Tests: Non-scalar containers.
// These tests verify that if the final nested type is not scalar, then
// scalar_type<T>::type is not defined.
//---------------------------------------------------------------------
TEST(NumericTypeTest, NonScalarContainer)
{
  // int is a numeric type, but not a scalar type
  EXPECT_TRUE(uni20::has_numeric_type<std::vector<int>>);
  EXPECT_TRUE(uni20::has_numeric_type<std::vector<std::vector<int>>>);
  EXPECT_FALSE(uni20::has_numeric_type<std::string>);
  EXPECT_FALSE(uni20::has_numeric_type<std::vector<std::string>>);
  EXPECT_FALSE(uni20::has_numeric_type<std::vector<std::vector<std::string>>>);
}
