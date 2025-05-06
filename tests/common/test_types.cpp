#include "common/types.hpp"
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
  static_assert(uni20::Integer<int>);
  static_assert(uni20::Integer<unsigned int>);
  static_assert(!uni20::Integer<bool>);
  static_assert(!uni20::Integer<float>);
}

TEST(ConceptTest, RealConcept)
{
  static_assert(uni20::Real<float>);
  static_assert(uni20::Real<double>);
  static_assert(!uni20::Real<std::complex<float>>);
}

TEST(ConceptTest, ComplexConcept)
{
  static_assert(uni20::Complex<std::complex<float>>);
  static_assert(uni20::Complex<std::complex<double>>);
  static_assert(!uni20::Complex<float>);
}

TEST(ConceptTest, RealOrComplexConcept)
{
  static_assert(uni20::RealOrComplex<float>);
  static_assert(uni20::RealOrComplex<std::complex<float>>);
  static_assert(!uni20::RealOrComplex<int>);
}

TEST(ConceptTest, NumericConcept)
{
  static_assert(uni20::Numeric<int>);
  static_assert(uni20::Numeric<float>);
  static_assert(uni20::Numeric<std::complex<double>>);
  static_assert(!uni20::Numeric<bool>);
  static_assert(!uni20::Numeric<std::string>);
}

TEST(ConceptTest, BlasRealConcept)
{
  static_assert(uni20::BlasReal<float>);
  static_assert(uni20::BlasReal<double>);
  static_assert(!uni20::BlasReal<long double>);
}

TEST(ConceptTest, BlasComplexConcept)
{
  static_assert(uni20::BlasComplex<uni20::cfloat>);
  static_assert(uni20::BlasComplex<uni20::cdouble>);
  static_assert(!uni20::BlasComplex<std::complex<long double>>);
}

TEST(ConceptTest, BlasScalarConcept)
{
  static_assert(uni20::BlasScalar<float>);
  static_assert(uni20::BlasScalar<uni20::cfloat>);
  static_assert(!uni20::BlasScalar<int>);
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
  EXPECT_FALSE(uni20::HasScalarType<std::vector<int>>);
  EXPECT_FALSE(uni20::HasScalarType<std::vector<std::vector<int>>>);
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
  EXPECT_TRUE(uni20::HasNumericType<std::vector<int>>);
  EXPECT_TRUE(uni20::HasNumericType<std::vector<std::vector<int>>>);
  EXPECT_FALSE(uni20::HasNumericType<std::string>);
  EXPECT_FALSE(uni20::HasNumericType<std::vector<std::string>>);
  EXPECT_FALSE(uni20::HasNumericType<std::vector<std::vector<std::string>>>);
}
