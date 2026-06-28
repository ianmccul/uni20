#include <gtest/gtest.h>
#include <uni20/core/scalar_concepts.hpp>
#include <vector>

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
  static_assert(!uni20::Real<uni20::complex<float>>);
}

TEST(ConceptTest, ComplexConcept)
{
  static_assert(uni20::Complex<uni20::complex<float>>);
  static_assert(uni20::Complex<uni20::complex<double>>);
  static_assert(!uni20::Complex<float>);
}

TEST(ConceptTest, RealOrComplexConcept)
{
  static_assert(uni20::RealOrComplex<float>);
  static_assert(uni20::RealOrComplex<uni20::complex<float>>);
  static_assert(!uni20::RealOrComplex<int>);
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
  static_assert(!uni20::BlasComplex<uni20::complex<long double>>);
}

TEST(ConceptTest, BlasScalarConcept)
{
  static_assert(uni20::BlasScalar<float>);
  static_assert(uni20::BlasScalar<uni20::cfloat>);
  static_assert(!uni20::BlasScalar<int>);
}

TEST(ConceptTest, LapackRealConcept)
{
  static_assert(uni20::LapackReal<float>);
  static_assert(uni20::LapackReal<double>);
  static_assert(!uni20::LapackReal<long double>);
}

TEST(ConceptTest, LapackComplexConcept)
{
  static_assert(uni20::LapackComplex<uni20::cfloat>);
  static_assert(uni20::LapackComplex<uni20::cdouble>);
  static_assert(!uni20::LapackComplex<uni20::complex<long double>>);
}

TEST(ConceptTest, LapackRealOrComplexConcept)
{
  static_assert(uni20::LapackRealOrComplex<float>);
  static_assert(uni20::LapackRealOrComplex<double>);
  static_assert(uni20::LapackRealOrComplex<uni20::cfloat>);
  static_assert(uni20::LapackRealOrComplex<uni20::cdouble>);
  static_assert(!uni20::LapackRealOrComplex<long double>);
  static_assert(!uni20::LapackRealOrComplex<uni20::complex<long double>>);
}

TEST(ConceptTest, LapackComplexRealConcept)
{
  static_assert(uni20::LapackComplexReal<float>);
  static_assert(uni20::LapackComplexReal<double>);
  static_assert(!uni20::LapackComplexReal<long double>);
}

TEST(ConceptTest, LapackScalarConcept)
{
  static_assert(uni20::LapackScalar<float>);
  static_assert(uni20::LapackScalar<uni20::cfloat>);
  static_assert(!uni20::LapackScalar<int>);
}

TEST(ConceptTest, HasScalarConcept)
{
  static_assert(uni20::HasScalar<std::vector<float>>);
  static_assert(uni20::HasScalar<std::vector<uni20::complex<double>>>);
  static_assert(!uni20::HasScalar<std::vector<std::string>>);
}

TEST(ConceptTest, HasRealScalarConcept)
{
  static_assert(uni20::HasRealScalar<std::vector<double>>);
  static_assert(!uni20::HasRealScalar<std::vector<uni20::complex<double>>>);
}

TEST(ConceptTest, HasComplexScalarConcept)
{
  static_assert(uni20::HasComplexScalar<std::vector<uni20::complex<float>>>);
  static_assert(!uni20::HasComplexScalar<std::vector<float>>);
}
