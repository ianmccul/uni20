#include "core/scalar_concepts.hpp"
#include <gtest/gtest.h>
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

TEST(ConceptTest, HasScalarConcept)
{
  static_assert(uni20::HasScalar<std::vector<float>>);
  static_assert(uni20::HasScalar<std::vector<std::complex<double>>>);
  static_assert(!uni20::HasScalar<std::vector<std::string>>);
}

TEST(ConceptTest, HasRealScalarConcept)
{
  static_assert(uni20::HasRealScalar<std::vector<double>>);
  static_assert(!uni20::HasRealScalar<std::vector<std::complex<double>>>);
}

TEST(ConceptTest, HasComplexScalarConcept)
{
  static_assert(uni20::HasComplexScalar<std::vector<std::complex<float>>>);
  static_assert(!uni20::HasComplexScalar<std::vector<float>>);
}
