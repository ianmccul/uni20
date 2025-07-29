#pragma once

#include "scalar_concepts.hpp"
#include <complex>
#include <numeric>

namespace uni20
{

// uni20 conj function. This differs from the std:: version by returning a real type for real arguments

template <typename T> inline constexpr bool has_trivial_conj = has_real_scalar<T> || has_integer_scalar<T>;

template <typename T> std::complex<T> conj(std::complex<T> x) { return std::conj(x); }

template <HasRealScalar R> R conj(R const& x) { return x; }

template <HasIntegerScalar I> I conj(I const& x) { return x; }

// uni20 herm function for scalar types
template <HasScalar S> auto herm(S x) { return conj(x); }

} // namespace uni20
