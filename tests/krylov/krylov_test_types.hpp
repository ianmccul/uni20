#pragma once

#include <uni20/config.hpp>
#include <uni20/core/types.hpp>

#include <gtest/gtest.h>

namespace uni20::krylov::test
{

#if UNI20_HAS_FLOAT128
using KrylovRealTestTypes = ::testing::Types<float, double, uni20::float128>;
using KrylovComplexTestTypes =
    ::testing::Types<uni20::complex<float>, uni20::complex<double>, uni20::complex<uni20::float128>>;
using KrylovScalarTestTypes = ::testing::Types<float, double, uni20::float128, uni20::complex<float>,
                                               uni20::complex<double>, uni20::complex<uni20::float128>>;
#else
using KrylovRealTestTypes = ::testing::Types<float, double>;
using KrylovComplexTestTypes = ::testing::Types<uni20::complex<float>, uni20::complex<double>>;
using KrylovScalarTestTypes = ::testing::Types<float, double, uni20::complex<float>, uni20::complex<double>>;
#endif

} // namespace uni20::krylov::test
