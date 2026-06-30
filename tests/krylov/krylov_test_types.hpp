#pragma once

#include <uni20/config.hpp>
#include <uni20/core/types.hpp>

#include <gtest/gtest.h>

namespace uni20::krylov::test
{

using KrylovRealTestTypes = ::testing::Types<float, double>;
using KrylovComplexTestTypes = ::testing::Types<uni20::complex<float>, uni20::complex<double>>;
using KrylovScalarTestTypes = ::testing::Types<float, double, uni20::complex<float>, uni20::complex<double>>;

} // namespace uni20::krylov::test
