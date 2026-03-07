#pragma once

#include <uni20/linalg/backends/cpu/linalg_cpu.hpp>
#include <uni20/linalg/backends/cusolver/linalg_cusolver.hpp>
#include <uni20/linalg/backends/lapack/linalg_lapack.hpp>
#include <uni20/tags/cpu.hpp>
#include <uni20/tags/cusolver.hpp>
#include <uni20/tags/lapack.hpp>

#include <tuple>

namespace uni20::linalg
{
using available_backends = std::tuple<cpu_tag, lapack_tag, cusolver_tag>;

} // namespace uni20::linalg
