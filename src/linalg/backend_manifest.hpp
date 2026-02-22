#pragma once

#include "backends/cpu/linalg_cpu.hpp"
#include "backends/cusolver/linalg_cusolver.hpp"
#include "backends/lapack/linalg_lapack.hpp"
#include "tags/cpu.hpp"
#include "tags/cusolver.hpp"
#include "tags/lapack.hpp"

#include <tuple>

namespace uni20::linalg
{
using available_backends = std::tuple<cpu_tag, lapack_tag, cusolver_tag>;

} // namespace uni20::linalg
