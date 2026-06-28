#pragma once

/**
 * \file common.hpp
 * \ingroup backend_lapack
 * \brief Shared helpers for LAPACK backend wrappers.
 */

#include <uni20/config.hpp>

namespace uni20::lapack::detail
{

template <typename> inline constexpr bool dependent_false_v = false;

} // namespace uni20::lapack::detail
