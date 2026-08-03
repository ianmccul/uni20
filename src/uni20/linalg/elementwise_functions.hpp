#pragma once

/**
 * \file elementwise_functions.hpp
 * \ingroup linalg
 * \brief Named backend-neutral function objects for elementwise transforms.
 */

#include <uni20/core/compiler_attributes.hpp>

namespace uni20::linalg
{

/// \brief Return the additive inverse of one element value.
/// \details The named type permits precompiled backends to register an explicit
///          lowering while the same object remains usable by generic host code.
struct negate
{
    template <class Value>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Value value) const noexcept(noexcept(-value))
        -> decltype(-value)
    {
      return -value;
    }
};

} // namespace uni20::linalg
