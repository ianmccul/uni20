#pragma once

/**
 * \file elementwise_functions.hpp
 * \ingroup linalg
 * \brief Named backend-neutral function objects for elementwise transforms.
 */

#include <uni20/core/compiler_attributes.hpp>

#include <type_traits>
#include <utility>

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

/// \brief Multiply one element value by a retained scalar factor.
/// \details The factor is part of the operation payload carried through
///          dispatch. Precompiled backends explicitly register supported
///          factor and element-type combinations.
template <class Factor> struct scale
{
    Factor factor;

    template <class Value>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Value value) const noexcept(noexcept(factor * value))
        -> decltype(factor * value)
    {
      return factor * value;
    }
};

/// \brief Deduce a scaling operation that owns a decayed factor value.
template <class Factor> scale(Factor) -> scale<std::decay_t<Factor>>;

/// \brief Return the sum of two element values.
struct add
{
    template <class Left, class Right>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Left left, Right right) const
        noexcept(noexcept(left + right)) -> decltype(left + right)
    {
      return left + right;
    }
};

} // namespace uni20::linalg
