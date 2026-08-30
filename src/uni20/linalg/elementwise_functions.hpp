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

/// \brief Nullary elementwise function that returns one retained value.
/// \tparam Value Stored value type.
template <class Value> struct constant
{
    [[no_unique_address]] Value value;

    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()() const -> Value { return value; }
};

template <class Value> constant(Value) -> constant<std::decay_t<Value>>;

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

/// \brief Return the product of one element value with itself.
struct square
{
    template <class Value>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Value value) const noexcept(noexcept(value * value))
        -> decltype(value * value)
    {
      return value * value;
    }
};

/// \brief Return the multiplicative inverse of one element value.
struct reciprocal
{
    template <class Value>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Value value) const noexcept(noexcept(Value{1} / value))
        -> decltype(Value{1} / value)
    {
      return Value{1} / value;
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

/// \brief Add one retained scalar multiple to an existing element value.
/// \details The first argument is the existing output value and the second is
///          the input value, giving `output + factor * input`.
template <class Factor> struct add_scaled
{
    Factor factor;

    template <class Output, class Input>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Output output, Input input) const
        noexcept(noexcept(output + factor * input)) -> decltype(output + factor * input)
    {
      return output + factor * input;
    }
};

/// \brief Deduce an add-scaled operation that owns a decayed factor value.
template <class Factor> add_scaled(Factor) -> add_scaled<std::decay_t<Factor>>;

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

/// \brief Return the difference of two element values.
struct subtract
{
    template <class Left, class Right>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Left left, Right right) const
        noexcept(noexcept(left - right)) -> decltype(left - right)
    {
      return left - right;
    }
};

/// \brief Return the product of two element values.
struct multiply
{
    template <class Left, class Right>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Left left, Right right) const
        noexcept(noexcept(left * right)) -> decltype(left * right)
    {
      return left * right;
    }
};

/// \brief Return the quotient of two element values.
struct divide
{
    template <class Left, class Right>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Left left, Right right) const
        noexcept(noexcept(left / right)) -> decltype(left / right)
    {
      return left / right;
    }
};

} // namespace uni20::linalg
