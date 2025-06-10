#pragma once

#include "async.hpp"
#include "reverse_value.hpp"
#include "scheduler.hpp"
#include <cmath>

namespace uni20::async
{

template <typename T> class Dual {
  public:
    using value_type = T;

    Dual() = default;
    Dual(Dual&&) = default;
    Dual& operator=(Dual&&) = default;

    Dual(Dual& other) : value(other.value) { other.grad += grad.input(); }

    Dual(Async<T> const& other) : value(other) {}

    Dual& operator=(Dual& other)
    {
      other.grad += grad.grad_output();
      value = other.value;
    }

    /// \brief Constructs with a copy of an initial value that can be implicitly converted to T
    template <typename U>
      requires std::convertible_to<U, T>
    Dual(U&& val) : value(std::forward<U>(val))
    {}

    Async<T> value;
    ReverseValue<T> grad;
};

} // namespace uni20::async
