#pragma once

// \brief Var type for reverse-mode automatic differentiation using Wirtinger calculus.
///
/// This represents the reverse-mode adjoint for functions of complex variables,
/// assuming a **real-valued scalar loss** function \( L \colon \mathbb{C} \to \mathbb{R} \).
///
/// Only the Wirtinger derivative with respect to \( z^* \) is propagated:
/// \f[
///   \bar{z} := \frac{\partial L}{\partial z^*}
/// \f]
///
/// For an intermediate variable \( f(z) \), the chain rule is:
/// \f[
///   \frac{\partial L}{\partial z^*} =
///     \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial z^*}
///   + \frac{\partial L}{\partial f^*} \cdot \frac{\partial f^*}{\partial z^*}
/// \f]
///
/// Under the real-valued loss assumption:
/// \f[
///   \frac{\partial L}{\partial f^*} = \overline{ \left( \frac{\partial L}{\partial f} \right) },
///   \quad
///   \frac{\partial f^*}{\partial z^*} = \overline{ \left( \frac{\partial f}{\partial z} \right) }
/// \f]
///
/// so the full gradient becomes:
/// \f[
///   \frac{\partial L}{\partial z^*} =
///     \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial z^*}
///   + \overline{ \left( \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial z} \right) }
/// \f]
///
/// \note This implementation assumes the upstream gradient is available via `grad`, and all
/// reverse-mode propagation targets the Wirtinger \( \partial / \partial z^* \) direction.

#include "async.hpp"
#include "reverse_value.hpp"
#include "scheduler.hpp"
#include <cmath>

namespace uni20::async
{

template <typename T> class Var {
  public:
    using value_type = T;

    Var() = default;
    Var(Var&&) = default;
    Var& operator=(Var&&) = default;

    /// Disable ordinary copying (Vars are never copied in the normal sense).
    Var(Var const&) = delete;
    Var& operator=(Var const&) = delete;

    Var(Var& other) : value(other.value) { other.grad += grad.input(); }

    Var(Async<T> const& other) : value(other) {}

    Var& operator=(Var& other)
    {
      grad = ReverseValue<T>{};
      other.grad += grad.input();
      value = other.value;
      return *this;
    }

    /// \brief Constructs with a copy of an initial value that can be implicitly converted to T
    template <typename U>
    requires std::convertible_to<U, T> Var(U&& val) : value(std::forward<U>(val)) {}

    Async<T> value;
    ReverseValue<T> grad;
};

} // namespace uni20::async
