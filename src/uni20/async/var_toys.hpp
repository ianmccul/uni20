#pragma once

/**
 * \file var_toys.hpp
 * \brief Demonstration reverse-mode helpers built on `Var<T>`.
 */

#include "async_toys.hpp"
#include <uni20/core/math.hpp>
#include "var.hpp"

namespace uni20::async
{

/// \brief Computes `sin(x)` and wires reverse-mode accumulation for the input gradient.
/// \tparam T Value type.
/// \param x Input variable.
/// \return Output variable representing `sin(x)`.
template <typename T> Var<T> sin(Var<T> x)
{
  Var<T> Result;

  schedule(co_sin(x.value.read(), Result.value.write()));

  // TRACE("Var Sin", Result.grad.value().queue().latest().get());
  schedule([](ReadBuffer<T> in, ReadBuffer<T> in_grad, WriteBuffer<T> out_grad) static->AsyncTask {
    using std::cos;
    // GCC 13 workaround for `(co_await ...).get()`: use an explicit owning proxy.
    // GCC 14+ supports the one-liner form, e.g.
    //   auto const in_g = (co_await in_grad.transfer().or_cancel()).get();
    auto in_grad_buffer = co_await in_grad.transfer().or_cancel();
    auto const in_g = in_grad_buffer.get();
    in_grad_buffer.release();

    auto in_buffer = co_await in.transfer();
    auto const cos_x = cos(in_buffer.get());
    in_buffer.release();
    co_await out_grad += uni20::conj(cos_x) * in_g;
  }(x.value.read(), Result.grad.input(), x.grad.output()));

  return Result;
}

/// \brief Computes `cos(x)` and wires reverse-mode accumulation for the input gradient.
/// \tparam T Value type.
/// \param x Input variable.
/// \return Output variable representing `cos(x)`.
template <typename T> Var<T> cos(Var<T> x)
{
  Var<T> Result;
  Result.value = cos(x.value);
  x.grad -= conj(sin(x.value)) * Result.grad;
  return Result;
}

/// \brief Computes `x - y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left input variable.
/// \param y Right scalar operand.
/// \return Output variable representing `x - y`.
template <typename T> Var<T> operator-(Var<T> x, T y)
{
  Var<T> Result;
  Result.value = x.value - y;
  x.grad += Result.grad;
  return Result;
}

/// \brief Computes `x - y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left scalar operand.
/// \param y Right input variable.
/// \return Output variable representing `x - y`.
template <typename T> Var<T> operator-(T x, Var<T> y)
{
  Var<T> Result;
  Result.value = x - y.value;
  y.grad -= Result.grad;
  return Result;
}

/// \brief Computes `x - y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left input variable.
/// \param y Right input variable.
/// \return Output variable representing `x - y`.
template <typename T> Var<T> operator-(Var<T> x, Var<T> y)
{
  Var<T> Result;
  Result.value = x.value - y.value;
  x.grad += Result.grad;
  y.grad -= Result.grad;
  return Result;
}

/// \brief Computes `x + y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left input variable.
/// \param y Right scalar operand.
/// \return Output variable representing `x + y`.
template <typename T> Var<T> operator+(Var<T> x, T y)
{
  Var<T> Result;
  Result.value = x.value + y;
  x.grad += Result.grad;
  return Result;
}

/// \brief Computes `x + y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left scalar operand.
/// \param y Right input variable.
/// \return Output variable representing `x + y`.
template <typename T> Var<T> operator+(T x, Var<T> y)
{
  Var<T> Result;
  Result.value = x + y.value;
  y.grad += Result.grad;
  return Result;
}

/// \brief Computes `x + y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left input variable.
/// \param y Right input variable.
/// \return Output variable representing `x + y`.
template <typename T> Var<T> operator+(Var<T> x, Var<T> y)
{
  Var<T> Result;
  Result.value = x.value + y.value;
  x.grad += Result.grad;
  y.grad += Result.grad;
  return Result;
}

/// \brief Computes `x * y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left scalar operand.
/// \param y Right input variable.
/// \return Output variable representing `x * y`.
template <typename T> Var<T> operator*(T x, Var<T> y)
{
  using uni20::herm;
  Var<T> Result;
  Result.value = x * y.value;
  y.grad += herm(x) * Result.grad;
  return Result;
}

/// \brief Computes `x * y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left input variable.
/// \param y Right scalar operand.
/// \return Output variable representing `x * y`.
template <typename T> Var<T> operator*(Var<T> x, T y)
{
  using uni20::herm;
  Var<T> Result;
  Result.value = x.value * y;
  x.grad += Result.grad * herm(y);
  return Result;
}

/// \brief Computes `x * y` with reverse-mode gradient propagation.
/// \tparam T Value type.
/// \param x Left input variable.
/// \param y Right input variable.
/// \return Output variable representing `x * y`.
template <typename T> Var<T> operator*(Var<T> x, Var<T> y)
{
  Var<T> Result;
  Result.value = x.value * y.value;
  x.grad += Result.grad * herm(y.value);
  y.grad += herm(x.value) * Result.grad;
  return Result;
}

/// \brief Computes the real part and propagates gradient to a complex source.
/// \tparam T Complex-like value type.
/// \param z Input variable.
/// \return Output variable representing `real(z)`.
template <typename T> Var<uni20::make_real_t<T>> real(Var<T> z)
{
  using uni20::real;
  using r_type = uni20::make_real_t<T>;
  Var<r_type> Result;
  Result.value = real(z.value);
  schedule([](ReadBuffer<r_type> in_grad, WriteBuffer<T> out_grad) static->AsyncTask {
    auto in_grad_buffer = co_await in_grad.transfer().or_cancel();
    auto const grad = in_grad_buffer.get();
    in_grad_buffer.release();
    co_await out_grad += grad;
  }(Result.grad.input(), z.grad.output()));
  return Result;
}

/// \brief Computes the imaginary part and propagates gradient to a complex source.
/// \tparam T Complex-like value type.
/// \param z Input variable.
/// \return Output variable representing `imag(z)`.
template <typename T> Var<uni20::make_real_t<T>> imag(Var<T> z)
{
  using uni20::imag;
  using r_type = uni20::make_real_t<T>;
  Var<r_type> Result;
  Result.value = imag(z.value);
  schedule([](ReadBuffer<r_type> in_grad, WriteBuffer<T> out_grad) static->AsyncTask {
    auto in_grad_buffer = co_await in_grad.transfer().or_cancel();
    auto const grad = in_grad_buffer.get();
    in_grad_buffer.release();
    T value{};
    imag(value) = grad;
    co_await out_grad += std::move(value);
  }(Result.grad.input(), z.grad.output()));
  return Result;
}

} // namespace uni20::async
