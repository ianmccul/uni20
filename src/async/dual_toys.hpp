#pragma once

#include "async_toys.hpp"
#include "core/math.hpp"
#include "dual.hpp"

namespace uni20::async
{

//
// Demonstration trig fuctions for Dual<T>
//
// sin(x) is implemented as coroutines
//
// cos(x) is implemented using basic operations on Dual<T> and Async<T>
//

// Chain rule: ∂L/∂z* = ∂L/∂f* ⋅ ∂f*/∂z* + ∂L/∂f ⋅ ∂f/∂z*
// Since L is real-valued, ∂L/∂f = conj(∂L/∂f*)
//
// We can rewrite this as
// ∂L/∂z* = ∂L/∂f* ⋅ conj(∂f/∂z) + conj(∂L/∂f) ⋅ ∂f/∂z*
// out_grad += in_grad . conj(∂f/∂z) + conj(in_grad) . ∂f/∂z*

template <typename T> Dual<T> sin(Dual<T>& x)
{
  Dual<T> Result;

  schedule(co_sin(x.value.read(), Result.value.write()));

  // TRACE("Dual Sin", Result.grad.value().queue().latest().get());
  schedule([](ReadBuffer<T> in, ReadBuffer<T> in_grad, WriteBuffer<T> out_grad) static->AsyncTask {
    using std::cos;
    using uni20::conj;
    TRACE("Dual Sin coroutine");
    auto in_g = co_await in_grad.or_cancel();
    TRACE("Dual Sin", in_g);
    auto cos_x = cos(co_await in);
    TRACE("Dual Sin", cos_x);
    co_await out_grad += conj(cos_x) * in_g;
    TRACE("Dual Sin finished");
  }(x.value.read(), Result.grad.input(), x.grad.output()));

  return Result;
}

template <typename T> Dual<T> sin(Dual<T>&& x)
{
  Dual<T> Result;

  schedule(co_sin(x.value.read(), Result.value.write()));

  // TRACE("Dual Sin", Result.grad.value().queue().latest().get());
  schedule([](ReadBuffer<T> in, ReadBuffer<T> in_grad, WriteBuffer<T> out_grad) static->AsyncTask {
    using std::cos;
    using uni20::conj;
    TRACE("Dual Sin coroutine");
    auto in_g = co_await in_grad.or_cancel();
    TRACE("Dual Sin", in_g);
    auto cos_x = cos(co_await in);
    TRACE("Dual Sin", cos_x);
    co_await out_grad += conj(cos_x) * in_g;
    TRACE("Dual Sin finished");
  }(x.value.read(), Result.grad.input(), x.grad.output()));

  return Result;
}

template <typename T> Dual<T> cos(Dual<T> x)
{
  Dual<T> Result;
  Result.value = cos(x.value);
  x.grad -= conj(sin(x.value)) * Result.grad;
  return Result;
}

template <typename T> Dual<T> operator-(Dual<T> x, T y)
{
  Dual<T> Result;
  Result.value = x.value - y;
  x.grad += Result.grad;
  return Result;
}

template <typename T> Dual<T> operator-(T x, Dual<T> y)
{
  Dual<T> Result;
  Result.value = x - y.value;
  y.grad -= Result.grad;
  return Result;
}

template <typename T> Dual<T> operator-(Dual<T> x, Dual<T> y)
{
  Dual<T> Result;
  Result.value = x.value - y.value;
  x.grad += Result.grad;
  y.grad -= Result.grad;
  return Result;
}

template <typename T> Dual<T> operator+(Dual<T> x, T y)
{
  Dual<T> Result;
  Result.value = x.value + y;
  x.grad += Result.grad;
  return Result;
}

template <typename T> Dual<T> operator+(T x, Dual<T> y)
{
  Dual<T> Result;
  Result.value = x + y.value;
  y.grad += Result.grad;
  return Result;
}
template <typename T> Dual<T> operator+(Dual<T> x, Dual<T> y)
{
  Dual<T> Result;
  Result.value = x.value + y.value;
  x.grad += Result.grad;
  y.grad += Result.grad;
  return Result;
}

template <typename T> Dual<T> operator*(T x, Dual<T> y)
{
  using uni20::herm;
  Dual<T> Result;
  Result.value = x * y.value;
  y.grad += herm(x) * Result.grad;
  return Result;
}

template <typename T> Dual<T> operator*(Dual<T> x, T y)
{
  using uni20::herm;
  Dual<T> Result;
  Result.value = x.value * y;
  x.grad += Result.grad * herm(y);
  return Result;
}

template <typename T> Dual<T> operator*(Dual<T> x, Dual<T> y)
{
  Dual<T> Result;
  Result.value = x.value * y.value;
  x.grad += Result.grad * herm(y.value);
  y.grad += herm(x.value) * Result.grad;
  return Result;
}

template <typename T> Dual<uni20::make_real_t<T>> real(Dual<T> z)
{
  using uni20::real;
  using r_type = uni20::make_real_t<T>;
  Dual<r_type> Result;
  Result.value = real(z.value);
  schedule([](ReadBuffer<r_type> in_grad, WriteBuffer<T> out_grad) static->AsyncTask {
    real(co_await out_grad) += co_await in_grad.or_cancel();
  }(Result.grad.input(), z.grad.output()));
  return Result;
}

template <typename T> Dual<uni20::make_real_t<T>> imag(Dual<T> z)
{
  using uni20::imag;
  using r_type = uni20::make_real_t<T>;
  Dual<r_type> Result;
  Result.value = imag(z.value);
  schedule([](ReadBuffer<r_type> in_grad, WriteBuffer<T> out_grad) static->AsyncTask {
    imag(co_await out_grad) += co_await in_grad.or_cancel();
  }(Result.grad.input(), z.grad.output()));
  return Result;
}

} // namespace uni20::async
