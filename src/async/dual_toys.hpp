#pragma once

#include "async_toys.hpp"
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

template <typename T> Dual<T> sin(Dual<T> x)
{
  Dual<T> Result;

  schedule(co_sin(x.value.read(), Result.value.write()));

  schedule([](ReadBuffer<T> in, ReadBuffer<T> in_grad, WriteBuffer<T> out_grad) -> AsyncTask {
    using std::cos;
    auto in_g = co_await in_grad.or_cancel();
    auto cos_x = cos(co_await in);
    co_await out_grad += cos_x * in_g;
  }(x.value.read(), Result.grad.input(), x.grad.output()));

  return Result;
}

template <typename T> Dual<T> cos(Dual<T> x)
{
  Dual<T> Result;
  Result.value = cos(x.value);
  x.grad -= sin(x.value) * Result.grad;
  return Result;
}

template <typename T> Dual<T> operator*(Dual<T> x, Dual<T> y)
{
  Dual<T> Result;
  Result.value = x.value * y.value;
  x.grad += Result.grad * y.value;
  y.grad += x.value * Result.grad;
  return Result;
}

template <typename T> Dual<T> operator-(Dual<T> x, T y)
{
  Dual<T> Result;
  Result.value = x.value - y;
  x.grad += 1.0 * Result.grad;
  return Result;
}

template <typename T> Dual<T> operator*(T x, Dual<T> y)
{
  Dual<T> Result;
  Result.value = x * y.value;
  y.grad += x * Result.grad;
  return Result;
}

} // namespace uni20::async
