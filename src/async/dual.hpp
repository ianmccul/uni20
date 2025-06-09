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

template <typename T> Dual<T> sin(Dual<T>& x)
{
  Dual<T> Result;
  schedule([](ReadBuffer<T> in, WriteBuffer<T> out) -> AsyncTask {
    using std::sin;
    auto x = co_await in;
    in.release();
    co_await out = sin(x);
  }(x.value.read(), Result.value.write()));

  schedule([](ReadBuffer<T> in, ReadBuffer<T> in_grad, WriteBuffer<T> out_grad) -> AsyncTask {
    using std::cos;
    auto in_g = co_await in_grad.or_cancel();
    auto cos_x = cos(co_await in);
    co_await out_grad += cos_x * in_g;
  }(x.value.read(), Result.grad.input(), x.grad.output()));

  return Result;
}

template <typename T> Async<T> sin(Async<T> x)
{
  Async<T> Result;
  schedule([](ReadBuffer<T> in, WriteBuffer<T> out) -> AsyncTask {
    using std::sin;
    auto x = co_await in;
    in.release();
    co_await out = sin(x);
  }(x.read(), Result.write()));
  return Result;
}

template <typename T> Async<T> cos(Async<T> x)
{
  Async<T> Result;
  schedule([](ReadBuffer<T> in, WriteBuffer<T> out) -> AsyncTask {
    using std::cos;
    auto x = co_await in;
    in.release();
    co_await out = cos(x);
  }(x.read(), Result.write()));
  return Result;
}

template <typename T> Dual<T> cos(Dual<T>& x)
{
  Dual<T> Result;
  Result.value = cos(x.value);
  x.grad -= sin(x.value) * Result.grad;
  return Result;
}

} // namespace uni20::async
