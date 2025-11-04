#pragma once

#include "async.hpp"
#include "core/math.hpp"
#include "scheduler.hpp"
#include <cmath>
#include <fmt/format.h>
#include <iostream>

namespace uni20::async
{

template <typename T> AsyncTask co_sin(ReadBuffer<T> in, WriteBuffer<T> out)
{
  using std::sin;
  // use release() to make sure we release the input before writing to the output
  // C+17 guarantees order of evaluation
  co_await out = sin(co_await release(in));
}

template <typename T> Async<T> sin(Async<T> const& x)
{
  Async<T> Result;
  schedule(co_sin(x.read(), Result.write()));
  return Result;
}

template <typename T> AsyncTask co_cos(ReadBuffer<T> in, WriteBuffer<T> out)
{
  using std::cos;
  auto x = co_await in;
  in.release();
  co_await out = cos(x);
}

template <typename T> Async<T> cos(Async<T> const& x)
{
  Async<T> Result;
  schedule(co_cos(x.read(), Result.write()));
  return Result;
}

template <typename T> AsyncTask co_conj(ReadBuffer<T> in, WriteBuffer<T> out)
{
  using uni20::conj;
  auto x = co_await in;
  in.release();
  co_await out = conj(x);
}

template <typename T> Async<T> conj(Async<T> const& x)
{
  Async<T> Result;
  schedule(co_conj(x.read(), Result.write()));
  return Result;
}

template <typename T> AsyncTask co_real(ReadBuffer<T> in, WriteBuffer<uni20::make_real_t<T>> out)
{
  using uni20::real;
  auto const& value = co_await in;
  auto const result = real(value);
  in.release();
  co_await out = result;
}

template <typename T> Async<uni20::make_real_t<T>> real(Async<T> const& x)
{
  Async<uni20::make_real_t<T>> Result;
  schedule(co_real(x.read(), Result.write()));
  return Result;
}

template <typename T> AsyncTask co_imag(ReadBuffer<T> in, WriteBuffer<uni20::make_real_t<T>> out)
{
  using uni20::imag;
  auto const& value = co_await in;
  auto const result = imag(value);
  in.release();
  co_await out = result;
}

template <typename T> Async<uni20::make_real_t<T>> imag(Async<T> const& x)
{
  Async<uni20::make_real_t<T>> Result;
  schedule(co_imag(x.read(), Result.write()));
  return Result;
}

template <typename T> AsyncTask co_herm(ReadBuffer<T> in, WriteBuffer<T> out)
{
  using uni20::conj;
  auto x = co_await in;
  in.release();
  co_await out = herm(x);
}

template <typename T> Async<T> herm(Async<T> const& x)
{
  Async<T> Result;
  schedule(co_herm(x.read(), Result.write()));
  return Result;
}

template <typename T> void async_print(std::string Format, Async<T> x)
{
  schedule([](std::string Format, ReadBuffer<T> in) static -> AsyncTask {
    auto x = co_await in;
    fmt::print(fmt::runtime(Format), x);
  }(std::move(Format), x.read()));
}

template <typename T> void async_read(std::string Prompt, Async<T>& x)
{
  schedule([](std::string Prompt, WriteBuffer<T> out) static -> AsyncTask {
    auto& x = co_await out;
    fmt::print("{}", Prompt);
    std::cin >> x;
  }(std::move(Prompt), x.write()));
}

} // namespace uni20::async
