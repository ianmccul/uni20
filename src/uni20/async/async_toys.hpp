#pragma once

#include "async.hpp"
#include "scheduler.hpp"
#include <cmath>
#include <fmt/format.h>
#include <iostream>
#include <uni20/core/math.hpp>

namespace uni20::async
{

template <typename T> AsyncTask co_sin(ReadBuffer<T> in, WriteBuffer<T> out)
{
  using std::sin;
  // The owning proxy from `co_await std::move(in)` transfers ownership of the buffer into the proxy.
  // `get_release()` reads and explicitly releases the buffer before we
  // begin write-buffer acquisition. This ordering matters because it is possible that
  // there are data dependencies that mean that we cannot get the WriteBuffer until after the Read
  // has finished.  For example consider
  // Async<double> x = 1.0;
  // auto rbuf = x.read();
  // x *= 2;
  // auto wbuf = x.write();
  // schedule(co_sin(std::move(rbuf), std::move(wbuf)));
  auto const input = (co_await std::move(in)).get_release();
  co_await out = sin(input);

  // NOTE: the one-liner should work here but gcc-13 gives a compiler error
  // "insufficient contextual information to determine type"
  // This is a compiler limitation that works in gcc-15
  // co_await out = sin((co_await std::move(in)).get_release());
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
  schedule([](std::string Format, ReadBuffer<T> in) static->AsyncTask {
    auto x = co_await in;
    fmt::print(fmt::runtime(Format), x);
  }(std::move(Format), x.read()));
}

template <typename T> void async_read(std::string Prompt, Async<T>& x)
{
  schedule([](std::string Prompt, WriteBuffer<T> out) static->AsyncTask {
    fmt::print("{}", Prompt);
    T value{};
    std::cin >> value;
    co_await out = std::move(value);
  }(std::move(Prompt), x.write()));
}

} // namespace uni20::async
