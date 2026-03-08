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
  // The owning proxy from `co_await std::move(in)` transfers ownership of the read buffer.
  // We explicitly call `release()` immediately after `get()`, before write-buffer acquisition.
  // This ordering matters because it is possible that
  // there are data dependencies that mean that we cannot get the WriteBuffer until after the Read
  // has finished.  For example consider
  // Async<double> x = 1.0;
  // auto rbuf = x.read();
  // x *= 2;
  // auto wbuf = x.write();
  // schedule(co_sin(std::move(rbuf), std::move(wbuf)));
  // GCC 13 limitation: `(co_await std::move(in)).get()` can fail with
  // "insufficient contextual information to determine type" in templates.
  // This is fixed in GCC 14+, where the one-liner below is preferred:
  //   auto const input = (co_await std::move(in)).get();
  auto input_buffer = co_await std::move(in);
  auto const input = input_buffer.get();
  input_buffer.release();
  co_await out = sin(input);
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
  auto input_buffer = co_await std::move(in);
  auto const input = input_buffer.get();
  input_buffer.release();
  co_await out = cos(input);
}

template <typename T> Async<T> cos(Async<T> const& x)
{
  Async<T> Result;
  schedule(co_cos(x.read(), Result.write()));
  return Result;
}

template <typename T> AsyncTask co_conj(ReadBuffer<T> in, WriteBuffer<T> out)
{
  auto input_buffer = co_await std::move(in);
  auto const input = input_buffer.get();
  input_buffer.release();
  co_await out = uni20::conj(input);
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
  auto input_buffer = co_await std::move(in);
  auto const input = input_buffer.get();
  input_buffer.release();
  co_await out = real(input);
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
  auto input_buffer = co_await std::move(in);
  auto const input = input_buffer.get();
  input_buffer.release();
  co_await out = imag(input);
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
  auto input_buffer = co_await std::move(in);
  auto const input = input_buffer.get();
  input_buffer.release();
  co_await out = herm(input);
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
