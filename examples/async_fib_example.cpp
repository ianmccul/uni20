
#define TRACE_DISABLE 1

#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <fmt/core.h>

using namespace uni20::async;

Async<uint64_t> fib(Async<uint64_t> const& in);

AsyncTask co_fib(ReadBuffer<uint64_t> in, WriteBuffer<uint64_t> out)
{
  auto n = co_await in;
  in.release();
  if (n < 2)
  {
    co_await out.emplace(n);
    co_return;
  }
  Async<uint64_t> i = n - 1;
  Async<uint64_t> j = n - 2;
  Async<uint64_t> f = 0;
  f += fib(i);
  f += fib(j);
  auto value = co_await f.read();
  co_await out.emplace(value);
  co_return;
}

Async<uint64_t> fib(Async<uint64_t> const& in)
{
  Async<uint64_t> out;
  schedule(co_fib(in.read(), out.write()));
  return out;
}

int main()
{
  // DebugScheduler sched;
  TbbScheduler sched{4};
  set_global_scheduler(&sched);

  int k = 20;
  Async<uint64_t> n = k;
  uint64_t f = fib(n).get_wait();

  fmt::print("fib({}) = {}\n", k, f);

  return 0;
}
