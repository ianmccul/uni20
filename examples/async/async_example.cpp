#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>

using namespace uni20::async;

AsyncTask async_assign(ReadBuffer<int> read_buffer, WriteBuffer<int> write_buffer)
{
  TRACE("starting coroutine");
  TRACE("async_assign", &read_buffer, &write_buffer);

  auto ready = co_await try_await(read_buffer);
  int input;
  if (ready)
    input = *ready;
  else
    input = co_await read_buffer;

  TRACE("Got the read_buffer");

  // Proxy assignment also constructs an initially empty destination.
  co_await write_buffer = input;

  TRACE("wrote the write_buffer");
  TRACE(input);

  co_return;
}

template <typename T> AsyncTask async_assign_sum(ReadBuffer<T> a, ReadBuffer<T> b, WriteBuffer<T> out)
{
  TRACE("starting async_assign_sum");

  auto [va, vb] = co_await all(a, b);
  T const result = va + vb;
  co_await out = result;

  TRACE(va, vb, result);

  co_return;
}

int main()
{
  DebugScheduler scheduler;

  Async<int> i = 10;
  Async<int> j = 5;
  Async<int> k = 2;

  // scheduler.schedule(async_assign(i.read(), j.write())); // j = i, but async
  scheduler.schedule(async_assign(j.read(), k.write())); // k = j, but async

  scheduler.schedule(async_assign_sum(i.read(), j.read(), k.write())); // k = i + j;

  auto kk = k.get_wait(scheduler);

  TRACE(kk);
}
