#include "async/async.hpp"
#include "async/awaiters.hpp"
#include "async/debug_scheduler.hpp"

using namespace uni20::async;

AsyncTask async_assign(ReadBuffer<int> readBuf, WriteBuffer<int> writeBuf)
{
  // Snapshot-read src, and register as next writer on dst

  TRACE("starting coroutine");
  TRACE("async_assign", &readBuf, &writeBuf);

  // Wait for src to be ready
  auto tin = co_await try_await(readBuf);
  int in;
  if (tin)
    in = *tin;
  else
    in = co_await readBuf;

  //  int in = co_await readBuf;

  TRACE("Got the readBuf");

  // Wait until it's our turn to write
  auto& out = co_await writeBuf;

  TRACE("got the writeBuf");

  // Perform the heavy copy
  TRACE(out, in);
  out = in;

  co_return;
}

AsyncTask async_assign_indirect(ReadBuffer<int> readBuf, WriteBuffer<int> writeBuf)
{
  TRACE("starting async_assign_indirect");
  co_await async_assign(readBuf, dup(writeBuf));
  TRACE("finished async_assign_indirect");
}

template <typename T> AsyncTask async_assign_sum(ReadBuffer<T> a, ReadBuffer<T> b, WriteBuffer<T> out)
{
  TRACE("starting async_assign_sum");

  // auto [va, vb, vout] = co_await all(a, b, out);
  // TRACE("got a, b, out");

  // auto va = co_await a;
  // auto vb = co_await b;

  auto [va, vb] = co_await all(a, b);
  auto& vout = co_await out;

  // auto [va, vb] = co_await all(a, b);

  //  auto [vout] = co_await all(out);

  vout = va + vb;

  TRACE(va, vb, vout);

  co_return;
}

int main()
{
  DebugScheduler sched;

  Async<int> i = 10;
  Async<int> j = 5;
  Async<int> k = 2;

  // sched.schedule(async_assign(i.read(), j.write())); // j = i, but async
  sched.schedule(async_assign(j.read(), k.write())); // k = j, but async

  sched.schedule(async_assign_indirect(k.read(), i.write())); // i = k, but async

  sched.schedule(async_assign_sum(i.read(), j.read(), k.write())); // k = i + j;

  auto kk = k.get_wait(sched);

  //  sched.run_all();

  TRACE(kk);
}
