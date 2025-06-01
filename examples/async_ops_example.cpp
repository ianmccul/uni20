#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/debug_scheduler.hpp"
#include <fmt/core.h>

using namespace uni20::async;

template <typename T> Async<T> branch_dag_static(int mode, Async<T> const& a, Async<T> const& b, Async<T> const& c)
{
  return (mode == 1) ? a + b * c : (a + b) * c;
}

template <typename T>
Async<T> branch_dag_dynamic_wait(Async<int> const& mode, Async<T> const& a, Async<T> const& b, Async<T> const& c)
{
  return (mode.get_wait() == 1) ? a + b * c : (a + b) * c;
}

template <typename T>
Async<T> branch_dag_dynamic(Async<int> const& mode, Async<T> const& a, Async<T> const& b, Async<T> const& c)
{
  Async<T> out;

  schedule([](auto m, auto x, auto y, auto z, auto out_) -> AsyncTask {
    fmt::print("Entering coroutine\n");

    auto [mode_val, av, bv, cv] = co_await all(m, x, y, z); // suspend the coroutine until all values are available

    T result = (mode_val == 1) ? av + bv * cv : (av + bv) * cv;

    co_await out_ = std::move(result);
    co_return;
  }(mode.read(), a.read(), b.read(), c.read(), out.write()));
  return out;
}

int main()
{
  DebugScheduler sched;
  set_global_scheduler(&sched);

  sched.block(); // prevent the scheduler from running

  Async<int> a = 2;     // a has a value that can be read immediately
  Async<int> b = a + 1; // schedule computation of a+1; b cannot be read until that completes
  Async<int> c = b + 1; // schedule computation of b+1; c cannot be read until that completes

  auto r1 = branch_dag_static(1, a, b, c); // 2 + 3*4 = 14
  auto r2 = branch_dag_static(2, a, b, c); // (2+3)*4 = 20

  Async<int> mode1 = 1;
  Async<int> mode2 = mode1 + 1;
  auto r3 = branch_dag_dynamic(mode1, a, b, c); // 14
  auto r4 = branch_dag_dynamic(mode2, a, b, c); // 20

  sched.unblock(); // alow the scheduler to run

  auto r5 = branch_dag_dynamic_wait(mode1, a, b, c); // this can run immediately; mode1 has a known value
  auto r6 = branch_dag_dynamic_wait(mode2, a, b, c); // if the scheduler is blocked, reading mode2 will deadlock

  sched.unblock(); // alow the scheduler to run

  fmt::print("Static mode 1: {}\n", r1.get_wait());
  fmt::print("Static mode 2: {}\n", r2.get_wait());
  fmt::print("Dynamic mode 1: {}\n", r3.get_wait());
  fmt::print("Dynamic mode 2: {}\n", r4.get_wait());
  fmt::print("Dynamic wait mode 1: {}\n", r5.get_wait());
  fmt::print("Dynamic wait mode 2: {}\n", r6.get_wait());

  return 0;
}
