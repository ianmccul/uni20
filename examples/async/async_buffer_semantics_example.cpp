#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <fmt/core.h>
#include <exception>
#include <stdexcept>

using namespace uni20::async;

namespace
{

void print_section(char const* title)
{
  fmt::print("\n=== {} ===\n", title);
}

void print_async_status(char const* name, Async<int>& value)
{
  try
  {
    fmt::print("{} value: {}\n", name, value.get_wait());
  }
  catch (std::exception const& ex)
  {
    fmt::print("{} failed with: {}\n", name, ex.what());
  }
}

} // namespace

int main()
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  print_section("Read/Write ownership and explicit release");
  Async<int> source_value = 10;
  Async<int> transformed_value;
  auto transform = [](ReadBuffer<int> in, WriteBuffer<int> out) static->AsyncTask {
    auto input = co_await in.transfer();
    int const value = input.get();
    input.release();
    co_await out = value + 5;
    co_return;
  }(source_value.read(), transformed_value.write());
  sched.schedule(std::move(transform));
  sched.run_all();
  fmt::print("transformed_value: {}\n", transformed_value.get_wait());

  print_section("First write and accumulation");
  Async<int> accum;
  auto accumulate = [](WriteBuffer<int> out) static->AsyncTask {
    co_await out += 5;
    co_await out += 7;
    co_await out -= 2;
    co_return;
  }(accum.write());
  sched.schedule(std::move(accumulate));
  sched.run_all();
  fmt::print("accum after +=5, +=7, -=2: {}\n", accum.get_wait());

  print_section("Cancellation paths");
  Async<int> cancelled;
  {
    auto writer = cancelled.write();
    writer.release();
  }

  bool maybe_empty = false;
  bool saw_cancel = false;

  auto maybe_probe = [](ReadBuffer<int> in, bool& maybe_empty) static->AsyncTask {
    auto maybe = co_await in.transfer().maybe();
    maybe_empty = !maybe.has_value();
    if (maybe) maybe->release();
    co_return;
  }(cancelled.read(), maybe_empty);
  sched.schedule(std::move(maybe_probe));

  auto cancel_probe = [](ReadBuffer<int> in, bool& saw_cancel) static->AsyncTask {
    try
    {
      auto value = co_await in.transfer().or_cancel();
      value.release();
    }
    catch (task_cancelled const&)
    {
      saw_cancel = true;
    }
    co_return;
  }(cancelled.read(), saw_cancel);
  sched.schedule(std::move(cancel_probe));

  sched.run_all();
  fmt::print("maybe() reported no value: {}\n", maybe_empty ? "true" : "false");
  fmt::print("or_cancel() threw task_cancelled: {}\n", saw_cancel ? "true" : "false");

  print_section("Unhandled exception routing");
  Async<int> source;
  Async<int> sink_auto;
  Async<int> sink_explicit;

  auto fail_source = [](WriteBuffer<int> source_writer) static->AsyncTask {
    (void)source_writer;
    throw std::runtime_error("source failure");
    co_return;
  }(source.write());
  sched.schedule(std::move(fail_source));

  auto route_failures = [](ReadBuffer<int> in, WriteBuffer<int> auto_sink, WriteBuffer<int> explicit_sink) static->AsyncTask {
    co_await propagate_exceptions_to(explicit_sink);
    (void)co_await in;
    co_await auto_sink = 0;
    co_return;
  }(source.read(), sink_auto.write(), sink_explicit.write());
  sched.schedule(std::move(route_failures));

  sched.run_all();
  print_async_status("sink_auto", sink_auto);
  print_async_status("sink_explicit", sink_explicit);

  return 0;
}
