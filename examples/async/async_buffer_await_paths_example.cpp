#include <fmt/core.h>
#include <string>
#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <utility>

using namespace uni20::async;

namespace
{

void print_section(char const* title) { fmt::print("\n=== {} ===\n", title); }

void run_task(DebugScheduler& scheduler, AsyncTask task)
{
  scheduler.schedule(std::move(task));
  scheduler.run_all();
}

} // namespace

int main()
{
  DebugScheduler scheduler;
  ScopedScheduler scoped(&scheduler);

  print_section("ReadBuffer paths through one reader capability");
  Async<int> readable = 12;
  int borrowed_value = 0;
  int maybe_value = 0;
  int required_value = 0;
  int owned_value = 0;

  run_task(scheduler, [](ReadBuffer<int> reader, int& borrowed, int& maybe, int& required) static -> AsyncTask {
    int const& first = co_await reader;
    borrowed = first;

    int const* optional = co_await reader.maybe();
    maybe = optional == nullptr ? -1 : *optional;

    int const& present = co_await reader.or_cancel();
    required = present;
    co_return;
  }(readable.read(), borrowed_value, maybe_value, required_value));

  run_task(scheduler, [](ReadBuffer<int> reader, int& result) static -> AsyncTask {
    auto owned = co_await reader.transfer();
    result = owned.get_release();
    co_return;
  }(readable.read(), owned_value));

  fmt::print("borrowed={}, maybe()={}, or_cancel()={}, transferred={}\n", borrowed_value, maybe_value, required_value,
             owned_value);

  print_section("WriteBuffer value paths");
  Async<int> mutable_value = 10;

  run_task(scheduler, [](WriteBuffer<int> writer) static -> AsyncTask {
    auto access = co_await writer;
    access.get() += 5;

    // Re-awaiting the buffer stays in the same writer epoch.
    auto same_epoch = co_await writer;
    same_epoch += 2;
    co_return;
  }(mutable_value.write()));
  int const after_borrowed_access = mutable_value.get_wait();

  run_task(scheduler, [](WriteBuffer<int> writer) static -> AsyncTask {
    auto owned = co_await writer.transfer();
    owned.get() *= 2;
    owned.release();
    co_return;
  }(mutable_value.write()));
  int const after_owned_access = mutable_value.get_wait();

  fmt::print("borrowed proxy result={}, owning proxy result={}\n", after_borrowed_access, after_owned_access);

  print_section("WriteBuffer storage paths");
  Async<std::string> stored;

  run_task(scheduler, [](WriteBuffer<std::string> writer) static -> AsyncTask {
    auto& storage = co_await writer.storage();
    storage.emplace("borrowed storage access");
    co_return;
  }(stored.write()));
  std::string const after_borrowed_storage = stored.get_wait();

  run_task(scheduler, [](WriteBuffer<std::string> writer) static -> AsyncTask {
    auto storage = co_await writer.transfer().storage();
    storage->emplace("owning storage access");
    storage.release();
    co_return;
  }(stored.write()));
  std::string const after_owned_storage = stored.get_wait();

  fmt::print("borrowed='{}', owning='{}'\n", after_borrowed_storage, after_owned_storage);

  print_section("WriteBuffer consuming paths");
  Async<std::string> payload = std::string("payload");
  std::string taken_without_release;

  run_task(scheduler, [](WriteBuffer<std::string> writer, std::string& taken) static -> AsyncTask {
    taken = co_await writer.take();
    co_await writer = std::string("reconstructed in the same epoch");
    co_return;
  }(payload.write(), taken_without_release));
  std::string const after_reconstruction = payload.get_wait();

  std::string taken_and_released;
  auto take_task = [](WriteBuffer<std::string> writer, std::string& taken) static -> AsyncTask {
    taken = co_await writer.take_release();
    co_return;
  }(payload.write(), taken_and_released);
  auto next_writer = [](WriteBuffer<std::string> writer) static -> AsyncTask {
    co_await writer = std::string("written by the next epoch");
    co_return;
  }(payload.write());

  scheduler.schedule(std::move(take_task));
  scheduler.schedule(std::move(next_writer));
  scheduler.run_all();
  std::string const after_release = payload.get_wait();

  fmt::print("take()='{}', then '{}'\n", taken_without_release, after_reconstruction);
  fmt::print("take_release()='{}', then '{}'\n", taken_and_released, after_release);

  bool const results_are_expected =
      borrowed_value == 12 && maybe_value == 12 && required_value == 12 && owned_value == 12 &&
      after_borrowed_access == 17 && after_owned_access == 34 && after_borrowed_storage == "borrowed storage access" &&
      after_owned_storage == "owning storage access" && taken_without_release == "payload" &&
      after_reconstruction == "reconstructed in the same epoch" &&
      taken_and_released == "reconstructed in the same epoch" && after_release == "written by the next epoch";
  return results_are_expected ? 0 : 1;
}
