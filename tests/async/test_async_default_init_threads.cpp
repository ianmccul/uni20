#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

namespace
{
struct Counting
{
    Counting() { ++constructed; }
    Counting(Counting const&) { ++constructed; }
    Counting& operator=(Counting const&) = default;

    static std::atomic<int> constructed;
};

std::atomic<int> Counting::constructed{0};
} // namespace

TEST(AsyncDefaultInit, InitializesOnceAcrossThreads)
{
  using namespace uni20::async;

  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Counting::constructed.store(0);
  Async<Counting> value;
  EXPECT_EQ(Counting::constructed.load(), 0);

  sched.schedule([](WriteBuffer<Counting> buffer) static->AsyncTask {
    auto& initialized = co_await buffer.emplace();
    (void)initialized;
    co_return;
  }(value.write()));
  sched.run_all();
  EXPECT_EQ(Counting::constructed.load(), 1);

  constexpr int kThreads = 16;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int i = 0; i < kThreads; ++i)
  {
    threads.emplace_back([&value]() {
      auto reader = value.read();
      reader.get_wait();
    });
  }

  for (auto& thread : threads)
    thread.join();

  EXPECT_EQ(Counting::constructed.load(), 1);
}
