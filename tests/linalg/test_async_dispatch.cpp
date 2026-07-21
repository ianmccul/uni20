#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async/dispatch.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>
#include <utility>

namespace uni20::linalg
{
namespace test_async_dispatch
{

struct AsyncDispatchTestOp
{
    static constexpr std::string_view name = "async_dispatch_test";
};

struct DecliningTaskBackend
{
    static constexpr std::string_view name = "declining_task";
};

struct DeferredTaskBackend
{
    static constexpr std::string_view name = "deferred_task";
};

struct FailingTaskBackend
{
    static constexpr std::string_view name = "failing_task";
};

struct BlockingTestBackend
{
    static constexpr std::string_view name = "blocking_test";
};

consteval auto kernel_accepts_types(DecliningTaskBackend const&, AsyncDispatchTestOp const&, int&)
{
  return kernel_types_maybe;
}

consteval auto kernel_accepts_types(DeferredTaskBackend const&, AsyncDispatchTestOp const&, int&)
{
  return kernel_types_maybe;
}

consteval auto kernel_accepts_types(BlockingTestBackend const&, AsyncDispatchTestOp const&, int&)
{
  return kernel_types_yes;
}

consteval auto kernel_accepts_types(FailingTaskBackend const&, AsyncDispatchTestOp const&, int&, int&)
{
  return kernel_types_maybe;
}

consteval auto kernel_accepts_types(BlockingTestBackend const&, AsyncDispatchTestOp const&, int&, int&)
{
  return kernel_types_yes;
}

KernelAttempt try_kernel(DecliningTaskBackend, AsyncDispatchTestOp const&, int&)
{
  PANIC("coroutine dispatch called the blocking implementation despite a task hook");
}

KernelAttempt try_kernel(DeferredTaskBackend, AsyncDispatchTestOp const&, int&)
{
  PANIC("coroutine dispatch called the blocking implementation despite a task hook");
}

KernelAttempt try_kernel(FailingTaskBackend, AsyncDispatchTestOp const&, int&, int&)
{
  PANIC("coroutine dispatch called the blocking implementation despite a task hook");
}

KernelAttempt try_kernel(BlockingTestBackend, AsyncDispatchTestOp const&, int& value)
{
  value = 17;
  return KernelAttempt::success;
}

KernelAttempt try_kernel(BlockingTestBackend, AsyncDispatchTestOp const&, int& value, int& calls)
{
  ++calls;
  value = 17;
  return KernelAttempt::success;
}

KernelTaskAttempt<async::AsyncTask> try_kernel_task(DecliningTaskBackend, AsyncDispatchTestOp const&, int&)
{
  return KernelTaskAttempt<async::AsyncTask>{KernelAttempt::unsupported_instance};
}

async::AsyncTask set_deferred_value(int& value)
{
  value = 29;
  co_return;
}

KernelTaskAttempt<async::AsyncTask> try_kernel_task(DeferredTaskBackend, AsyncDispatchTestOp const&, int& value)
{
  return KernelTaskAttempt<async::AsyncTask>{set_deferred_value(value)};
}

async::AsyncTask fail_deferred_value(int&, int&)
{
  throw std::runtime_error("deferred kernel failed");
  co_return;
}

KernelTaskAttempt<async::AsyncTask> try_kernel_task(FailingTaskBackend, AsyncDispatchTestOp const&, int& value,
                                                    int& calls)
{
  return KernelTaskAttempt<async::AsyncTask>{fail_deferred_value(value, calls)};
}

template <class Selector> async::AsyncTask run_async_dispatch(Selector selector, int& value)
{
  co_await co_dispatch_kernel(std::move(selector), AsyncDispatchTestOp{}, value);
}

template <class Selector>
async::AsyncTask run_failing_async_dispatch(Selector selector, async::WriteBuffer<int> output, int& blocking_calls)
{
  int& value = co_await output;
  co_await co_dispatch_kernel(std::move(selector), AsyncDispatchTestOp{}, value, blocking_calls);
}

TEST(AsyncDispatchTest, TaskDeclineFallsThroughToBlockingKernel)
{
  async::DebugScheduler scheduler;
  int value = 0;
  auto task = run_async_dispatch(backend_list{DecliningTaskBackend{}, BlockingTestBackend{}}, value);
  scheduler.schedule(std::move(task));
  scheduler.run_all();
  EXPECT_EQ(value, 17);
}

TEST(AsyncDispatchTest, SuccessfulDeferredTaskBypassesBlockingKernel)
{
  async::DebugScheduler scheduler;
  int value = 0;
  auto task = run_async_dispatch(backend_list{DeferredTaskBackend{}, BlockingTestBackend{}}, value);
  scheduler.schedule(std::move(task));
  scheduler.run_all();
  EXPECT_EQ(value, 29);
}

TEST(AsyncDispatchTest, DeferredTaskFailureIsTerminal)
{
  async::DebugScheduler scheduler;
  async::Async<int> output = 0;
  int blocking_calls = 0;
  auto task = run_failing_async_dispatch(backend_list{FailingTaskBackend{}, BlockingTestBackend{}}, output.write(),
                                         blocking_calls);
  scheduler.schedule(std::move(task));
  scheduler.run_all();

  EXPECT_EQ(blocking_calls, 0);
  EXPECT_THROW((void)output.get_wait(scheduler), std::runtime_error);
}

} // namespace test_async_dispatch
} // namespace uni20::linalg
