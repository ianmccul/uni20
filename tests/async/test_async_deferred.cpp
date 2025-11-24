#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>
#include <utility>
#include <vector>

using namespace uni20;
using namespace uni20::async;

TEST(AsyncDeferredTest, InitializesAfterScheduling)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<std::vector<int>> data(std::vector<int>{1, 2, 3, 4});

  // schedule a task that modifies the data
  sched.schedule([](WriteBuffer<std::vector<int>> b) static->AsyncTask {
    auto& writer = co_await b;
    writer = std::vector<int>{3, 4, 5, 6, 7, 8, 9, 10};
  }(data.write()));

  int view_element;

  {
    // create a view
    Async<int const*> view(deferred, data);
    sched.schedule([](ReadBuffer<std::vector<int>> r, EmplaceBuffer<int const*> view) static->AsyncTask {
      auto const& vec = co_await r;
      co_await std::move(view)(vec.data());
    }(data.read(), view.emplace()));

    // read the data via the view
    sched.schedule(
        [](ReadBuffer<int const*> r, int& v) static->AsyncTask { v = (co_await r)[0]; }(view.read(), view_element));
  }

  // schedule a task that modifies the data again
  sched.schedule([](WriteBuffer<std::vector<int>> b) static->AsyncTask {
    auto& writer = co_await b;
    writer.resize(1024);
    writer[0] = 5;
  }(data.write()));

  sched.run_all();
  EXPECT_EQ(view_element, 3);
}
