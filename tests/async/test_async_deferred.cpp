#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/debug_scheduler.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace uni20;
using namespace uni20::async;

TEST(AsyncDeferredTest, InitializesAfterScheduling)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<std::vector<int>> data(std::vector<int>{1, 2, 3, 4});
  Async<bool> view_consumed(false);

  // schedule a task that modifies the data
  sched.schedule([](WriteBuffer<std::vector<int>> b) static->AsyncTask {
    auto& writer = co_await b;
    writer = std::vector<int>{3, 4, 5, 6, 7, 8, 9, 10};
  }(data.write()));

  int view_element;

  {
    // create a view
    Async<int const*> view;
    sched.schedule([](ReadBuffer<std::vector<int>> r, WriteBuffer<int const*> view) static->AsyncTask {
      auto const& vec = co_await r;
      co_await view.emplace(vec.data());
    }(data.read(), view.write()));

    // read the data via the view
    sched.schedule([](ReadBuffer<int const*> r, int& v, WriteBuffer<bool> ready) static->AsyncTask {
      v = (co_await r)[0];
      auto& consumed = co_await ready;
      consumed = true;
    }(view.read(), view_element, view_consumed.write()));
  }

  // schedule a task that modifies the data again
  sched.schedule([](ReadBuffer<bool> ready, WriteBuffer<std::vector<int>> b) static->AsyncTask {
    co_await ready;
    auto& writer = co_await b;
    writer.resize(1024);
    writer[0] = 5;
  }(view_consumed.read(), data.write()));

  sched.run_all();
  EXPECT_EQ(view_element, 3);
}

namespace
{

struct TrackingView
{
    TrackingView(std::shared_ptr<std::vector<std::string>> log, int const* ptr, int id)
        : log_(std::move(log)), ptr_(ptr), id_(id)
    {
      log_->push_back("construct " + std::to_string(id_));
    }

    TrackingView(TrackingView const& other) : log_(other.log_), ptr_(other.ptr_), id_(other.id_)
    {
      log_->push_back("copy " + std::to_string(id_));
    }

    TrackingView(TrackingView&& other) noexcept : log_(std::move(other.log_)), ptr_(other.ptr_), id_(other.id_)
    {
      log_->push_back("move " + std::to_string(id_));
    }

    ~TrackingView() { log_->push_back("destroy " + std::to_string(id_)); }

    int value() const { return *ptr_; }

  private:
    std::shared_ptr<std::vector<std::string>> log_;
    int const* ptr_;
    int id_;
};

struct MutableTrackingView
{
    MutableTrackingView(std::shared_ptr<std::vector<std::string>> log, int* ptr, int id)
        : log_(std::move(log)), ptr_(ptr), id_(id)
    {
      log_->push_back("construct " + std::to_string(id_));
    }

    ~MutableTrackingView() { log_->push_back("destroy " + std::to_string(id_)); }

    void set_value(int value) const
    {
      log_->push_back("mutate");
      *ptr_ = value;
    }

    int value() const { return *ptr_; }

  private:
    std::shared_ptr<std::vector<std::string>> log_;
    int* ptr_;
    int id_;
};

} // namespace

TEST(AsyncDeferredTest, NonTrivialViewConstructsAndDestroysInOrder)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  auto log = std::make_shared<std::vector<std::string>>();
  Async<bool> view_consumed(false);

  Async<std::vector<int>> data(std::vector<int>{1, 2, 3});
  int observed_value = 0;

  {
    Async<TrackingView> view;

    sched.schedule(
        [](WriteBuffer<std::vector<int>> b, std::shared_ptr<std::vector<std::string>> log) static->AsyncTask {
          log->push_back("write start");
          auto& writer = co_await b;
          writer[0] = 7;
          log->push_back("write done");
        }(data.write(), log));

    sched.schedule([](ReadBuffer<std::vector<int>> r, WriteBuffer<TrackingView> v,
                      std::shared_ptr<std::vector<std::string>> log) static->AsyncTask {
      auto const& vec = co_await r;
      auto view_log = log;
      co_await v.emplace(std::move(view_log), vec.data(), 1);
      log->push_back("emplace done");
    }(data.read(), view.write(), log));

    sched.schedule([](ReadBuffer<TrackingView> r, int& result, std::shared_ptr<std::vector<std::string>> log,
                      WriteBuffer<bool> ready_signal) static->AsyncTask {
      auto const& view = co_await r;
      result = view.value();
      log->push_back("consume");
      auto& ready = co_await ready_signal;
      ready = true;
    }(view.read(), observed_value, log, view_consumed.write()));

    sched.schedule([](ReadBuffer<bool> ready, WriteBuffer<std::vector<int>> b,
                      std::shared_ptr<std::vector<std::string>> log) static->AsyncTask {
      co_await ready;
      log->push_back("post-write start");
      auto& writer = co_await b;
      writer[1] = 9;
      log->push_back("post-write done");
    }(view_consumed.read(), data.write(), log));

    sched.run_all();
  }

  std::vector<std::string> expected{"write start", "write done",       "construct 1",     "emplace done",
                                    "consume",     "post-write start", "post-write done", "destroy 1"};

  EXPECT_EQ(observed_value, 7);
  EXPECT_EQ(*log, expected);
}

TEST(AsyncDeferredTest, MutableViewCanModifyUnderlyingData)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  auto log = std::make_shared<std::vector<std::string>>();
  Async<bool> view_consumed(false);

  Async<std::vector<int>> data(std::vector<int>{4, 5, 6});
  int observed_value = 0;

  {
    Async<MutableTrackingView> view;

    sched.schedule([](WriteBuffer<std::vector<int>> b, WriteBuffer<MutableTrackingView> v,
                      std::shared_ptr<std::vector<std::string>> log) static->AsyncTask {
      log->push_back("emplace start");
      auto& vec = co_await b;
      auto view_log = log;
      co_await v.emplace(std::move(view_log), vec.data(), 2);
      log->push_back("emplace done");
    }(data.write(), view.write(), log));

    sched.schedule([](ReadBuffer<MutableTrackingView> r, int& observed, std::shared_ptr<std::vector<std::string>> log,
                      WriteBuffer<bool> ready) static->AsyncTask {
      auto const& view = co_await r;
      view.set_value(11);
      observed = view.value();
      log->push_back("consume");
      auto& ready_flag = co_await ready;
      ready_flag = true;
    }(view.read(), observed_value, log, view_consumed.write()));

    sched.schedule([](ReadBuffer<bool> ready, ReadBuffer<std::vector<int>> data_view,
                      std::shared_ptr<std::vector<std::string>> log) static->AsyncTask {
      co_await ready;
      auto const& vec = co_await data_view;
      log->push_back("post-read");
      EXPECT_EQ(vec[0], 11);
    }(view_consumed.read(), data.read(), log));

    sched.run_all();
  }

  std::vector<std::string> expected{"emplace start", "construct 2", "emplace done", "mutate",
                                    "consume",       "post-read",   "destroy 2"};

  EXPECT_EQ(observed_value, 11);
  EXPECT_EQ(*log, expected);
}
