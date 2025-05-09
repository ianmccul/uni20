#pragma once
#include "async.hpp"
#include "common/trace.hpp"
#include <concepts>
#include <coroutine>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

/// \brief Scheduler interface
struct Scheduler
{
    void schedule(AsyncTask&& H)
    {
      Handles.push_back(std::move(H.h_));
      H.h_ = nullptr;
      H.promise().sched_ = this;
    }

    void run()
    {
      std::vector<std::coroutine_handle<>> HCopy;
      std::swap(Handles, HCopy);
      TRACE("Got some coroutines to resume", HCopy.size());
      for (auto h : HCopy)
      {
        TRACE("running coroutine...");
        h.resume();
      }
    }

    void run_all()
    {
      TRACE(this->done());
      while (!this->done())
      {
        TRACE("running...");
        this->run();
        TRACE(this->done());
      }
    }

    bool done() { return Handles.empty(); }

    std::vector<std::coroutine_handle<>> Handles;
};
