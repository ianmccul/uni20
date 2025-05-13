/// \file async_task.hpp
/// \brief Defines AsyncTask, the fire-and-forget coroutine handle.
/// \ingroup async_core

#pragma once

#include <atomic>
#include <coroutine>
#include <exception>

namespace uni20::async
{

/// \brief A fire-and-forget coroutine handle.
/// \ingroup async_core
struct AsyncTask
{
    /// \brief Promise type for AsyncTask, forward declare
    struct promise_type;

    std::coroutine_handle<promise_type> h_; ///< Underlying coroutine handle.

    /// \brief Construct from a coroutine handle.
    /// \param h The coroutine handle.
    explicit AsyncTask(std::coroutine_handle<promise_type> h) noexcept : h_{h} {}

    /// \brief Destroy the coroutine if still present.
    ~AsyncTask()
    {
      if (h_)
      {
        h_.destroy();
      }
    }

    AsyncTask(const AsyncTask&) = delete;            ///< non-copyable
    AsyncTask& operator=(const AsyncTask&) = delete; ///< non-copyable

    /// \brief Move-construct.
    AsyncTask(AsyncTask&& other) noexcept : h_{other.h_} { other.h_ = nullptr; }

    /// \brief Move-assign.
    AsyncTask& operator=(AsyncTask&& other) noexcept
    {
      if (this != &other)
      {
        if (h_)
        {
          h_.destroy();
        }
        h_ = other.h_;
        other.h_ = nullptr;
      }
      return *this;
    }
};

} // namespace uni20::async
