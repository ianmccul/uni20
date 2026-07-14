/// \file async_errors.hpp
/// \brief Exception hierarchy for async/dataflow operations.

#pragma once

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace uni20::async
{

/// \brief Base class for all async subsystem exceptions.
class async_error : public std::runtime_error {
  public:
    explicit async_error(std::string message) : std::runtime_error(std::move(message)) {}
};

/// \brief Base class for invalid async state transitions or accesses.
class async_state_error : public async_error {
  public:
    explicit async_state_error(std::string message) : async_error(std::move(message)) {}
};

/// \brief Thrown when Async storage metadata is missing unexpectedly.
class async_storage_missing : public async_state_error {
  public:
    async_storage_missing() : async_state_error("Async storage is missing") {}
};

/// \brief Thrown when an Async value is accessed before initialization.
class async_value_uninitialized : public async_state_error {
  public:
    async_value_uninitialized() : async_state_error("Async value requires initialization before access") {}
};

/// \brief Raised when a blocking async wait observes no scheduler-visible progress before its watchdog expires.
class async_wait_timeout : public async_error {
  public:
    /// \brief Construct a no-progress timeout diagnostic.
    /// \param timeout Idle interval that expired.
    /// \param scheduler_paused Whether the scheduler was paused at the timeout.
    async_wait_timeout(std::chrono::milliseconds timeout, bool scheduler_paused)
        : async_error("Async wait made no scheduler-visible progress for " + std::to_string(timeout.count()) +
                      " ms (scheduler " + (scheduler_paused ? "paused" : "running") + ")"),
          timeout_(timeout), scheduler_paused_(scheduler_paused)
    {}

    /// \brief Return the expired idle interval.
    /// \return Configured watchdog timeout.
    [[nodiscard]] std::chrono::milliseconds timeout() const noexcept { return timeout_; }

    /// \brief Report whether the scheduler was paused when the timeout was diagnosed.
    /// \return `true` when the scheduler was paused.
    [[nodiscard]] bool scheduler_paused() const noexcept { return scheduler_paused_; }

  private:
    std::chrono::milliseconds timeout_;
    bool scheduler_paused_;
};

/// \brief Base class for cancellation-related async exceptions.
class async_cancellation : public async_error {
  public:
    explicit async_cancellation(std::string message) : async_error(std::move(message)) {}
};

/// \brief Raised when an AsyncTask is cancelled before completion.
class task_cancelled : public async_cancellation {
  public:
    task_cancelled() : async_cancellation("AsyncTask was cancelled") {}
};

/// \brief Base class for buffer-related failures.
class buffer_error : public async_state_error {
  public:
    explicit buffer_error(std::string message) : async_state_error(std::move(message)) {}
};

/// \brief Raised when a buffer read path is cancelled intentionally.
class buffer_cancelled : public async_cancellation {
  public:
    buffer_cancelled() : async_cancellation("ReadBuffer was cancelled: no value written") {}
};

/// \brief Base class for access to a buffer without a constructed value.
class buffer_uninitialized : public buffer_error {
  public:
    explicit buffer_uninitialized(std::string message) : buffer_error(std::move(message)) {}
};

/// \brief Raised when a reader attempts to access an uninitialized buffer.
class buffer_read_uninitialized : public buffer_uninitialized {
  public:
    buffer_read_uninitialized() : buffer_uninitialized("Attempt to read from a buffer that has not been initialized") {}
};

/// \brief Raised when a writer requests a mutable reference before construction.
class buffer_write_uninitialized : public buffer_uninitialized {
  public:
    buffer_write_uninitialized()
        : buffer_uninitialized("Attempt to write via mutable reference to an uninitialized buffer; use emplace()")
    {}
};

} // namespace uni20::async
