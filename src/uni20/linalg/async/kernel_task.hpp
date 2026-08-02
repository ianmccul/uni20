#pragma once

/**
 * \file kernel_task.hpp
 * \ingroup linalg
 * \brief Deferred task result for an optional coroutine kernel implementation.
 */

#include <uni20/async/async_task.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/kernel_attempt.hpp>

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Runtime backend attempt optionally carrying a deferred kernel task.
/// \details A decline carries no task and must satisfy the ordinary
///          clean-decline contract, including its allowance for provisional
///          preparation of operation-declared replaceable outputs. Success
///          without a task represents an operation that is already complete,
///          such as an empty output. Success with a task commits the backend
///          once that task is awaited; later failures are terminal exceptions
///          rather than backend declines.
template <class Task>
  requires std::derived_from<Task, async::BasicTask>
class KernelTaskAttempt {
  public:
    using task_type = Task;

    /// \brief Construct a completed attempt without deferred work.
    explicit KernelTaskAttempt(KernelAttempt attempt) noexcept : attempt_(attempt) {}

    /// \brief Construct a successful attempt with deferred work.
    explicit KernelTaskAttempt(Task task) noexcept : task_(std::move(task)) {}

    KernelTaskAttempt(KernelTaskAttempt const&) = delete;
    KernelTaskAttempt& operator=(KernelTaskAttempt const&) = delete;
    KernelTaskAttempt(KernelTaskAttempt&&) noexcept = default;
    KernelTaskAttempt& operator=(KernelTaskAttempt&&) noexcept = default;

    /// \brief Return the runtime acceptance result.
    [[nodiscard]] KernelAttempt attempt() const noexcept { return attempt_; }

    /// \brief Return whether successful execution still requires awaiting a task.
    [[nodiscard]] bool has_task() const noexcept { return task_.has_value(); }

    /// \brief Transfer the deferred task from a successful attempt.
    [[nodiscard]] Task take_task() &&
    {
      CHECK(kernel_attempt_succeeded(attempt_));
      CHECK(task_.has_value());
      return std::move(*task_);
    }

  private:
    KernelAttempt attempt_ = KernelAttempt::success;
    std::optional<Task> task_{};
};

namespace detail
{
template <class T> struct IsKernelTaskAttempt : std::false_type
{};

template <class Task> struct IsKernelTaskAttempt<KernelTaskAttempt<Task>> : std::true_type
{};

template <class T> inline constexpr bool is_kernel_task_attempt = IsKernelTaskAttempt<std::remove_cvref_t<T>>::value;
} // namespace detail

} // namespace uni20::linalg
