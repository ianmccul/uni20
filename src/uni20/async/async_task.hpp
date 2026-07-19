/**
 * \file async_task.hpp
 * \brief Promise-neutral coroutine task ownership and the public AsyncTask type.
 */

#pragma once

#include "task_registry.hpp"

#include <atomic>
#include <concepts>
#include <coroutine>
#include <exception>
#include <optional>
#include <string>
#include <uni20/common/trace.hpp>
#include <utility>

namespace uni20::async
{

class IScheduler;
class TaskPromiseBase;
class AsyncTaskPromise;
class CudaTaskPromise;
class TaskFactory;
class AsyncTask;
class CudaTask;

/// \brief Promise concept accepted by Uni20 task machinery.
template <typename T>
concept TaskPromise = std::derived_from<T, TaskPromiseBase>;

/// \brief Non-owning erased identity for one Uni20 coroutine frame.
/// \details The coroutine and promise pointers are always constructed together
///          from the original typed coroutine handle. A TaskHandle does not add
///          or release an ownership claim and never destroys a frame
///          automatically.
class TaskHandle {
  public:
    constexpr TaskHandle() noexcept = default;

    template <TaskPromise Promise> [[nodiscard]] static TaskHandle from(std::coroutine_handle<Promise> handle) noexcept;

    [[nodiscard]] explicit constexpr operator bool() const noexcept { return static_cast<bool>(coroutine_); }

    [[nodiscard]] constexpr std::coroutine_handle<> coroutine() const noexcept { return coroutine_; }

    [[nodiscard]] TaskPromiseBase& promise() const noexcept;

    [[nodiscard]] bool done() const noexcept { return !coroutine_ || coroutine_.done(); }

    friend constexpr bool operator==(TaskHandle lhs, TaskHandle rhs) noexcept
    {
      return lhs.coroutine_ == rhs.coroutine_;
    }

  private:
    TaskHandle(std::coroutine_handle<> coroutine, TaskPromiseBase* promise) noexcept
        : coroutine_(coroutine), promise_(promise)
    {
      DEBUG_CHECK_EQUAL(static_cast<bool>(coroutine_), promise_ != nullptr);
    }

    std::coroutine_handle<> coroutine_{};
    TaskPromiseBase* promise_ = nullptr;

    friend class TaskPromiseBase;
};

/// \brief Move-only promise-neutral ownership claim for a Uni20 coroutine.
/// \details BasicTask is the internal task currency used by awaiters,
///          continuations, and schedulers after typed initial admission.
class BasicTask {
  public:
    /// \brief Construct an empty task.
    BasicTask() noexcept = default;

    BasicTask(BasicTask const&) = delete;
    BasicTask& operator=(BasicTask const&) = delete;

    /// \brief Move-construct one ownership claim.
    BasicTask(BasicTask&& other) noexcept
        : handle_(std::exchange(other.handle_, {})), await_exception_(std::exchange(other.await_exception_, {}))
    {}

    /// \brief Move-assign one ownership claim.
    BasicTask& operator=(BasicTask&& other) noexcept;

    /// \brief Release the ownership claim and destroy the frame when appropriate.
    ~BasicTask() noexcept;

    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(handle_); }

    /// \brief Return the erased task identity.
    [[nodiscard]] TaskHandle handle() const noexcept { return handle_; }

    /// \brief Return the generic coroutine handle for diagnostics.
    [[nodiscard]] std::coroutine_handle<> coroutine_handle() const noexcept { return handle_.coroutine(); }

    /// \brief Assign an optional debug label.
    BasicTask& debug_name(std::string const& label)
    {
      TaskRegistry::name_task(this->coroutine_handle(), label);
      return *this;
    }

    /// \brief Resume the coroutine and transfer this ownership claim.
    void resume();

    /// \brief Intentionally leak the coroutine during emergency unwinding.
    void abandon_leak();

    /// \brief Release this ownership claim to a scheduler activation.
    [[nodiscard]] TaskHandle release_handle();

    /// \brief Request cancellation before the next resume.
    void set_cancel_on_resume() noexcept;

    /// \brief Store an exception for delivery on the next resume.
    void exception_on_resume(std::exception_ptr error) noexcept;

    /// \brief Destroy an exclusively owned coroutine.
    void destroy() noexcept;

    /// \brief Install the scheduler used for later resumption.
    [[nodiscard]] bool set_scheduler(IScheduler* scheduler);

    [[nodiscard]] std::optional<int> preferred_numa_node() const noexcept;
    void set_preferred_numa_node(std::optional<int> node) noexcept;

    /// \brief Reschedule a task when it becomes the sole ownership claimant.
    static void reschedule(BasicTask task);

    /// \brief Retain a task only when it is the sole ownership claimant.
    [[nodiscard]] static BasicTask make_sole_owner(BasicTask&& task);

    [[nodiscard]] bool await_ready() const noexcept { return !handle_ || handle_.done(); }

    /// \brief Await a nested task from any Uni20 task promise type.
    template <TaskPromise ParentPromise>
    std::coroutine_handle<> await_suspend(std::coroutine_handle<ParentPromise> parent);

    void await_resume();

  protected:
    struct construction_key
    {};

    explicit BasicTask(TaskHandle handle, construction_key) noexcept : handle_(handle) {}

  private:
    [[nodiscard]] bool can_destroy_coroutine(TaskHandle handle) const noexcept;
    [[nodiscard]] TaskHandle release_ownership();
    void release() noexcept;
    void destroy_owned_coroutine() noexcept;

    TaskHandle handle_{};
    std::exception_ptr await_exception_{};

    friend class TaskPromiseBase;
    friend class TaskFactory;
    friend struct AsyncTaskTestAccess;
};

/// \brief Canonical host-oriented coroutine task type.
class AsyncTask final : public BasicTask {
  public:
    using base_type = BasicTask;
    using promise_type = AsyncTaskPromise;

    AsyncTask() = default;
    AsyncTask(AsyncTask const&) = delete;
    AsyncTask& operator=(AsyncTask const&) = delete;
    AsyncTask(AsyncTask&&) noexcept = default;
    AsyncTask& operator=(AsyncTask&&) noexcept = default;

    AsyncTask& debug_name(std::string const& label)
    {
      this->BasicTask::debug_name(label);
      return *this;
    }

  private:
    explicit AsyncTask(TaskHandle handle) noexcept : BasicTask(handle, construction_key{}) {}

    friend class AsyncTaskPromise;
};

} // namespace uni20::async

#include "async_task_promise.hpp"
