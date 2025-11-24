/// \file async_task.hpp
/// \brief Defines BasicAsyncTask, the fire-and-forget coroutine handle.
/// \ingroup async_core

#pragma once

#include "common/trace.hpp"
#include <atomic>
#include <coroutine>
#include <exception>
#include <optional>

namespace uni20::async
{

class IScheduler;

class BasicAsyncTaskPromise;

#if 0 // an experiment
class AsyncAwaiter {
  public:
    virtual void set_cancel() = 0;
    virtual void set_exception(std::exception_ptr e) = 0;
};
#endif

class BasicAsyncTaskPromise;

// We require the BasicAsyncTask promise type to inherit from BasicAsyncTaskPromise,
// so that it is layout-compatible for type-safe upcasts
template <typename T>
concept IsAsyncTaskPromise = std::derived_from<T, BasicAsyncTaskPromise>;

/// \brief A fire-and-forget coroutine handle.
/// \ingroup async_core
template <IsAsyncTaskPromise Promise> class BasicAsyncTask { //}: public AsyncAwaiter {
  public:
    using promise_type = Promise;
    using handle_type = std::coroutine_handle<promise_type>;

    /// \brief Release the coroutine if this object owns the final reference.
    /// \ingroup async_core
    ~BasicAsyncTask() noexcept;

    /// \brief Drop the reference to the coroutine handle, destroying it if we are the least reference
    /// \ingroup internal
  private:
    [[nodiscard]] bool can_destroy_coroutine(handle_type h) const noexcept;

    void release() noexcept;
    void destroy_owned_coroutine() noexcept;

  public:
    // bool done() const noexcept { return !h_ || h_.done(); }

    /// \brief Check if this task refers to a coroutine.
    /// \return true if the handle is non-null.
    explicit operator bool() const noexcept { return static_cast<bool>(h_); }

    /// \brief Resume the coroutine.
    /// \pre We are the sole owner of the coroutine.
    /// \note This releases ownership to the scheduler.
    /// \ingroup async_core
    void resume();

    /// \brief Transfer ownership of the coroutine and get the handle.
    /// \pre We are the sole owner of the coroutine.
    /// \note The returned handle may be null if the coroutine has been cancelled.
    /// \return Coroutine handle previously managed by this task.
    /// \ingroup async_core
    handle_type release_handle();

    /// \brief Indicate that the coroutine should be cancelled upon resume.
    /// \ingroup async_core
    void cancel_on_resume() noexcept;

    /// \brief Indicate that the coroutine should pass an exception upon resume.
    /// \param e Exception pointer to deliver when resumed.
    /// \ingroup async_core
    void exception_on_resume(std::exception_ptr e) noexcept;

    /// \brief Destroy the coroutine.
    /// \pre We are the sole owner of the coroutine.
    /// \note This releases ownership.
    /// \ingroup async_core
    void destroy() noexcept;

    /// \brief Attempt to set the coroutine scheduler to use when rescheduling the task.
    /// \return true if the handle is non-null (in which case the scheduler has been set).
    /// \ingroup async_core
    bool set_scheduler(IScheduler* sched);

    /// \brief Retrieve the preferred NUMA node associated with the coroutine, if any.
    /// \return Optional containing the preferred NUMA node.
    /// \ingroup async_core
    std::optional<int> preferred_numa_node() const noexcept;

    /// \brief Update the preferred NUMA node associated with the coroutine.
    /// \param node Preferred node index, or empty to clear the preference.
    /// \ingroup async_core
    void set_preferred_numa_node(std::optional<int> node) noexcept;

    /// \brief Resubmit a suspended BasicAsyncTask to its scheduler, if this is the sole remaining owner.
    ///
    /// This transfers ownership to the scheduler only if the task has exclusive ownership of the coroutine.
    /// If other awaiters remain, the task is discarded and not rescheduled.
    ///
    /// \pre The scheduler in the promise must have been set.
    /// \param task The task to be conditionally rescheduled.
    /// \ingroup async_core
    static void reschedule(BasicAsyncTask task);

    /// \brief Retain the task only if it is the sole remaining owner.
    ///
    /// This operation checks the awaiter count. If this is the last owner,
    /// then return the task unchanged. Otherwise, releases the task and leaves
    /// it requivalent to a moved-from state.
    ///
    /// \return The original task if it had exclusive ownership; otherwise a null task.
    /// \ingroup async_core
    static BasicAsyncTask make_sole_owner(BasicAsyncTask&& task);

    /// \brief Construct an empty task handle.
    /// \ingroup async_core
    BasicAsyncTask() = default;

    BasicAsyncTask(const BasicAsyncTask&) = delete;            ///< non-copyable
    BasicAsyncTask& operator=(const BasicAsyncTask&) = delete; ///< non-copyable

    /// \brief Move-construct.
    /// \param other Task being moved from.
    /// \ingroup async_core
    BasicAsyncTask(BasicAsyncTask&& other) noexcept
        : h_{other.h_}, cancel_{other.cancel_.load(std::memory_order_acquire)}
    {
      // DEBUG_TRACE("BasicAsyncTask move", this, &other, h_);
      other.h_ = nullptr;
    }

    /// \brief Move-assign.
    /// \param other Task being moved from.
    /// \return Reference to *this after transfer.
    /// \ingroup async_core
    BasicAsyncTask& operator=(BasicAsyncTask&& other) noexcept;

    //
    // Awaiter interface
    // If we co_await on an BasicAsyncTask, then it has the semantics of transferring execution
    // to that task, and then resuming the current coroutine afterwards.
    // if coroutine Outer contains co_await Inner, then we end up with
    // Inner.await_suspend(Outer)
    //

    /// \brief Check whether the coroutine has already completed.
    /// \return true if no suspension is required.
    /// \ingroup async_core
    bool await_ready() const noexcept { return !h_ || h_.done(); }

    /// \brief Transfer execution to the managed coroutine.
    /// \param Outer Handle to the awaiting coroutine.
    /// \return Handle to resume after the awaited coroutine completes.
    /// \ingroup async_core
    handle_type await_suspend(handle_type Outer);

    /// \brief Resume the awaiting coroutine after completion.
    /// \ingroup async_core
    void await_resume() const noexcept;

    // void set_cancel() override final { cancel_.store(true, std::memory_order_release); }
    //
    // void set_exception(std::exception_ptr e) override final
    // {
    //   exception_ = e;
    //   cancel_.store(true, std::memory_order_release);
    // }

    /// \brief Flag the coroutine so that the next resume will deliver a cancellation.
    /// \ingroup async_core
    void cancel_if_unwritten() { cancel_.store(true, std::memory_order_release); }

    /// \brief Indicate that the coroutine produced a value successfully.
    /// \ingroup async_core
    void written() { cancel_.store(false, std::memory_order_release); }

    // void set_current_awaiter(AsyncAwaiter* awaiter)
    // {
    //   DEBUG_CHECK(!current_awaiter_);
    //   current_awaiter_ = awaiter;
    // }
    // void clear_current_awaiter() noexcept
    // {
    //   DEBUG_CHECK(current_awaiter_);
    //   current_awaiter_ = nullptr;
    // }

    //  private:
    handle_type h_; ///< Underlying coroutine handle.

    // When a buffer detects an error condition, we need to know which awaiter to send it to
    //  AsyncAwaiter* current_awaiter_ = nullptr;

    /// \brief indicates that the coroutine has an error condition waiting on resume.
    /// If eptr_ is non-null, then an exception has been thrown; otherwise the buffer has been cancelled
    /// \note this is only used by the AsyncTask awaiter interface
    //    std::exception_ptr exception_ = nullptr;
    std::atomic<bool> cancel_ = false;

    // inline static IScheduler* sched_ = nullptr;

    /// \brief Construct from a coroutine handle.
    /// \param h The coroutine handle.
    /// \ingroup async_core
    explicit BasicAsyncTask(std::coroutine_handle<promise_type> h) noexcept : h_(h) {}

    // explicit BasicAsyncTask(std::coroutine_handle<promise_type> h, AsyncAwaiter* awaiter) noexcept
    //     : h_(h) //, current_awaiter_(awaiter)
    // {}

    friend promise_type;
    friend class BasicAsyncTaskFactory;
    friend struct AsyncTaskTestAccess;
};

using AsyncTask = BasicAsyncTask<BasicAsyncTaskPromise>;

} // namespace uni20::async

#include "async_task_impl.hpp"
