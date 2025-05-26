/// \file async_task_promise.hpp
/// \brief Defines AsyncTask::promise_type, the fire-and-forget coroutine handle.
/// \ingroup async_core

#pragma once

#include "scheduler.hpp"
#include <atomic>
#include <coroutine>
#include <exception>

namespace uni20::async
{

class AsyncTaskFactory;

/// \brief Concept for the valid return types of await_suspend.
/// \details
/// An awaiter may return void, or it can return an AsyncTask, which means that
/// executation of the current coroutine should be transferred to the new task,
/// resuming the coroutine only after the new task is complete.
/// If an AsyncTask is returned, it must have exclusive ownership, otherwise the
/// task cannot be scheduled
template <typename Ret>
concept AwaitSuspendResult = std::same_as<Ret, void> || std::same_as<Ret, AsyncTask>;

/// \brief Concept for awaitables that accept ownership via an AsyncTask.
///
/// This concept is satisfied if the awaitable provides:
/// - `await_suspend(AsyncTask)`
/// - The return type of `await_suspend` must be `void` or `bool`
///
/// \note This concept disallows await_suspend() from returning a coroutine_handle,
///       to ensure that ownership and resumption are managed solely by the scheduler.
template <typename T>
concept AsyncTaskAwaitable = requires(T a, AsyncTask t) {
                               {
                                 a.await_suspend(std::move(t))
                                 } -> AwaitSuspendResult;
                             };

/// \brief Concept for awaitables that support shared ownership via AsyncTaskFactory.
///
/// This concept is satisfied if:
/// - The awaitable provides a `num_awaiters()` method returning an integer count
/// - It provides `await_suspend(AsyncTaskFactory)`
/// - The return type of `await_suspend` must be `void` or `bool`
///
/// \note This is used by composite awaiters like `all(...)` that must split
///       ownership across multiple sub-awaitables.
template <typename T>
concept AsyncTaskFactoryAwaitable = requires(T a, AsyncTaskFactory t) {
                                      {
                                        a.await_suspend(std::move(t))
                                        } -> AwaitSuspendResult;
                                      {
                                        a.num_awaiters()
                                        } -> std::convertible_to<int>;
                                    };

/// \brief Forwarding awaiter that takes ownership of a std::coroutine_handle and forwards to an awaiter as an AsyncTask
template <AsyncTaskAwaitable A> struct AsyncTaskAwaiter;

/// \brief Forwarding awaiter that takes shared ownership of a std::coroutine_handle and forwards to an awaiter as an
/// AsyncTaskFactory
template <AsyncTaskFactoryAwaitable A> struct AsyncTaskFactoryAwaiter;

/// \brief Promise type for AsyncTask.
/// \ingroup async_core
struct AsyncTask::promise_type
{
    /// \brief Scheduler to notify when the coroutine is ready to resume.
    IScheduler* sched_ = nullptr;

    /// \brief Number of active awaiters (owners) of this coroutine.
    ///        This is exactly equal to the number of AsyncTask instances that refer to this coroutine.
    /// \note When the count reaches zero, the coroutine is considered unowned.
    ///       Ownership must be transferred explicitly using take_ownership().
    std::atomic<int> awaiter_count = 0;

    std::coroutine_handle<> continuation_ = nullptr;

    /// \brief Decrease the number of active awaiters by one.
    /// \return true if this was the last awaiter and the coroutine is now unowned.
    bool release_awaiter() { return awaiter_count.fetch_sub(1, std::memory_order_acq_rel) == 1; }

    /// \brief Decrease the number of active awaiters by a specified count.
    /// \param count The number of awaiters to release.
    /// \return true if the count reached zero exactly as a result of this call.
    bool release_awaiter(int count) { return awaiter_count.fetch_sub(count, std::memory_order_acq_rel) == count; }

    /// \brief Increase the number of active awaiters by one.
    /// \return The value of the counter prior to the increment.
    int add_awaiter() { return awaiter_count.fetch_add(1, std::memory_order_relaxed); }

    /// \brief Increase the number of active awaiters by a specified count.
    /// \param count The number of awaiters to add.
    /// \return The value of the counter prior to the increment.
    int add_awaiter(int count) { return awaiter_count.fetch_add(count, std::memory_order_relaxed); }

    /// \brief Transform the awaiter to provide transfer of ownership of the AsyncTask
    template <AsyncTaskAwaitable A> auto await_transform(A& a);
    template <AsyncTaskAwaitable A> auto await_transform(A&& a);

    /// \brief Transform the awaiter to provide transfer of shared ownership of the AsyncTaskFactory
    template <AsyncTaskFactoryAwaitable A> auto await_transform(A& a);
    template <AsyncTaskFactoryAwaitable A> auto await_transform(A&& a);

    // Pass-through for AsyncTask itself
    AsyncTask& await_transform(AsyncTask& t) noexcept { return t; }
    AsyncTask&& await_transform(AsyncTask&& t) noexcept { return std::move(t); }

    template <typename T> auto await_transform(T&&)
    {
      static_assert(
          sizeof(T) == 0,
          "co_await expression does not match any known AsyncTaskAwaitable or AsyncTaskFactoryAwaitable type.");
    }

    /// \brief Acquire exclusive ownership of the coroutine.
    ///       This increments the awaiter count and asserts that the coroutine was previously unowned.
    /// \pre The coroutine must be unowned (awaiter_count == 0).
    /// \return A newly constructed AsyncTask that takes ownership of the coroutine.
    AsyncTask take_ownership() noexcept
    {
      int prior_count = this->add_awaiter();
      DEBUG_CHECK_EQUAL(prior_count, 0, "expected handle to be previously unowned!");
      return AsyncTask{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    /// \brief Acquire shared ownership of the coroutine for use with multi-await constructs.
    /// \param count The number of distinct AsyncTask instances to be created.
    /// \pre The coroutine must be unowned (awaiter_count == 0).
    /// \return A factory that will dispense up to \p count owning AsyncTask handles.
    AsyncTaskFactory take_shared_ownership(int count);

    /// \brief Release ownership of the coroutine and return it, if it was exclusively owned
    /// \return the coroutine handle, if we had exclusive ownership; otherwise returns null
    /// \post the task has been released and *this is a null AsyncTask
    std::coroutine_handle<promise_type> release_ownership()
    {
      return this->release_awaiter() ? std::coroutine_handle<promise_type>::from_promise(*this) : nullptr;
    }

    /// \brief Default-construct the promise.
    constexpr promise_type() noexcept = default;

    /// \brief Constructs the coroutine's return object for the caller.
    ///
    /// \note This is invoked exactly once, before initial_suspend().
    /// \return An `AsyncTask` owning the coroutine handle corresponding to this promise,
    ///         transferring lifetime responsibility to the caller.
    AsyncTask get_return_object() noexcept
    {
      this->add_awaiter();
      return AsyncTask{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    /// \brief Suspend immediately on coroutine entry.
    constexpr std::suspend_always initial_suspend() noexcept { return {}; }

    /// \brief Suspend at ther end of the coroutine. This makes no practical difference in our case,
    ///        either way the sceduler that called .resume() would see a done() coroutine and the only
    ///        valid operation is to call .destroy()

    auto final_suspend() noexcept
    {
      struct FinalAwaiter
      {
          constexpr bool await_ready() noexcept { return false; }

          std::coroutine_handle<> await_suspend(std::coroutine_handle<AsyncTask::promise_type> h) noexcept
          {
            auto continuation = h.promise().continuation_;
            TRACE("Final suspend of coroutine", h, continuation);
            h.destroy();
            TRACE("Destroy is done");
            return continuation ? continuation : std::noop_coroutine();
          }

          void await_resume() noexcept {}
      };
      return FinalAwaiter{};
    }

    /// \brief Called when the coroutine returns normally.
    constexpr void return_void() noexcept {}

    /// \brief Called on unhandled exception escaping the coroutine.
    [[noreturn]] void unhandled_exception() { std::terminate(); }

    // /// \brief Called when the task is ready to run again.
    // /// \param h The coroutine handle to schedule.
    // void notify_ready(std::coroutine_handle<promise_type> h) { sched_->schedule(h); }
};

/// \brief Factory for producing multiple AsyncTask instances that share ownership of the same coroutine.
///
/// This is used when multiple awaiters (e.g., in an all(A, B, C) construct) need to take independent ownership
/// of the same coroutine. The factory pre-allocates all references atomically and ensures that they are
/// handed out exactly once.
///
/// \note This class must only be created while the coroutine is unowned. Attempting to add awaiters after
///       the coroutine is active leads to race conditions and is undefined behavior.
/// \ingroup async_core
class AsyncTaskFactory {
  public:
    /// \brief Dispense the next AsyncTask from the pool of shared ownership handles.
    /// \pre `count_ > 0` — there must be remaining tasks to dispense.
    /// \post One fewer task will be available from this factory.
    /// \return An AsyncTask that shares ownership of the coroutine.
    AsyncTask take_next()
    {
      DEBUG_PRECONDITION(count_ > 0);
      --count_;
      return AsyncTask(handle_);
    }

    AsyncTaskFactory(AsyncTaskFactory&& other) noexcept
        : handle_(std::exchange(other.handle_, {})), count_(std::exchange(other.count_, 0))
    {}

    // Move assignment
    AsyncTaskFactory& operator=(AsyncTaskFactory&& other) noexcept
    {
      if (this != &other)
      {
        handle_ = std::exchange(other.handle_, {});
        count_ = std::exchange(other.count_, 0);
      }
      return *this;
    }

    AsyncTaskFactory(AsyncTaskFactory const&) = delete;
    AsyncTaskFactory& operator=(AsyncTaskFactory const&) = delete;

    /// \brief Destructor returns any unused ownership claims and may destroy the coroutine.
    ///
    /// If any AsyncTask instances were not handed out via `take_next()`, they are released here.
    /// If the coroutine is unowned after release, it will be destroyed.
    ~AsyncTaskFactory() noexcept
    {
      // return the outstanding references. If this results in a zero reference count, then destroy the handle.
      // it is possible that the handle has already been destroyed, but that could only happen if we gave out
      // all of the references (and they were since destructed), which would require count_ == 0.
      DEBUG_TRACE(this, handle_, count_);
      if (count_ > 0 && handle_.promise().release_awaiter(count_))
      {
        handle_.destroy();
      }
    }

  private:
    friend class AsyncTask::promise_type;

    using HandleType = std::coroutine_handle<AsyncTask::promise_type>;

    /// \brief Construct a factory with N shared references to the coroutine.
    /// \pre The coroutine must be unowned (`awaiter_count == 0`) at the time of construction.
    /// \param h The coroutine handle.
    /// \param count The number of AsyncTask instances to dispense.
    AsyncTaskFactory(HandleType h, int count) : handle_(h), count_(count)
    {
      int prior_count = handle_.promise().add_awaiter(count);
      DEBUG_CHECK_EQUAL(prior_count, 0, "expected handle to be previously unowned!");
      // if we requested zero references, then we can destroy the handle immediately
      if (count_ == 0) handle_.destroy();
    }

    HandleType handle_;
    size_t count_;
};

template <AsyncTaskAwaitable A> struct AsyncTaskAwaiter
{
    A awaitable;

    using await_return_type = decltype(awaitable.await_suspend(std::declval<AsyncTask>()));

    bool await_ready() { return awaitable.await_ready(); }

    auto await_suspend(std::coroutine_handle<AsyncTask::promise_type> h)
    {
      if constexpr (std::is_void_v<await_return_type>)
      {
        return awaitable.await_suspend(h.promise().take_ownership());
      }
      else
      {
        auto t = awaitable.await_suspend(h.promise().take_ownership());
        if (!t.h_) return std::noop_coroutine();
        auto h_new = t.h_.promise().release_ownership();
        CHECK(h_new, "coroutine handle was not exclusively owned!");
        h_new.promise().continuation_ = h;
        return h_new;
      }
    }

    decltype(auto) await_resume() { return awaitable.await_resume(); }
};

template <AsyncTaskFactoryAwaitable A> struct AsyncTaskFactoryAwaiter
{
    A awaitable;

    bool await_ready() { return awaitable.await_ready(); }

    auto await_suspend(std::coroutine_handle<AsyncTask::promise_type> h)
    {
      return awaitable.await_suspend(h.promise().take_shared_ownership(awaitable.num_awaiters()));
    }

    decltype(auto) await_resume() { return awaitable.await_resume(); }
};

template <AsyncTaskAwaitable A> inline auto AsyncTask::promise_type::await_transform(A& a)
{
  return AsyncTaskAwaiter<A&>(a);
}

template <AsyncTaskAwaitable A> inline auto AsyncTask::promise_type::await_transform(A&& a)
{
  return AsyncTaskAwaiter<A&>(a);
}

template <AsyncTaskFactoryAwaitable A> inline auto AsyncTask::promise_type::await_transform(A& a)
{
  return AsyncTaskFactoryAwaiter<A&>(a);
}

template <AsyncTaskFactoryAwaitable A> inline auto AsyncTask::promise_type::await_transform(A&& a)
{
  return AsyncTaskFactoryAwaiter<A>(a);
}

inline void AsyncTask::reschedule(AsyncTask task)
{
  TRACE("rescheduling AsyncTask", &task);
  task = AsyncTask::make_sole_owner(std::move(task));
  if (task)
  {
    DEBUG_CHECK(task.h_.promise().sched_, "unexpected: task scheduler is not set!");
    TRACE("rescheduling AsyncTask, submitting to queue", &task, task.h_);
    auto sched = task.h_.promise().sched_;
    sched->reschedule(std::move(task));
  }
}

inline AsyncTask AsyncTask::make_sole_owner(AsyncTask&& task)
{
  auto& p = task.h_.promise();
  if (p.release_awaiter() == 1)
  {
    // We were the last - reacquire ownership explicitly
    p.add_awaiter();
  }
  else
  {
    // Not the last owner, release ownership
    task.h_ = nullptr;
  }
  return std::move(task);
}

inline void AsyncTask::resume()
{
  CHECK(h_);
  TRACE("Resuming AsyncTask", h_);
  if (!h_.promise().release_awaiter()) PANIC("Attempt to resume() a non-exclusive AsyncTask");

  auto handle = h_;
  h_ = nullptr; // Always drop ownership, we are now effectively in a 'moved from' state
  // we need to do this *before* calling h_.resume(), because it is possible that this AsyncTask
  // is on that coroutine frame, and it might get destroyed.

  handle.resume();
  TRACE("returned from coroutine::resume");
}

inline AsyncTask& AsyncTask::operator=(AsyncTask&& other) noexcept
{
  trace::TracingBaseClass<AsyncTask>::operator=(std::move(other));
  TRACE("AsyncTask move assignment", this, h_, &other, other.h_);
  if (this != &other)
  {
    if (h_ && h_.promise().release_awaiter()) h_.destroy();
    h_ = other.h_;
    other.h_ = nullptr;
  }
  return *this;
}

inline AsyncTask::~AsyncTask()
{
  TRACE("Destroying AsyncTask", this, h_);
  if (h_ && h_.promise().release_awaiter())
  {
    TRACE("AsyncTask destructor is destroying the coroutine!", this, h_);
    h_.destroy(); // We are the last owner, safe to destroy
  }
}

inline bool AsyncTask::set_scheduler(IScheduler* sched)
{
  if (h_)
  {
    h_.promise().sched_ = sched;
    return true;
  }
  return false;
}

inline AsyncTaskFactory AsyncTask::promise_type::take_shared_ownership(int count)
{
  return AsyncTaskFactory(std::coroutine_handle<promise_type>::from_promise(*this), count);
}

inline std::coroutine_handle<AsyncTask::promise_type>
AsyncTask::await_suspend(std::coroutine_handle<AsyncTask::promise_type> Outer)
{
  DEBUG_CHECK(h_);
  DEBUG_CHECK(!h_.promise().continuation_);
  h_.promise().continuation_ = Outer;
  h_.promise().sched_ = Outer.promise().sched_;
  auto h_transfer = h_.promise().release_ownership();
  h_ = nullptr; // finish transferring ownership
  CHECK(h_transfer, "error: co_await on an AsyncTask that has shared ownership");
  return h_transfer;
}

inline void AsyncTask::await_resume() const noexcept {}

} // namespace uni20::async
