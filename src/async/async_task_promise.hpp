/// \file async_task_promise.hpp
/// \brief Defines AsyncTask::promise_type, the fire-and-forget coroutine handle.
/// \ingroup async_core

#pragma once

#include "async_node.hpp"
#include "async_errors.hpp"
#include "async_task.hpp"
#include "scheduler.hpp"
#include <atomic>
#include <coroutine>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace uni20::async
{

class AsyncTaskFactory;
class EpochContext;

/// \brief Inject an unhandled writer-side coroutine exception into an epoch.
/// \param epoch Target epoch context associated with a writer buffer.
/// \param eptr Exception captured by the coroutine promise.
void propagate_unhandled_writer_exception(EpochContext* epoch, std::exception_ptr eptr) noexcept;

// class AsyncTask;
//

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
struct BasicAsyncTaskPromise
{
    using promise_type = BasicAsyncTaskPromise;

    /// \brief Intrusive node describing one exception propagation sink.
    struct ExceptionSinkNode
    {
        BasicAsyncTaskPromise* owner{nullptr};
        ExceptionSinkNode* prev{nullptr};
        ExceptionSinkNode* next{nullptr};
        std::shared_ptr<EpochContext> epoch{};
        bool explicit_sink{false};
    };

    /// \brief Scheduler to notify when the coroutine is ready to resume.
    IScheduler* sched_ = nullptr;

    /// \brief Tracks whether the coroutine has been scheduled or otherwise started.
    std::atomic<bool> started_{false};

    /// \brief AsyncTask coroutines are nestable -- we can co_await one AsyncTask inside another.
    /// continuation_ tracks the 'parent' coroutine in this case, so that when we finish execution we return to
    /// executing the parent coroutine.
    std::coroutine_handle<promise_type> continuation_ = nullptr;

    /// \brief Number of active awaiters (owners) of this coroutine.
    ///        This is exactly equal to the number of AsyncTask instances that refer to this coroutine.
    /// \note When the count reaches zero, the coroutine is considered unowned.
    ///       Ownership must be transferred explicitly using take_ownership().
    std::atomic<int> awaiter_count_ = 0;

    /// \brief EpochContext can inject an exception into a coroutine, which will be thrown when resuming
    /// from an awaiter.  This could, in principle, happen from multiple threads if we are awaiting multiple
    /// buffers so we protect access by the exception_ flag.  Only the first attempt to set exception_ to true
    /// is permitted to set eptr_; subsequent exceptions are simply dropped.
    std::atomic<bool> exception_{false};
    std::exception_ptr eptr_ = nullptr;

    /// \brief If true, then when the coroutine is next resumed (i.e. transferred to the scheduler)
    /// it is destroyed (stack unwound and coroutine frame freed).
    std::atomic<bool> cancel_on_resume_{false};

    static constexpr int kNoPreferredNumaNode = std::numeric_limits<int>::min();

    /// \brief Preferred NUMA node recorded for the coroutine.
    std::atomic<int> preferred_numa_node_{kNoPreferredNumaNode};

    // debugging / DAG info
    std::string Name;  // function name of the coroutine
    uint64_t Instance; // instance number, global
    ExceptionSinkNode* exception_sinks_head_{nullptr};

#if UNI20_DEBUG_DAG
    // For debug tracking the DAG, we store the nodes of the incoming (ReadBuffer) and outgoing (WriteBuffer) objects.
    std::vector<NodeInfo const*> ReadDependencies;
    std::vector<NodeInfo const*> WriteDependencies;
#endif

    // /// \brief To propogate exceptions and cancellations to the appropriate awaiter, whenever we suspend
    // /// we stash the awaiter here, so we can pass on set_cancel() and set_exception()
    // AsyncAwaiter* current_awaiter_ = nullptr;
    //
    // void set_current_awaiter(AsyncAwaiter* a) { current_awaiter_ = a; }
    //
    // void set_cancel() noexcept { cancel_.store(true, std::memory_order_release); }
    //
    // void set_exception(std::exception_ptr e) noexcept
    // {
    //   eptr_ = e;
    //   // write to the atomic, mostly for the memory barrier
    //   awaiter_has_error_.store(false, std::memory_order_release);
    // }
    //
    // bool is_cancel_on_resume() const noexcept { return awaiter_has_error_.load(std::memory_order_acquire); }
    //
    // std::exception_ptr is_exception_on_resume() const noexcept
    // {
    //   return awaiter_has_error_.load(std::memory_order_acquire) ? nullptr : eptr_;
    // }

    template <typename... Args> BasicAsyncTaskPromise(Args&&... args)
    {
      // For each parameter, detect ReadBuffer / WriteBuffer
      (ProcessCoroutineArgument(this, args), ...);
    }

    /// \brief safely destroy this coroutine, returning the continuation_ (which also must now be destroyed)
    std::coroutine_handle<promise_type> destroy_with_continuation() noexcept
    {
      auto c = continuation_;
      continuation_ = nullptr;
      auto h = std::coroutine_handle<promise_type>::from_promise(*this);
      promise_type::destroy_and_track(h);
      return c;
    }

    ~BasicAsyncTaskPromise() noexcept { DEBUG_CHECK_EQUAL(continuation_, nullptr); }

    /// \brief Decrease the number of active awaiters by one.
    /// \return true if this was the last awaiter and the coroutine is now unowned.
    bool release_awaiter() noexcept { return awaiter_count_.fetch_sub(1, std::memory_order_acq_rel) == 1; }

    /// \brief Decrease the number of active awaiters by a specified count.
    /// \param count The number of awaiters to release.
    /// \return true if the count reached zero exactly as a result of this call.
    bool release_awaiter(int count) noexcept
    {
      return awaiter_count_.fetch_sub(count, std::memory_order_acq_rel) == count;
    }

    /// \brief Increase the number of active awaiters by one.
    /// \return The value of the counter prior to the increment.
    int add_awaiter() noexcept { return awaiter_count_.fetch_add(1, std::memory_order_relaxed); }

    /// \brief Increase the number of active awaiters by a specified count.
    /// \param count The number of awaiters to add.
    /// \return The value of the counter prior to the increment.
    int add_awaiter(int count) noexcept { return awaiter_count_.fetch_add(count, std::memory_order_relaxed); }

    void set_exception(std::exception_ptr e) noexcept
    {
      if (!exception_.exchange(true, std::memory_order_acq_rel))
      {
        eptr_ = e;
      }
    }

    /// \brief Register one exception propagation sink with this promise.
    /// \param node Intrusive node owned by a buffer object.
    /// \param epoch Epoch that should receive unhandled coroutine exceptions.
    /// \param explicit_sink true when registered via propagate_exceptions_to(...).
    void register_exception_sink(ExceptionSinkNode& node, std::shared_ptr<EpochContext> epoch, bool explicit_sink)
    {
      if (node.owner) node.owner->unregister_exception_sink(node, false);
      if (!epoch) return;

      node.owner = this;
      node.epoch = std::move(epoch);
      node.explicit_sink = explicit_sink;
      node.prev = nullptr;
      node.next = exception_sinks_head_;
      if (exception_sinks_head_) exception_sinks_head_->prev = &node;
      exception_sinks_head_ = &node;
    }

    /// \brief Unregister one exception propagation sink from this promise.
    /// \param node Intrusive node currently linked into this promise.
    /// \param from_destructor true when called from buffer destruction.
    void unregister_exception_sink(ExceptionSinkNode& node, bool from_destructor) noexcept
    {
      if (node.owner != this) return;
      if (from_destructor && node.explicit_sink && std::uncaught_exceptions() > 0)
      {
        CHECK(false,
              "propagate_exceptions_to sink destroyed during exception unwinding before coroutine unhandled_exception()");
      }

      if (node.prev)
        node.prev->next = node.next;
      else
        exception_sinks_head_ = node.next;

      if (node.next) node.next->prev = node.prev;

      node.owner = nullptr;
      node.prev = nullptr;
      node.next = nullptr;
      node.epoch.reset();
      node.explicit_sink = false;
    }

    /// \brief Get the current exception pointer, or nullptr if there is no current exception.
    /// \pre caller must be the sole owner of the coroutine, in order to avoid race conditions with set_exception()
    std::exception_ptr get_exception() noexcept
    {
      if (exception_.load(std::memory_order_acquire))
        return eptr_;
      else
        return nullptr;
    }

    /// \brief Throw the current exception, if there is one.
    /// \pre caller must be the sole owner of the coroutine, in order to avoid race conditions with set_exception()
    void rethrow_exception()
    {
      if (exception_.load(std::memory_order_acquire)) std::rethrow_exception(eptr_);
    }

    void set_cancel_on_resume() noexcept { cancel_on_resume_.store(true, std::memory_order_release); }

    bool cancel_on_resume() const noexcept { return cancel_on_resume_.load(std::memory_order_acquire); }

    /// \brief Transform the awaiter to provide transfer of ownership of the AsyncTask
    template <AsyncTaskAwaitable A> auto await_transform(A& a);
    template <AsyncTaskAwaitable A> auto await_transform(A&& a);

    /// \brief Transform the awaiter to provide transfer of shared ownership of the AsyncTaskFactory
    template <AsyncTaskFactoryAwaitable A> auto await_transform(A& a);
    template <AsyncTaskFactoryAwaitable A> auto await_transform(A&& a);

    // Pass-through for AsyncTask itself
    template <IsAsyncTaskPromise Promise> BasicAsyncTask<Promise>& await_transform(BasicAsyncTask<Promise>& t) noexcept
    {
      return t;
    }

    template <IsAsyncTaskPromise Promise>
    BasicAsyncTask<Promise>&& await_transform(BasicAsyncTask<Promise>&& t) noexcept
    {
      return std::move(t);
    }

    template <typename T> auto await_transform(T&&)
    {
      static_assert(
          sizeof(T) == 0,
          "co_await expression does not match any known AsyncTaskAwaitable or AsyncTaskFactoryAwaitable type.");
    }

    /// \brief Set the preferred NUMA node recorded for this coroutine.
    /// \param node Preferred node value, or empty to clear the preference.
    void set_preferred_numa_node(std::optional<int> node) noexcept
    {
      preferred_numa_node_.store(node ? *node : kNoPreferredNumaNode, std::memory_order_release);
    }

    /// \brief Retrieve the preferred NUMA node for this coroutine.
    /// \return Optional containing the preferred node, if one was recorded.
    std::optional<int> preferred_numa_node() const noexcept
    {
      int node = preferred_numa_node_.load(std::memory_order_acquire);
      if (node == kNoPreferredNumaNode) return std::nullopt;
      return node;
    }

    /// \brief Mark the coroutine as having been scheduled to start executing.
    void mark_started() noexcept { started_.store(true, std::memory_order_release); }

    /// \brief Query whether the coroutine has begun executing.
    bool has_started() const noexcept { return started_.load(std::memory_order_acquire); }

    /// \brief Record that a coroutine has transitioned into a runnable/running state.
    static void note_running(std::coroutine_handle<promise_type> h) noexcept { TaskRegistry::mark_running(h); }

    /// \brief Record that a coroutine has suspended and is waiting for resumption.
    static void note_suspended(std::coroutine_handle<promise_type> h) noexcept { TaskRegistry::mark_suspended(h); }

    /// \brief Record that a coroutine has been intentionally leaked.
    static void note_leaked(std::coroutine_handle<promise_type> h) noexcept { TaskRegistry::leak_task(h); }

    /// \brief Resume a coroutine handle while recording the running transition.
    static void resume_and_track(std::coroutine_handle<promise_type> h)
    {
      note_running(h);
      h.resume();
    }

    /// \brief Destroy a coroutine handle while recording destruction in the registry.
    static void destroy_and_track(std::coroutine_handle<promise_type> h) noexcept
    {
      TaskRegistry::destroy_task(h);
      h.destroy();
    }

    /// \brief Resolve an await_suspend AsyncTask return value into a scheduler/runtime transfer handle.
    /// \param h Current coroutine handle.
    /// \param t Returned task object from awaitable.await_suspend(...).
    /// \return Handle selected for immediate transfer by coroutine semantics.
    static std::coroutine_handle<> resolve_await_suspend_result(std::coroutine_handle<promise_type> h, AsyncTask&& t)
    {
      // null handle means suspend the current coroutine until externally resumed
      if (!t.h_)
      {
        note_suspended(h);
        return std::noop_coroutine();
      }

      // Transfer ownership into a raw coroutine_handle
      auto h_new = t.h_.promise().release_ownership();
      CHECK(h_new, "coroutine handle was not exclusively owned!");

      // If the same AsyncTask is given back to us, resume this coroutine immediately.
      if (h_new == h)
      {
        note_running(h);
        return h;
      }

      // Otherwise run a nested coroutine and continue back here once that coroutine completes.
      h_new.promise().continuation_ = h;
      note_suspended(h);
      note_running(h_new);
      return h_new;
    }

    /// \brief Execute await_suspend for AsyncTaskAwaitable and apply TaskRegistry tracking.
    template <AsyncTaskAwaitable A> static auto suspend_task_awaitable(std::coroutine_handle<promise_type> h, A& a);

    /// \brief Execute await_suspend for AsyncTaskFactoryAwaitable and apply TaskRegistry tracking.
    template <AsyncTaskFactoryAwaitable A>
    static auto suspend_factory_awaitable(std::coroutine_handle<promise_type> h, A& a);

    /// \brief Acquire exclusive ownership of the coroutine.
    ///       This increments the awaiter count and asserts that the coroutine was previously unowned.
    /// \pre The coroutine must be unowned (awaiter_count_ == 0).
    /// \return A newly constructed AsyncTask that takes ownership of the coroutine.
    AsyncTask take_ownership() noexcept
    {
      [[maybe_unused]] int prior_count = this->add_awaiter();
      DEBUG_CHECK_EQUAL(prior_count, 0, "expected handle to be previously unowned!");
      return AsyncTask(std::coroutine_handle<promise_type>::from_promise(*this));
    }

    /// \brief Acquire shared ownership of the coroutine for use with multi-await constructs.
    /// \param count The number of distinct AsyncTask instances to be created.
    /// \pre The coroutine must be unowned (awaiter_count_ == 0).
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
    constexpr BasicAsyncTaskPromise() noexcept = default;

    /// \brief Constructs the coroutine's return object for the caller.
    ///
    /// \note This is invoked exactly once, before initial_suspend().
    /// \return An `AsyncTask` owning the coroutine handle corresponding to this promise,
    ///         transferring lifetime responsibility to the caller.
    AsyncTask get_return_object() noexcept
    {
      auto h = std::coroutine_handle<promise_type>::from_promise(*this);
      this->add_awaiter();
      TaskRegistry::register_task(h);
      return AsyncTask(h);
    }

    /// \brief Suspend immediately on coroutine entry.
    auto initial_suspend() noexcept
    {
      struct InitialAwaiter
      {
          constexpr bool await_ready() noexcept { return false; }
          void await_suspend(std::coroutine_handle<AsyncTask::promise_type> h) noexcept
          {
            promise_type::note_suspended(h);
          }
          constexpr void await_resume() noexcept {}
      };
      return InitialAwaiter{};
    }

    /// \note At final_suspend the coroutine frame is owned exclusively by the coroutine.
    ///       The scheduler must not retain or access the coroutine_handle after resume().
    ///       This function eagerly destroys the coroutine and resumes its continuation.

    auto final_suspend() noexcept
    {
      struct FinalAwaiter
      {
          constexpr bool await_ready() noexcept { return false; }

          std::coroutine_handle<> await_suspend(std::coroutine_handle<AsyncTask::promise_type> h) noexcept
          {
            auto continuation = std::exchange(h.promise().continuation_, nullptr);
            bool cancelled = h.promise().cancel_on_resume();
            auto eptr = h.promise().get_exception();
            TRACE_MODULE(ASYNC, "Final suspend of coroutine", h, continuation, cancelled);

            promise_type::destroy_and_track(h);

            TRACE_MODULE(ASYNC, "Destroy is done");

            if (cancelled)
            {
              while (continuation)
                continuation = continuation.promise().destroy_with_continuation();
            }

            if (continuation)
            {
              if (eptr) continuation.promise().set_exception(eptr);
              promise_type::note_running(continuation);
              return continuation;
            }
            else
              return std::noop_coroutine();
          }

          void await_resume() noexcept {}
      };
      return FinalAwaiter{};
    }

    /// \brief Called when the coroutine returns normally.
    constexpr void return_void() noexcept {}

    /// \brief Called on unhandled exception escaping the coroutine.
    void unhandled_exception()
    {
      try
      {
        throw;
      }
      catch (task_cancelled const&)
      {
        this->set_cancel_on_resume();
      }
      catch (...)
      {
        auto eptr = std::current_exception();
        this->set_exception(eptr);
        for (auto* node = exception_sinks_head_; node; node = node->next)
          propagate_unhandled_writer_exception(node->epoch.get(), eptr);
      }
    }

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
        this->release_outstanding();
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
      DEBUG_TRACE_MODULE(ASYNC, this, handle_, count_);
      this->release_outstanding();
    }

  private:
    friend AsyncTask::promise_type;

    using HandleType = std::coroutine_handle<AsyncTask::promise_type>;

    /// \brief Construct a factory with N shared references to the coroutine.
    /// \pre The coroutine must be unowned (`awaiter_count_ == 0`) at the time of construction.
    /// \param h The coroutine handle.
    /// \param count The number of AsyncTask instances to dispense.
    AsyncTaskFactory(HandleType h, int count) : handle_(h), count_(count)
    {
      [[maybe_unused]] int prior_count = handle_.promise().add_awaiter(count);
      DEBUG_CHECK_EQUAL(prior_count, 0, "expected handle to be previously unowned!");
      // if we requested zero references, then we can destroy the handle immediately
      if (count_ == 0)
      {
        AsyncTask::promise_type::destroy_and_track(handle_);
        handle_ = {};
      }
    }

    /// \brief Release any undispatched ownership claims currently held by this factory.
    void release_outstanding() noexcept
    {
      // Return the outstanding references. If this reaches zero, destroy the coroutine.
      if (count_ > 0 && handle_.promise().release_awaiter(count_))
      {
        AsyncTask::promise_type::destroy_and_track(handle_);
      }
      count_ = 0;
      handle_ = {};
    }

    HandleType handle_;
    size_t count_;
};

inline AsyncTaskFactory BasicAsyncTaskPromise::take_shared_ownership(int count)
{
  return AsyncTaskFactory(std::coroutine_handle<promise_type>::from_promise(*this), count);
}

template <AsyncTaskAwaitable A>
auto BasicAsyncTaskPromise::suspend_task_awaitable(std::coroutine_handle<promise_type> h, A& a)
{
  using await_return_type = decltype(a.await_suspend(std::declval<AsyncTask>()));
  if constexpr (std::is_void_v<await_return_type>)
  {
    auto t = h.promise().take_ownership();
    a.await_suspend(std::move(t));
    promise_type::note_suspended(h);
    return;
  }
  else if constexpr (std::is_same_v<await_return_type, AsyncTask>)
  {
    auto t_owning = h.promise().take_ownership();
    auto t = a.await_suspend(std::move(t_owning));
    return promise_type::resolve_await_suspend_result(h, std::move(t));
  }
  else
  {
    static_assert(std::is_same_v<await_return_type, void>,
                  "Unsupported await_suspend() return type: must be void or AsyncTask");
  }
}

template <AsyncTaskFactoryAwaitable A>
auto BasicAsyncTaskPromise::suspend_factory_awaitable(std::coroutine_handle<promise_type> h, A& a)
{
  using await_return_type = decltype(a.await_suspend(std::declval<AsyncTaskFactory>()));
  if constexpr (std::is_void_v<await_return_type>)
  {
    auto factory = h.promise().take_shared_ownership(a.num_awaiters());
    a.await_suspend(std::move(factory));
    promise_type::note_suspended(h);
    return;
  }
  else if constexpr (std::is_same_v<await_return_type, AsyncTask>)
  {
    auto factory = h.promise().take_shared_ownership(a.num_awaiters());
    auto t = a.await_suspend(std::move(factory));
    return promise_type::resolve_await_suspend_result(h, std::move(t));
  }
  else
  {
    static_assert(std::is_same_v<await_return_type, void>,
                  "Unsupported await_suspend() return type: must be void or AsyncTask");
  }
}

/// \brief AsyncTaskAwaiter is a wrapper that manages the transfer of ownership from an AsyncTask into
/// a coroutine_handle and back again.
template <AsyncTaskAwaitable A> struct AsyncTaskAwaiter //: public AsyncAwaiter
{
    A awaitable;
    AsyncTask::promise_type& promise;

    bool await_ready() { return awaitable.await_ready(); }

    auto await_suspend(std::coroutine_handle<AsyncTask::promise_type> h)
    {
      return AsyncTask::promise_type::suspend_task_awaitable(h, awaitable);
    }

    decltype(auto) await_resume()
    {
      if constexpr (requires { awaitable.register_exception_sinks(promise); })
      {
        awaitable.register_exception_sinks(promise);
      }
      // we can call await_resume on a moved awaitable here, because this is the last time
      // that we refer to awaitable
      return awaitable.await_resume();
    }

    // decltype(auto) await_resume() &
    // {
    //   // we can call await_resume on a moved awaitable here, because this is the last time
    //   // that we refer to awaitable
    //   return awaitable.await_resume();
    // }
    //
    // decltype(auto) await_resume() &&
    // {
    //   // we can call await_resume on a moved awaitable here, because this is the last time
    //   // that we refer to awaitable
    //   return std::move(awaitable).await_resume();
    // }

    // void set_cancel() override final { awaitable.set_cancel(); }
    //
    // void set_exception(std::exception_ptr e) override final { awaitable.set_exception(e); }
};

// Process an argument of the coroutine.  By default we do nothing
template <typename T> void ProcessCoroutineArgument(BasicAsyncTaskPromise* promise, T const&) {}

template <AsyncTaskFactoryAwaitable A> struct AsyncTaskFactoryAwaiter //: public AsyncAwaiter
{
    A awaitable;
    AsyncTask::promise_type& promise;

    bool await_ready() { return awaitable.await_ready(); }

    auto await_suspend(std::coroutine_handle<AsyncTask::promise_type> h)
    {
      return AsyncTask::promise_type::suspend_factory_awaitable(h, awaitable);
    }

    decltype(auto) await_resume() { return awaitable.await_resume(); }

    // void set_cancel() override final { awaitable.set_cancel(); }
    //
    // void set_exception(std::exception_ptr e) override final { awaitable.set_exception(e); }
};
} // namespace uni20::async
