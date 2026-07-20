/**
 * \file async_task_promise.hpp
 * \brief Defines shared and concrete task coroutine promises.
 */

#pragma once

#include "async_errors.hpp"
#include "async_node.hpp"
#include "async_task.hpp"
#include "scheduler.hpp"
#include <atomic>
#include <coroutine>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

namespace uni20::async
{

class TaskFactory;
class EpochContext;

/// \brief Marker base for awaiters that require a `CudaTaskPromise` handle.
struct CudaTaskAwaiterTag
{};

/// \brief Identifies an explicitly opted-in CUDA task awaiter.
template <typename T>
concept CudaTaskAwaitable = std::derived_from<std::remove_cvref_t<T>, CudaTaskAwaiterTag>;

/// \brief Inject an unhandled writer-side coroutine exception into an epoch.
/// \param epoch Target epoch context associated with a writer buffer.
/// \param eptr Exception captured by the coroutine promise.
void propagate_unhandled_writer_exception(EpochContext* epoch, std::exception_ptr eptr) noexcept;

/// \brief Concept for the valid return types of await_suspend.
/// \details
/// An awaiter may return void, or it can return a BasicTask, which means that
/// execution of the current coroutine should be transferred to the new task,
/// resuming the coroutine only after the new task is complete.
/// If a BasicTask is returned, it must have exclusive ownership, otherwise the
/// task cannot be scheduled
template <typename Ret>
concept AwaitSuspendResult = std::same_as<Ret, void> || std::same_as<Ret, BasicTask>;

/// \brief Concept for awaitables that accept promise-neutral task ownership.
///
/// This concept is satisfied if the awaitable provides:
/// - `await_suspend(BasicTask)`
/// - The return type of `await_suspend` is `void` or `BasicTask`
///
/// \note This concept disallows await_suspend() from returning a coroutine_handle,
///       to ensure that ownership and resumption are managed solely by the scheduler.
template <typename T>
concept TaskAwaitable = requires(T a, BasicTask task) {
  { a.await_suspend(std::move(task)) } -> AwaitSuspendResult;
};

/// \brief Awaitable that can hold one ownership claim in a multi-input join.
/// \details A join child must retain or release the supplied parent task. It
///          cannot request nested task transfer because the parent has other
///          outstanding ownership claims in the same join.
template <typename T>
concept TaskFactoryChildAwaitable = requires(T a, BasicTask task) {
  { a.await_suspend(std::move(task)) } -> std::same_as<void>;
};

/// \brief Concept for awaitables that support shared promise-neutral ownership.
///
/// This concept is satisfied if:
/// - The awaitable provides a `num_awaiters()` method returning an integer count
/// - It provides `await_suspend(TaskFactory)`
/// - The return type of `await_suspend` is `void`
///
/// \note This is used by composite awaiters like `all(...)` that must split
///       ownership across multiple sub-awaitables.
template <typename T>
concept TaskFactoryAwaitable = requires(T a, TaskFactory factory) {
  { a.await_suspend(std::move(factory)) } -> std::same_as<void>;
  { a.num_awaiters() } -> std::convertible_to<int>;
};

/// \brief Forwarding awaiter that transfers one task ownership claim to an awaitable.
template <TaskAwaitable A> struct TaskAwaiter;

/// \brief Forwarding awaiter that transfers shared task ownership through a TaskFactory.
template <TaskFactoryAwaitable A> struct TaskFactoryAwaiter;

/// \brief Shared non-polymorphic implementation for Uni20 task promises.
class TaskPromiseBase {
  public:
    /// \brief Intrusive node describing one exception propagation sink.
    struct ExceptionSinkNode
    {
        TaskPromiseBase* owner{nullptr};
        ExceptionSinkNode* prev{nullptr};
        ExceptionSinkNode* next{nullptr};
        std::shared_ptr<EpochContext> epoch{};
        bool explicit_sink{false};
    };

    /// \brief Non-owning scheduler route used when the coroutine is ready to resume.
    /// \note The scheduler must outlive this coroutine and every possible later resumption.
    IScheduler* sched_ = nullptr;

    /// \brief Tracks whether the coroutine has been scheduled or otherwise started.
    std::atomic<bool> started_{false};

    /// \brief Uni20 task coroutines are nestable.
    /// \details continuation_ tracks the parent coroutine so final suspension
    ///          can return through the parent's scheduler route.
    TaskHandle continuation_{};

    /// \brief Destination for an exception returned by a directly awaited task.
    /// \details This points into the awaiting BasicTask stored in the suspended
    ///          parent frame and is cleared before the child frame is destroyed.
    std::exception_ptr* continuation_exception_ = nullptr;

    /// \brief Number of active ownership claims on this coroutine.
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

    /// \brief Construct the promise and process coroutine arguments for debug metadata.
    /// \tparam Args Coroutine argument types.
    /// \param args Coroutine arguments forwarded for `ProcessCoroutineArgument`.
    template <typename... Args> TaskPromiseBase(Args&&... args)
    {
      // For each parameter, detect ReadBuffer / WriteBuffer
      (ProcessCoroutineArgument(this, args), ...);
    }

    /// \brief safely destroy this coroutine, returning the continuation_ (which also must now be destroyed)
    TaskHandle destroy_with_continuation() noexcept
    {
      auto c = continuation_;
      continuation_ = {};
      continuation_exception_ = nullptr;
      auto handle = this->self();
      TaskPromiseBase::destroy_and_track(handle);
      return c;
    }

    ~TaskPromiseBase() noexcept
    {
      DEBUG_CHECK(!continuation_);
      DEBUG_CHECK(continuation_exception_ == nullptr);
    }

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

    /// \brief Record an exception for deferred rethrow on resume.
    /// \param e Exception pointer to record.
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
        CHECK(
            false,
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

    /// \brief Mark the coroutine for cancellation at next resume.
    void set_cancel_on_resume() noexcept { cancel_on_resume_.store(true, std::memory_order_release); }

    /// \brief Reports whether cancellation-on-resume is currently set.
    /// \return `true` when cancellation is requested.
    [[nodiscard]] bool is_cancel_on_resume() const noexcept
    {
      return cancel_on_resume_.load(std::memory_order_acquire);
    }

    /// \brief Transform an awaiter to transfer one task ownership claim.
    template <TaskAwaitable A> auto await_transform(A& a);
    template <TaskAwaitable A> auto await_transform(A&& a);

    /// \brief Transform the awaiter to provide transfer of shared ownership of the TaskFactory
    template <TaskFactoryAwaitable A> auto await_transform(A& a);
    template <TaskFactoryAwaitable A> auto await_transform(A&& a);

    /// \brief Pass through a concrete task that uses the shared basic task representation.
    /// \tparam Task Concrete task or basic task reference type.
    /// \param task Task being awaited.
    /// \return Forwarded task reference.
    template <typename Task>
      requires std::derived_from<std::remove_cvref_t<Task>, BasicTask>
    Task&& await_transform(Task&& task) noexcept
    {
      return std::forward<Task>(task);
    }

    /// \brief Fallback await_transform overload that rejects unsupported awaitables.
    /// \tparam T Unsupported awaitable type.
    template <typename T> auto await_transform(T&&)
    {
      static_assert(sizeof(T) == 0,
                    "co_await expression does not match any known TaskAwaitable or TaskFactoryAwaitable type.");
    }

    /// \brief Set the preferred NUMA node recorded for this coroutine.
    /// \param node Preferred node value, or empty to clear the preference.
    void set_preferred_numa_node(std::optional<int> node) noexcept
    {
      preferred_numa_node_.store(node ? *node : kNoPreferredNumaNode, std::memory_order_release);
    }

    /// \brief Retrieve the preferred NUMA node for this coroutine.
    /// \return Optional containing the preferred node, if one was recorded.
    [[nodiscard]] std::optional<int> preferred_numa_node() const noexcept
    {
      int node = preferred_numa_node_.load(std::memory_order_acquire);
      if (node == kNoPreferredNumaNode) return std::nullopt;
      return node;
    }

    /// \brief Mark the coroutine as having been scheduled to start executing.
    void mark_started() noexcept { started_.store(true, std::memory_order_release); }

    /// \brief Query whether the coroutine has begun executing.
    [[nodiscard]] bool has_started() const noexcept { return started_.load(std::memory_order_acquire); }

    /// \brief Return the scheduler currently recorded for this task.
    [[nodiscard]] IScheduler* scheduler() const noexcept { return sched_; }

    /// \brief Return this promise's erased coroutine identity.
    [[nodiscard]] TaskHandle self() noexcept
    {
      DEBUG_PRECONDITION(self_);
      return self_;
    }

    /// \brief Record that a coroutine has transitioned into a runnable/running state.
    static void note_running(TaskHandle handle) noexcept { TaskRegistry::mark_running(handle.coroutine()); }

    /// \brief Record that a coroutine has suspended and is waiting for resumption.
    static void note_suspended(TaskHandle handle) noexcept { TaskRegistry::mark_suspended(handle.coroutine()); }

#if UNI20_DEBUG_DAG
    /// \brief Records a DAG edge for awaitables that expose debug node metadata.
    /// \tparam A Awaitable type being observed.
    /// \param h Coroutine handle that is awaiting.
    /// \param awaitable Awaitable being awaited.
    template <typename A> static void note_await_dependency(TaskHandle handle, A const& awaitable)
    {
      auto record = [coroutine = handle.coroutine()](NodeInfo const* node, TaskRegistry::EpochTaskRole role) {
        TaskRegistry::record_await_dependency(coroutine, node, role);
      };
      if constexpr (requires { awaitable.debug_each_dependency(record); })
      {
        awaitable.debug_each_dependency(record);
      }
      else if constexpr (requires {
                           awaitable.node();
                           awaitable.debug_task_role();
                         })
      {
        TaskRegistry::record_await_dependency(handle.coroutine(), awaitable.node(), awaitable.debug_task_role());
      }
    }
#endif

    /// \brief Record that a coroutine has been intentionally leaked.
    static void note_leaked(TaskHandle handle) noexcept { TaskRegistry::leak_task(handle.coroutine()); }

    /// \brief Resume a coroutine handle while recording the running transition.
    static void resume_and_track(TaskHandle handle)
    {
      note_running(handle);
      handle.coroutine().resume();
    }

    /// \brief Destroy a coroutine handle while recording destruction in the registry.
    static void destroy_and_track(TaskHandle handle) noexcept
    {
      TaskRegistry::destroy_task(handle.coroutine());
      handle.coroutine().destroy();
    }

    /// \brief Complete an unbound child's route before nested execution.
    /// \param parent Currently executing parent task.
    /// \param child Child selected for execution.
    static void prepare_nested_route(TaskHandle parent, TaskHandle child);

    /// \brief Report whether symmetric transfer is valid for two prepared task routes.
    /// \param from Currently executing task.
    /// \param to Task that would execute next on the current thread.
    [[nodiscard]] static bool can_transfer_directly(TaskHandle from, TaskHandle to) noexcept;

    /// \brief Resolve an await_suspend task return value into a coroutine transfer handle.
    /// \param current Current coroutine identity.
    /// \param task Returned ownership token from awaitable.await_suspend(...).
    /// \return Handle selected for immediate transfer by coroutine semantics.
    static std::coroutine_handle<> resolve_await_suspend_result(TaskHandle current, BasicTask&& task)
    {
      if (!task)
      {
        note_suspended(current);
        return std::noop_coroutine();
      }

      auto nested = task.release_ownership();
      CHECK(nested, "coroutine handle was not exclusively owned!");

      if (nested == current)
      {
        note_running(current);
        return current.coroutine();
      }

      prepare_nested_route(current, nested);
      nested.promise().continuation_ = current;
      note_suspended(current);

      if (!can_transfer_directly(current, nested))
      {
        auto nested_task = nested.promise().take_ownership();
        BasicTask::reschedule(std::move(nested_task));
        return std::noop_coroutine();
      }

      nested.promise().mark_started();
      note_running(nested);
      return nested.coroutine();
    }

    /// \brief Execute await_suspend for TaskAwaitable and apply TaskRegistry tracking.
    template <TaskAwaitable A> static auto suspend_task_awaitable(TaskHandle current, A& awaitable);

    /// \brief Execute await_suspend for TaskFactoryAwaitable and apply TaskRegistry tracking.
    template <TaskFactoryAwaitable A> static void suspend_factory_awaitable(TaskHandle current, A& awaitable);

    /// \brief Acquire exclusive ownership of the coroutine.
    ///       This increments the awaiter count and asserts that the coroutine was previously unowned.
    /// \pre The coroutine must be unowned (awaiter_count_ == 0).
    /// \return A newly constructed BasicTask that takes ownership of the coroutine.
    BasicTask take_ownership() noexcept
    {
      [[maybe_unused]] int prior_count = this->add_awaiter();
      DEBUG_CHECK_EQUAL(prior_count, 0, "expected handle to be previously unowned!");
      return BasicTask(this->self(), BasicTask::construction_key{});
    }

    /// \brief Acquire shared ownership of the coroutine for use with multi-await constructs.
    /// \param count The number of distinct BasicTask ownership claims to create.
    /// \pre The coroutine must be unowned (awaiter_count_ == 0).
    /// \return A factory that will dispense up to \p count owning BasicTask values.
    TaskFactory take_shared_ownership(int count);

    /// \brief Release ownership of the coroutine and return it, if it was exclusively owned
    /// \return the coroutine handle, if we had exclusive ownership; otherwise returns null
    /// \post The promise no longer owns the released claim.
    TaskHandle release_ownership() { return this->release_awaiter() ? this->self() : TaskHandle{}; }

    /// \brief Default-construct the promise.
    constexpr TaskPromiseBase() noexcept = default;

    /// \brief Cache erased self-identity and register the initial ownership claim.
    template <TaskPromise Promise>
    [[nodiscard]] TaskHandle initialize_task(std::coroutine_handle<Promise> handle) noexcept
    {
      DEBUG_PRECONDITION(!self_);
      self_ = TaskHandle::from(handle);
      this->add_awaiter();
      TaskRegistry::register_task(self_.coroutine());
#if UNI20_DEBUG_DAG
      TaskRegistry::record_task_dependencies(self_.coroutine(), ReadDependencies, WriteDependencies);
#endif
      return self_;
    }

    /// \brief Suspend immediately on coroutine entry.
    auto initial_suspend() noexcept
    {
      struct InitialAwaiter
      {
          TaskPromiseBase* promise;

          [[nodiscard]] constexpr bool await_ready() noexcept { return false; }
          void await_suspend(std::coroutine_handle<>) noexcept { TaskPromiseBase::note_suspended(promise->self()); }
          constexpr void await_resume() noexcept {}
      };
      return InitialAwaiter{this};
    }

    /// \note At final_suspend the coroutine frame is owned exclusively by the coroutine.
    ///       The scheduler must not retain or access the coroutine_handle after resume().
    ///       This function eagerly destroys the coroutine and resumes its continuation.

    auto final_suspend() noexcept
    {
      struct FinalAwaiter
      {
          TaskPromiseBase* promise;

          [[nodiscard]] constexpr bool await_ready() noexcept { return false; }

          std::coroutine_handle<> await_suspend(std::coroutine_handle<>) noexcept
          {
            auto completed = promise->self();
            auto continuation = std::exchange(promise->continuation_, {});
            auto* continuation_exception = std::exchange(promise->continuation_exception_, nullptr);
            bool cancelled = promise->is_cancel_on_resume();
            auto eptr = promise->get_exception();
            bool const direct_transfer =
                continuation && TaskPromiseBase::can_transfer_directly(completed, continuation);
            TRACE_MODULE(ASYNC, "Final suspend of coroutine", completed.coroutine(), continuation.coroutine(),
                         cancelled);

            TaskPromiseBase::destroy_and_track(completed);

            TRACE_MODULE(ASYNC, "Destroy is done");

            if (cancelled)
            {
              while (continuation)
                continuation = continuation.promise().destroy_with_continuation();
            }

            if (continuation)
            {
              if (eptr)
              {
                if (continuation_exception)
                  *continuation_exception = eptr;
                else
                  continuation.promise().set_exception(eptr);
              }

              if (!direct_transfer)
              {
                auto continuation_task = continuation.promise().take_ownership();
                BasicTask::reschedule(std::move(continuation_task));
                return std::noop_coroutine();
              }

              TaskPromiseBase::note_running(continuation);
              return continuation.coroutine();
            }
            else
              return std::noop_coroutine();
          }

          void await_resume() noexcept {}
      };
      return FinalAwaiter{this};
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
        bool const originating_failure = !exception_.load(std::memory_order_acquire);
        auto eptr = std::current_exception();
        TaskRegistry::record_unhandled_exception(this->self().coroutine(), eptr, originating_failure);
        this->set_exception(eptr);
        for (auto* node = exception_sinks_head_; node; node = node->next)
          propagate_unhandled_writer_exception(node->epoch.get(), eptr);
      }
    }

    // /// \brief Called when the task is ready to run again.
    // /// \param h The coroutine handle to schedule.
    // void notify_ready(TaskHandle handle) { sched_->schedule(handle); }

  private:
    TaskHandle self_{};
};

/// \brief Promise type for ordinary AsyncTask coroutines.
class AsyncTaskPromise final : public TaskPromiseBase {
  public:
    static constexpr TaskDomain task_domain = TaskDomain::host;

    using TaskPromiseBase::TaskPromiseBase;

    [[nodiscard]] AsyncTask get_return_object() noexcept
    {
      auto handle = std::coroutine_handle<AsyncTaskPromise>::from_promise(*this);
      return AsyncTask(this->initialize_task(handle));
    }
};

/// \brief Promise type for CUDA-constrained CudaTask coroutines.
class CudaTaskPromise final : public TaskPromiseBase {
  public:
    static constexpr TaskDomain task_domain = TaskDomain::cuda;

    using TaskPromiseBase::TaskPromiseBase;
    using TaskPromiseBase::await_transform;

    [[nodiscard]] CudaTask get_return_object() noexcept;

    /// \brief Return the task's established CUDA device affinity, if any.
    /// \details An empty result is valid while the task performs device-neutral
    ///          work. CUDA schedulers route such activations through their
    ///          configured default device without changing the promise.
    [[nodiscard]] std::optional<int> device() const noexcept { return device_; }

    /// \brief Pass through an explicitly opted-in CUDA task awaiter.
    /// \tparam Awaiter CUDA backend awaiter type.
    /// \param awaiter Awaiter that requires the concrete CUDA promise handle.
    /// \return The forwarded awaiter.
    template <CudaTaskAwaitable Awaiter> Awaiter&& await_transform(Awaiter&& awaiter) noexcept
    {
      return std::forward<Awaiter>(awaiter);
    }

    /// \brief Bind the CUDA device before the coroutine starts.
    /// \param device Non-negative CUDA runtime device ordinal.
    /// \pre The task has not started and is either unbound or already bound to the same device.
    void bind_device(int device)
    {
      CHECK(device >= 0, "CUDA device ordinal must be non-negative", device);
      CHECK(!this->has_started(), "cannot bind a CUDA task device after the task has started");
      if (device_)
      {
        CHECK_EQUAL(*device_, device, "cannot rebind a CUDA task to another device");
        return;
      }
      device_ = device;
    }

    /// \brief Select the device used for subsequent CUDA task activations.
    /// \details The caller must exclusively own the suspended coroutine. Unlike
    ///          initial binding, this operation may change affinity after the
    ///          task has started.
    void select_device(int device)
    {
      CHECK(device >= 0, "CUDA device ordinal must be non-negative", device);
      device_ = device;
    }

  private:
    std::optional<int> device_{};
};

/// \brief Narrow an erased task promise after verifying that it is CUDA-specific.
/// \param handle Erased task identity produced from its original typed coroutine handle.
/// \return The concrete CUDA promise.
inline CudaTaskPromise& cuda_promise(TaskHandle handle) noexcept
{
  CHECK(handle);
  CHECK(handle.domain() == TaskDomain::cuda);
  return static_cast<CudaTaskPromise&>(handle.promise());
}

inline void TaskPromiseBase::prepare_nested_route(TaskHandle parent, TaskHandle child)
{
  CHECK(parent);
  CHECK(child);
  CHECK(parent.promise().scheduler(), "nested task parent has no scheduler route");

  if (!child.promise().scheduler())
  {
    CHECK(parent.domain() == child.domain(),
          "an unbound nested task cannot inherit a scheduler across task domains");
    child.promise().sched_ = parent.promise().scheduler();
  }

  if (child.domain() != TaskDomain::cuda) return;

  auto& child_promise = cuda_promise(child);
  if (child_promise.device() || parent.domain() != TaskDomain::cuda) return;

  if (auto const parent_device = cuda_promise(parent).device()) child_promise.bind_device(*parent_device);
}

inline bool TaskPromiseBase::can_transfer_directly(TaskHandle from, TaskHandle to) noexcept
{
  CHECK(from);
  CHECK(to);
  auto* scheduler = from.promise().scheduler();
  CHECK(scheduler, "running task has no scheduler route");

  if (scheduler != to.promise().scheduler()) return false;
  if (from.domain() != to.domain()) return false;
  return scheduler->can_direct_transfer(from, to);
}

/// \brief Factory for producing multiple BasicTask ownership claims for one coroutine.
///
/// This is used when multiple awaiters (e.g., in an all(A, B, C) construct) need to take independent ownership
/// of the same coroutine. The factory pre-allocates all references atomically and ensures that they are
/// handed out exactly once.
///
/// \note This class must only be created while the coroutine is unowned. Attempting to add awaiters after
///       the coroutine is active leads to race conditions and is undefined behavior.
class TaskFactory {
  public:
    /// \brief Dispense the next BasicTask from the pool of shared ownership handles.
    /// \pre `count_ > 0` — there must be remaining tasks to dispense.
    /// \post One fewer task will be available from this factory.
    /// \return A BasicTask that shares ownership of the coroutine.
    BasicTask take_next()
    {
      DEBUG_PRECONDITION(count_ > 0);
      --count_;
      return BasicTask(handle_, BasicTask::construction_key{});
    }

    TaskFactory(TaskFactory&& other) noexcept
        : handle_(std::exchange(other.handle_, {})), count_(std::exchange(other.count_, 0))
    {}

    /// \brief Move assignment.
    /// \param other Source factory.
    /// \return Reference to `*this`.
    TaskFactory& operator=(TaskFactory&& other) noexcept
    {
      if (this != &other)
      {
        this->release_outstanding();
        handle_ = std::exchange(other.handle_, {});
        count_ = std::exchange(other.count_, 0);
      }
      return *this;
    }

    TaskFactory(TaskFactory const&) = delete;
    TaskFactory& operator=(TaskFactory const&) = delete;

    /// \brief Destructor returns any unused ownership claims and may destroy the coroutine.
    ///
    /// If any BasicTask values were not handed out via `take_next()`, they are released here.
    /// If the coroutine is unowned after release, it will be destroyed.
    ~TaskFactory() noexcept
    {
      DEBUG_TRACE_MODULE(ASYNC, this, handle_.coroutine(), count_);
      this->release_outstanding();
    }

  private:
    friend class TaskPromiseBase;

    /// \brief Construct a factory with N shared references to the coroutine.
    /// \pre The coroutine must be unowned (`awaiter_count_ == 0`) at the time of construction.
    /// \param h The coroutine handle.
    /// \param count The number of BasicTask values to dispense.
    TaskFactory(TaskHandle handle, int count) : handle_(handle), count_(count)
    {
      [[maybe_unused]] int prior_count = handle_.promise().add_awaiter(count);
      DEBUG_CHECK_EQUAL(prior_count, 0, "expected handle to be previously unowned!");
      // if we requested zero references, then we can destroy the handle immediately
      if (count_ == 0)
      {
        TaskPromiseBase::destroy_and_track(handle_);
        handle_ = {};
      }
    }

    /// \brief Release any undispatched ownership claims currently held by this factory.
    void release_outstanding() noexcept
    {
      // Return the outstanding references. If this reaches zero, destroy the coroutine.
      if (count_ > 0 && handle_.promise().release_awaiter(count_))
      {
        TaskPromiseBase::destroy_and_track(handle_);
      }
      count_ = 0;
      handle_ = {};
    }

    TaskHandle handle_;
    size_t count_;
};

inline TaskFactory TaskPromiseBase::take_shared_ownership(int count) { return TaskFactory(this->self(), count); }

template <TaskAwaitable A> auto TaskPromiseBase::suspend_task_awaitable(TaskHandle current, A& awaitable)
{
#if UNI20_DEBUG_DAG
  TaskPromiseBase::note_await_dependency(current, awaitable);
#endif
  using await_return_type = decltype(awaitable.await_suspend(std::declval<BasicTask>()));
  if constexpr (std::is_void_v<await_return_type>)
  {
    auto task = current.promise().take_ownership();
    awaitable.await_suspend(std::move(task));
    TaskPromiseBase::note_suspended(current);
    return;
  }
  else if constexpr (std::is_same_v<await_return_type, BasicTask>)
  {
    auto owning_task = current.promise().take_ownership();
    auto task = awaitable.await_suspend(std::move(owning_task));
    return TaskPromiseBase::resolve_await_suspend_result(current, std::move(task));
  }
  else
  {
    static_assert(std::is_same_v<await_return_type, void>,
                  "Unsupported await_suspend() return type: must be void or BasicTask");
  }
}

template <TaskFactoryAwaitable A> void TaskPromiseBase::suspend_factory_awaitable(TaskHandle current, A& awaitable)
{
#if UNI20_DEBUG_DAG
  TaskPromiseBase::note_await_dependency(current, awaitable);
#endif
  auto factory = current.promise().take_shared_ownership(awaitable.num_awaiters());
  awaitable.await_suspend(std::move(factory));
  TaskPromiseBase::note_suspended(current);
}

/// \brief Forward an awaitable using promise-neutral task ownership.
template <TaskAwaitable A> struct TaskAwaiter //: public AsyncAwaiter
{
    A awaitable;
    TaskPromiseBase& promise;

    /// \brief Checks whether the wrapped awaitable is ready.
    /// \return `true` when no suspension is needed.
    [[nodiscard]] bool await_ready() { return awaitable.await_ready(); }

    /// \brief Suspend using promise-neutral ownership-transfer semantics.
    /// \tparam Promise Concrete enclosing promise type.
    /// \param handle Current coroutine handle.
    /// \return Transfer handle selected by suspend logic.
    template <TaskPromise Promise> auto await_suspend(std::coroutine_handle<Promise> handle)
    {
      return TaskPromiseBase::suspend_task_awaitable(TaskHandle::from(handle), awaitable);
    }

    /// \brief Resume wrapped awaitable and register explicit exception sinks if provided.
    /// \return Result produced by the wrapped awaitable.
    [[nodiscard]] decltype(auto) await_resume()
    {
#if UNI20_DEBUG_DAG
      TaskPromiseBase::note_await_dependency(promise.self(), awaitable);
#endif
      if constexpr (requires { awaitable.register_exception_sinks(promise); })
      {
        awaitable.register_exception_sinks(promise);
      }
      if constexpr (std::is_lvalue_reference_v<A>)
      {
        return awaitable.await_resume();
      }
      else
      {
        return std::move(awaitable).await_resume();
      }
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

/// \brief Process a coroutine argument for debug metadata (default no-op).
/// \tparam T Argument type.
/// \param promise Promise receiving metadata.
/// \param value Argument ignored by default.
template <typename T> void ProcessCoroutineArgument(TaskPromiseBase* promise, T const& value)
{
  static_cast<void>(promise);
  static_cast<void>(value);
}

template <TaskFactoryAwaitable A> struct TaskFactoryAwaiter //: public AsyncAwaiter
{
    A awaitable;
    TaskPromiseBase& promise;

    /// \brief Checks whether the wrapped awaitable is ready.
    /// \return `true` when no suspension is needed.
    [[nodiscard]] bool await_ready() { return awaitable.await_ready(); }

    /// \brief Suspend using shared-ownership await-suspend semantics.
    /// \tparam Promise Concrete enclosing promise type.
    /// \param handle Current coroutine handle.
    template <TaskPromise Promise> void await_suspend(std::coroutine_handle<Promise> handle)
    {
      TaskPromiseBase::suspend_factory_awaitable(TaskHandle::from(handle), awaitable);
    }

    /// \brief Resume wrapped awaitable and return its await result.
    /// \return Result produced by the wrapped awaitable.
    [[nodiscard]] decltype(auto) await_resume()
    {
#if UNI20_DEBUG_DAG
      TaskPromiseBase::note_await_dependency(promise.self(), awaitable);
#endif
      if constexpr (std::is_lvalue_reference_v<A>)
      {
        return awaitable.await_resume();
      }
      else
      {
        return std::move(awaitable).await_resume();
      }
    }

    // void set_cancel() override final { awaitable.set_cancel(); }
    //
    // void set_exception(std::exception_ptr e) override final { awaitable.set_exception(e); }
};

template <TaskPromise Promise> TaskHandle TaskHandle::from(std::coroutine_handle<Promise> handle) noexcept
{
  if (!handle) return {};
  return TaskHandle(std::coroutine_handle<>::from_address(handle.address()), std::addressof(handle.promise()),
                    Promise::task_domain);
}

inline TaskPromiseBase& TaskHandle::promise() const noexcept
{
  DEBUG_PRECONDITION(promise_ != nullptr);
  return *promise_;
}

} // namespace uni20::async

#include "async_task_impl.hpp"
