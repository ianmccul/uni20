/**
 * \file async_task_impl.hpp
 * \brief Inline implementation of promise-neutral task ownership and nesting.
 */

#pragma once

namespace uni20::async
{

inline void BasicTask::reschedule(BasicTask task)
{
  TRACE_MODULE(ASYNC, "BasicTask::reschedule", &task, task.coroutine_handle());
  task = BasicTask::make_sole_owner(std::move(task));
  if (task)
  {
    auto* scheduler = task.handle_.promise().scheduler();
    DEBUG_CHECK(scheduler, "unexpected: task scheduler is not set!");
    scheduler->reschedule(std::move(task));
  }
}

inline BasicTask BasicTask::make_sole_owner(BasicTask&& task)
{
  DEBUG_CHECK(task.handle_);
  auto& promise = task.handle_.promise();
  if (promise.release_awaiter())
  {
    promise.add_awaiter();
  }
  else
  {
    task.handle_ = {};
  }
  return std::move(task);
}

inline bool BasicTask::can_destroy_coroutine(TaskHandle handle) const noexcept
{
  if (!handle) return true;
  return handle.promise().is_cancel_on_resume() || handle.done();
}

inline TaskHandle BasicTask::release_ownership()
{
  CHECK(handle_);
  CHECK(handle_.promise().release_awaiter(), "coroutine handle was not exclusively owned!");
  return std::exchange(handle_, {});
}

inline TaskHandle BasicTask::release_handle()
{
  TRACE_MODULE(ASYNC, "BasicTask::release_handle", this->coroutine_handle());
  CHECK(handle_);
  if (!handle_.promise().release_awaiter()) PANIC("Attempt to resume() a non-exclusive BasicTask");

  bool const destroy_on_resume = handle_.promise().is_cancel_on_resume();
  auto handle = std::exchange(handle_, {});

  if (destroy_on_resume)
  {
    CHECK(this->can_destroy_coroutine(handle), "unexpected destruction of an active task without cancellation",
          handle.coroutine());
    while (handle)
    {
      TRACE_MODULE(ASYNC, "Destroying task due to cancellation", handle.coroutine());
      handle = handle.promise().destroy_with_continuation();
    }
  }
  else
  {
    handle.promise().mark_started();
  }
  return handle;
}

inline void BasicTask::resume()
{
  auto handle = this->release_handle();
  TRACE_MODULE(ASYNC, "Resuming task", handle.coroutine());
  if (handle) TaskPromiseBase::resume_and_track(handle);
}

inline void BasicTask::abandon_leak()
{
  auto handle = this->release_handle();
  TaskPromiseBase::note_leaked(handle);
  TRACE_MODULE(ASYNC, "Abandoning task handle", handle.coroutine());
}

inline void BasicTask::set_cancel_on_resume() noexcept
{
  TRACE_MODULE(ASYNC, "Setting cancel flag on coroutine", this, this->coroutine_handle());
  handle_.promise().set_cancel_on_resume();
}

inline void BasicTask::exception_on_resume(std::exception_ptr error) noexcept
{
  handle_.promise().set_exception(error);
}

inline BasicTask& BasicTask::operator=(BasicTask&& other) noexcept
{
  TRACE_MODULE(ASYNC, "BasicTask move assignment", this, this->coroutine_handle(), &other, other.coroutine_handle());
  if (this != &other)
  {
    if (handle_ && handle_.promise().release_awaiter()) this->destroy_owned_coroutine();
    handle_ = std::exchange(other.handle_, {});
    await_exception_ = std::exchange(other.await_exception_, {});
  }
  return *this;
}

inline void BasicTask::destroy_owned_coroutine() noexcept
{
  auto handle = handle_;
  CHECK(this->can_destroy_coroutine(handle), "unexpected destruction of an active task without cancellation", this,
        handle.coroutine());
  while (handle)
  {
    DEBUG_TRACE_MODULE(ASYNC, "BasicTask destructor is destroying the coroutine", this, handle.coroutine());
    handle = handle.promise().destroy_with_continuation();
  }
  handle_ = {};
}

inline void BasicTask::release() noexcept
{
  TRACE_MODULE(ASYNC, "BasicTask::release", this, this->coroutine_handle());
  if (handle_ && handle_.promise().release_awaiter()) this->destroy_owned_coroutine();
}

inline BasicTask::~BasicTask() noexcept { this->release(); }

inline void BasicTask::destroy() noexcept
{
  CHECK(handle_);
  CHECK(handle_.promise().release_awaiter(), "Attempt to destroy a non-exclusive BasicTask");
  this->destroy_owned_coroutine();
}

inline bool BasicTask::set_scheduler(IScheduler* scheduler)
{
  if (!handle_) return false;
  TaskPromiseBase::validate_scheduler_route(scheduler, task_route(handle_));
  handle_.promise().sched_ = scheduler;
  return true;
}

inline std::optional<int> BasicTask::preferred_numa_node() const noexcept
{
  if (!handle_) return std::nullopt;
  return handle_.promise().preferred_numa_node();
}

inline void BasicTask::set_preferred_numa_node(std::optional<int> node) noexcept
{
  if (handle_) handle_.promise().set_preferred_numa_node(node);
}

template <TaskPromise ParentPromise>
std::coroutine_handle<> BasicTask::await_suspend(std::coroutine_handle<ParentPromise> parent_handle)
{
  DEBUG_CHECK(handle_);
  DEBUG_CHECK(!handle_.promise().continuation_);

  auto parent = TaskHandle::from(parent_handle);
  TaskPromiseBase::prepare_nested_route(parent, handle_);

  TaskPromiseBase::note_suspended(parent);
  handle_.promise().continuation_ = parent;
  handle_.promise().continuation_exception_ = &await_exception_;

  if (!TaskPromiseBase::can_transfer_directly(parent, handle_))
  {
    BasicTask::reschedule(std::move(*this));
    return std::noop_coroutine();
  }

  handle_.promise().mark_started();
  auto transfer = this->release_ownership();
  TaskPromiseBase::note_running(transfer);
  return transfer.coroutine();
}

inline void BasicTask::await_resume()
{
  if (auto error = std::exchange(await_exception_, {})) std::rethrow_exception(error);
  if (handle_) handle_.promise().rethrow_exception();
}

} // namespace uni20::async
