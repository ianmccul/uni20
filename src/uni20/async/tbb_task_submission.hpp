/**
 * \file tbb_task_submission.hpp
 * \brief Internal non-blocking oneTBB arena admission helper.
 */

#pragma once

#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <uni20/common/trace.hpp>

#include <exception>
#include <string_view>
#include <utility>

namespace uni20::async::detail
{

/// \brief Diagnostic context for fatal oneTBB task-admission failures.
struct TbbTaskAdmissionContext
{
    std::string_view scheduler;
    int device = -1;
};

[[noreturn]] inline void tbb_task_admission_failure(TbbTaskAdmissionContext context, std::string_view reason)
{
  if (context.device >= 0)
  {
    PANIC("oneTBB scheduler admission failed", context.scheduler, context.device, reason);
  }
  PANIC("oneTBB scheduler admission failed", context.scheduler, reason);
}

/// \brief Register a task-group activation and enqueue it without entering the arena.
/// \details `task_group::defer()` registers the activation with the group's wait
///          set before `task_arena::enqueue()` publishes it. This is the portable
///          spelling of `task_arena::enqueue(function, group)` for oneTBB 2021.x.
template <typename Function>
void enqueue_tbb_task(oneapi::tbb::task_arena& arena, oneapi::tbb::task_group& group, Function&& function,
                      TbbTaskAdmissionContext context) noexcept
{
  try
  {
    auto task = group.defer(std::forward<Function>(function));
    arena.enqueue(std::move(task));
  }
  catch (std::exception const& error)
  {
    tbb_task_admission_failure(context, error.what());
  }
  catch (...)
  {
    tbb_task_admission_failure(context, "unknown exception");
  }
}

} // namespace uni20::async::detail
