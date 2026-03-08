/// \file async_task.hpp
/// \brief Defines CudaTask, the fire-and-forget coroutine handle.

#pragma once

#include "async_task_promise.hpp"
#include <uni20/common/trace.hpp>

namespace uni20::async
{

/// \brief Promise type for CudaTask, forward declare
struct CudaTaskPromise : public AsyncTaskPromise
{
    // CUDA-specific data here
};

/// \brief A fire-and-forget coroutine handle.
class CudaTask : public BasicAsyncTask<CudaTaskPromise> {};

} // namespace uni20::async
