/// \file cuda_task.hpp
/// \brief Defines the CUDA task admission type and scheduler interface.

#pragma once

#include "async_task_promise.hpp"

namespace uni20::async
{

/// \brief CUDA-oriented initial-admission type sharing the canonical task promise.
class CudaTask final : public BasicTask {
  public:
    using base_type = BasicTask;
    using promise_type = BasicAsyncTaskPromise;
    using handle_type = base_type::handle_type;

    CudaTask() = default;
    CudaTask(CudaTask const&) = delete;
    CudaTask& operator=(CudaTask const&) = delete;
    CudaTask(CudaTask&&) noexcept = default;
    CudaTask& operator=(CudaTask&&) noexcept = default;

    /// \brief Assign an optional debug label while preserving the concrete task type.
    /// \param label Human-readable label used only by debug diagnostics.
    /// \return Reference to this task for call chaining.
    CudaTask& debug_name(std::string const& label)
    {
      this->base_type::debug_name(label);
      return *this;
    }

  private:
    struct construction_key
    {};

    CudaTask(handle_type handle, construction_key) noexcept : base_type(handle) {}

    friend class BasicTaskReturnObject;
};

inline BasicTaskReturnObject::operator CudaTask() && noexcept
{
  return CudaTask(this->release(), CudaTask::construction_key{});
}

/// \brief Scheduler interface for initial `CudaTask` admission.
class ICudaScheduler : public IScheduler {
  public:
    ~ICudaScheduler() override = default;

    /// \brief Bind and submit a CUDA task for initial execution.
    /// \param task CUDA task to admit.
    virtual void schedule(CudaTask&& task) = 0;
};

} // namespace uni20::async
