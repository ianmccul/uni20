#pragma once

/**
 * \file task_awaiters.hpp
 * \ingroup backend_cublas
 * \brief Non-blocking CUDA-task acquisition of cuBLAS execution resources.
 */

#include <uni20/async/cuda_task.hpp>
#include <uni20/backend/cublas/execution.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>

#include <atomic>
#include <optional>

namespace uni20::cublas
{

/// \brief Awaiter that reserves a cuBLAS handle before waiting for an idle stream.
class ExecutionAcquireAwaiter : public async::CudaTaskAwaiterTag,
                                private cuda::detail::ResourceWaiter<HandleSlot>,
                                private cuda::detail::StreamWaiter {
  public:
    explicit ExecutionAcquireAwaiter(ExecutionPool& pool) noexcept
        : cuda::detail::ResourceWaiter<HandleSlot>(&ExecutionAcquireAwaiter::handle_available),
          cuda::detail::StreamWaiter(&ExecutionAcquireAwaiter::stream_available), pool_(&pool)
    {}

    ExecutionAcquireAwaiter(ExecutionAcquireAwaiter const&) = delete;
    ExecutionAcquireAwaiter& operator=(ExecutionAcquireAwaiter const&) = delete;
    ExecutionAcquireAwaiter(ExecutionAcquireAwaiter&&) = delete;
    ExecutionAcquireAwaiter& operator=(ExecutionAcquireAwaiter&&) = delete;

    [[nodiscard]] bool await_ready()
    {
      handle_ = pool_->handles_.try_acquire();
      if (!handle_) return false;
      stream_ = pool_->streams_->try_acquire();
      return stream_.has_value();
    }

    bool await_suspend(std::coroutine_handle<async::CudaTaskPromise> handle) noexcept
    {
      auto current = async::TaskHandle::from(handle);
      state_.store(registration_state::registering, std::memory_order_relaxed);
      task_ = handle.promise().take_ownership();
      async::TaskPromiseBase::note_suspended(current);

      this->advance();

      auto const prior = state_.exchange(registration_state::suspended, std::memory_order_acq_rel);
      if (prior == registration_state::ready) async::BasicTask::reschedule(std::move(task_));
      return true;
    }

    [[nodiscard]] ExecutionLease await_resume()
    {
      CHECK(handle_.has_value());
      CHECK(stream_.has_value());

      int current_device = -1;
      cuda::check(cudaGetDevice(&current_device), "cudaGetDevice after cuBLAS execution-resource acquisition");
      CHECK_EQUAL(current_device, pool_->device(), "CUDA task resumed outside the cuBLAS pool's device");
      return pool_->make_lease(std::move(*handle_), std::move(*stream_));
    }

  private:
    enum class registration_state
    {
      registering,
      suspended,
      ready
    };

    void advance() noexcept
    {
      if (!handle_)
      {
        auto handle = pool_->handles_.acquire_or_enqueue(static_cast<cuda::detail::ResourceWaiter<HandleSlot>&>(*this));
        if (!handle) return;
        handle_ = std::move(*handle);
      }

      auto stream = pool_->streams_->acquire_or_enqueue(static_cast<cuda::detail::StreamWaiter&>(*this));
      if (!stream) return;
      stream_ = std::move(*stream);
      this->publish_ready();
    }

    void publish_ready() noexcept
    {
      auto const prior = state_.exchange(registration_state::ready, std::memory_order_acq_rel);
      if (prior == registration_state::suspended) async::BasicTask::reschedule(std::move(task_));
    }

    static void handle_available(cuda::detail::ResourceWaiter<HandleSlot>& waiter,
                                 cuda::ResourceLease<HandleSlot> handle) noexcept
    {
      auto& self = static_cast<ExecutionAcquireAwaiter&>(waiter);
      self.handle_ = std::move(handle);
      self.advance();
    }

    static void stream_available(cuda::detail::StreamWaiter& waiter, cuda::Stream stream) noexcept
    {
      auto& self = static_cast<ExecutionAcquireAwaiter&>(waiter);
      self.stream_ = std::move(stream);
      self.publish_ready();
    }

    ExecutionPool* pool_;
    std::optional<cuda::ResourceLease<HandleSlot>> handle_;
    std::optional<cuda::Stream> stream_;
    async::BasicTask task_;
    std::atomic<registration_state> state_{registration_state::registering};
};

/// \brief Acquire a cuBLAS handle and idle stream without blocking a CUDA task.
[[nodiscard]] inline ExecutionAcquireAwaiter acquire_execution(ExecutionPool& pool) noexcept
{
  return ExecutionAcquireAwaiter(pool);
}

} // namespace uni20::cublas
