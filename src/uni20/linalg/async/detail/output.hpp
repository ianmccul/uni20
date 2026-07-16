#pragma once

/**
 * \file output.hpp
 * \ingroup linalg
 * \brief Shared output capabilities for Async Tensor operation wrappers.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <memory>
#include <utility>

namespace uni20::detail
{

/// \brief Mutable Tensor output supported by async operation wrappers.
/// \details Async aliases must be copyable because their writer exposes a
///          const bound descriptor that the coroutine copies before dispatch.
template <class Tensor>
concept AsyncTensorOutput =
    MutableTensorView<Tensor> && (!async::is_async_alias_v<Tensor> || std::copy_constructible<Tensor>);

/// \brief Await a fixed async alias writer and expose its bound descriptor.
/// \details The descriptor remains read-only in storage so the alias cannot be
///          retargeted. A wrapper copies it locally to obtain mutable element
///          access through the existing binding.
template <class Alias> class AsyncAliasWriteDescriptorAwaiter {
  public:
    explicit AsyncAliasWriteDescriptorAwaiter(async::WriteBuffer<Alias>& output) : output_(std::addressof(output)) {}

    [[nodiscard]] bool await_ready() const noexcept { return output_->await_ready(); }

    void await_suspend(async::AsyncTask task) noexcept { output_->await_suspend(std::move(task)); }

    [[nodiscard]] Alias const& await_resume() { return output_->await_resume().get(); }

#if UNI20_DEBUG_DAG
    [[nodiscard]] auto node() const { return output_->node(); }
    static constexpr auto debug_task_role() noexcept { return TaskRegistry::EpochTaskRole::Writer; }
#endif

  private:
    async::WriteBuffer<Alias>* output_;
};

} // namespace uni20::detail
