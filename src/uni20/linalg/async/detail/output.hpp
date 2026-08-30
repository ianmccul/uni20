#pragma once

/**
 * \file output.hpp
 * \ingroup linalg
 * \brief Shared output capabilities for Async Tensor operation wrappers.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async/concepts.hpp>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

/// \brief Await a fixed async alias writer and expose its bound descriptor.
/// \details The descriptor remains read-only in storage so the alias cannot be
///          retargeted. A wrapper copies it locally to obtain mutable element
///          access through the existing binding.
template <class Alias> class AsyncAliasWriteDescriptorAwaiter {
  public:
    explicit AsyncAliasWriteDescriptorAwaiter(async::WriteBuffer<Alias>& output) : output_(std::addressof(output)) {}

    [[nodiscard]] bool await_ready() const noexcept { return output_->await_ready(); }

    void await_suspend(async::BasicTask task) noexcept { output_->await_suspend(std::move(task)); }

    [[nodiscard]] Alias const& await_resume() { return output_->await_resume().get(); }

#if UNI20_DEBUG_DAG
    [[nodiscard]] auto node() const { return output_->node(); }
    static constexpr auto debug_task_role() noexcept { return TaskRegistry::EpochTaskRole::Writer; }
#endif

  private:
    async::WriteBuffer<Alias>* output_;
};

/// \brief Return the writer awaiter for a mutable async Tensor operand.
/// \details Independent values expose their shared storage so the operation
///          can diagnose an unconstructed workspace. Async aliases expose a
///          copied fixed descriptor and cannot be rebound.
template <AsyncTensorOutput Tensor> [[nodiscard]] auto mutable_async_tensor_awaiter(async::WriteBuffer<Tensor>& output)
{
  if constexpr (async::is_async_alias_v<Tensor>)
    return AsyncAliasWriteDescriptorAwaiter<Tensor>{output};
  else
    return output.storage();
}

/// \brief Resolve an awaited mutable async Tensor operand.
/// \details Async aliases are copied to obtain mutable element access through
///          the fixed binding. Independent values remain owned by their writer
///          storage and must already be constructed.
template <AsyncTensorOutput Tensor, class Awaited>
[[nodiscard]] decltype(auto) mutable_async_tensor_value(Awaited&& awaited)
{
  if constexpr (async::is_async_alias_v<Tensor>)
  {
    return Tensor(std::forward<Awaited>(awaited));
  }
  else
  {
    if (!awaited.constructed()) throw async::buffer_write_uninitialized{};
    return *awaited;
  }
}

/// \brief Return an existing scalar output or default-construct its owner.
template <AsyncTensorOutput Tensor>
[[nodiscard]] Tensor& prepare_async_scalar_output(async::shared_storage<Tensor>& storage)
{
  if (storage.constructed()) return *storage;
  if constexpr (std::default_initializable<Tensor>)
    return storage.emplace();
  else
    throw async::buffer_write_uninitialized{};
}

} // namespace detail
} // namespace uni20
