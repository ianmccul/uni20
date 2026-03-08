#pragma once

/**
 * \file task_registry_debug.hpp
 * \brief Declares the debug task registry used for async coroutine diagnostics.
 */

#include <coroutine>
#include <vector>

namespace uni20::async
{
class EpochContext;
}

namespace uni20
{

/// \brief Forward declaration for the debug task registry type.
class TaskRegistryDebug;

/// \brief Alias selecting the active task registry implementation.
using TaskRegistry = TaskRegistryDebug;

/// \brief Tracks coroutine and epoch-context lifecycle events in debug builds.
class TaskRegistryDebug {
  public:
    /// \brief Verbosity levels for registry dump output.
    enum class DumpMode
    {
      None,
      Basic,
      Full,
    };

    /// \brief Logical lifecycle states tracked for each coroutine handle.
    enum class TaskState
    {
      Constructed,
      Running,
      Suspended,
      Leaked,
    };

    /// \brief Role of a coroutine relative to an epoch context.
    enum class EpochTaskRole
    {
      Reader,
      Writer,
    };

    /// \brief Registers a newly-created coroutine handle.
    /// \param h Coroutine handle to register.
    static void register_task(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as destroyed.
    /// \param h Coroutine handle being destroyed.
    static void destroy_task(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as leaked.
    /// \param h Coroutine handle intentionally leaked.
    static void leak_task(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as currently running.
    /// \param h Coroutine handle that resumed execution.
    static void mark_running(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as suspended.
    /// \param h Coroutine handle that suspended execution.
    static void mark_suspended(std::coroutine_handle<> h);
    /// \brief Registers an epoch context for debug tracking.
    /// \param epoch_context Epoch context pointer to register.
    static void register_epoch_context(async::EpochContext const* epoch_context);
    /// \brief Removes an epoch context from debug tracking.
    /// \param epoch_context Epoch context pointer to remove.
    static void destroy_epoch_context(async::EpochContext const* epoch_context);
    /// \brief Associates a coroutine with an epoch context as reader or writer.
    /// \param epoch_context Epoch context being accessed.
    /// \param h Coroutine handle to bind.
    /// \param role Reader or writer role for the binding.
    static void bind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                EpochTaskRole role);
    /// \brief Removes a coroutine-role binding from an epoch context.
    /// \param epoch_context Epoch context that owns the binding.
    /// \param h Coroutine handle to unbind.
    /// \param role Reader or writer role being removed.
    static void unbind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                  EpochTaskRole role);
    /// \brief Returns coroutine handles currently bound as readers.
    /// \param epoch_context Epoch context to inspect.
    /// \return Reader coroutine handles for the requested epoch context.
    static std::vector<std::coroutine_handle<>> epoch_reader_tasks(async::EpochContext const* epoch_context);
    /// \brief Returns coroutine handles currently bound as writers.
    /// \param epoch_context Epoch context to inspect.
    /// \return Writer coroutine handles for the requested epoch context.
    static std::vector<std::coroutine_handle<>> epoch_writer_tasks(async::EpochContext const* epoch_context);
    /// \brief Reports the current debug dump mode.
    /// \return Active dump-mode setting.
    static DumpMode dump_mode() noexcept;
    /// \brief Prints diagnostics for a specific epoch context.
    /// \param epoch_context Epoch context to print.
    /// \param reason Optional label for the dump trigger.
    static void dump_epoch_context(async::EpochContext const* epoch_context, char const* reason = nullptr);
    /// \brief Prints global task-registry diagnostics.
    static void dump();
};

} // namespace uni20
