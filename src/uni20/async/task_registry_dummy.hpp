#pragma once

/**
 * \file task_registry_dummy.hpp
 * \brief Provides a no-op task registry used when async debug tracking is disabled.
 */

#include <coroutine>
#include <vector>

namespace uni20::async
{
class EpochContext;
}

namespace uni20
{

/// \brief No-op task registry implementation for non-debug async builds.
class TaskRegistry {
  public:
    /// \brief Verbosity levels mirrored from the debug registry interface.
    enum class DumpMode
    {
      None,
      Basic,
      Full,
    };

    /// \brief Lifecycle states mirrored from the debug registry interface.
    enum class TaskState
    {
      Constructed,
      Running,
      Suspended,
      Leaked,
    };

    /// \brief Reader/writer role mirrored from the debug registry interface.
    enum class EpochTaskRole
    {
      Reader,
      Writer,
    };

    /// \brief No-op task registration hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void register_task(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op task destruction hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void destroy_task(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op task leak hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void leak_task(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op running-state hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void mark_running(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op suspended-state hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void mark_suspended(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op epoch-context registration hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    static constexpr void register_epoch_context(async::EpochContext const* epoch_context) noexcept
    {
      static_cast<void>(epoch_context);
    }
    /// \brief No-op epoch-context destruction hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    static constexpr void destroy_epoch_context(async::EpochContext const* epoch_context) noexcept
    {
      static_cast<void>(epoch_context);
    }
    /// \brief No-op epoch-task binding hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \param h Coroutine handle ignored in dummy mode.
    /// \param role Role value ignored in dummy mode.
    static constexpr void bind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                          EpochTaskRole role) noexcept
    {
      static_cast<void>(epoch_context);
      static_cast<void>(h);
      static_cast<void>(role);
    }
    /// \brief No-op epoch-task unbinding hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \param h Coroutine handle ignored in dummy mode.
    /// \param role Role value ignored in dummy mode.
    static constexpr void unbind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                            EpochTaskRole role) noexcept
    {
      static_cast<void>(epoch_context);
      static_cast<void>(h);
      static_cast<void>(role);
    }
    /// \brief Returns an empty reader list in dummy mode.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \return Empty vector.
    static std::vector<std::coroutine_handle<>> epoch_reader_tasks(async::EpochContext const* epoch_context)
    {
      static_cast<void>(epoch_context);
      return {};
    }
    /// \brief Returns an empty writer list in dummy mode.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \return Empty vector.
    static std::vector<std::coroutine_handle<>> epoch_writer_tasks(async::EpochContext const* epoch_context)
    {
      static_cast<void>(epoch_context);
      return {};
    }
    /// \brief Returns `DumpMode::None` in dummy mode.
    /// \return Always `DumpMode::None`.
    static constexpr DumpMode dump_mode() noexcept { return DumpMode::None; }
    /// \brief No-op epoch-context dump hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \param reason Optional reason string ignored in dummy mode.
    static constexpr void dump_epoch_context(async::EpochContext const* epoch_context, char const* reason = nullptr) noexcept
    {
      static_cast<void>(epoch_context);
      static_cast<void>(reason);
    }
    /// \brief No-op global dump hook.
    static constexpr void dump() noexcept {}
};

} // namespace uni20
