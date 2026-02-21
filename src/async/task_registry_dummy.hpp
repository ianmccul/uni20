#pragma once

#include <coroutine>
#include <vector>

namespace uni20 {

class TaskRegistry {
public:
    enum class TaskState {
        Constructed,
        Running,
        Suspended,
        Leaked,
    };

    enum class EpochTaskRole {
        Reader,
        Writer,
    };

    static constexpr void register_task(std::coroutine_handle<>) noexcept {}
    static constexpr void destroy_task(std::coroutine_handle<>) noexcept {}
    static constexpr void leak_task(std::coroutine_handle<>) noexcept {}
    static constexpr void mark_running(std::coroutine_handle<>) noexcept {}
    static constexpr void mark_suspended(std::coroutine_handle<>) noexcept {}
    static constexpr void register_epoch_context(void const*) noexcept {}
    static constexpr void destroy_epoch_context(void const*) noexcept {}
    static constexpr void bind_epoch_task(void const*, std::coroutine_handle<>, EpochTaskRole) noexcept {}
    static constexpr void unbind_epoch_task(void const*, std::coroutine_handle<>, EpochTaskRole) noexcept {}
    static std::vector<std::coroutine_handle<>> epoch_reader_tasks(void const*) { return {}; }
    static std::vector<std::coroutine_handle<>> epoch_writer_tasks(void const*) { return {}; }
    static constexpr void dump() noexcept {}
};

} // namespace uni20
