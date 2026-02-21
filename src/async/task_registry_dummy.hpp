#pragma once

#include <coroutine>

namespace uni20 {

class TaskRegistry {
public:
    enum class TaskState {
        Constructed,
        Running,
        Suspended,
        Leaked,
    };

    static constexpr void register_task(std::coroutine_handle<>) noexcept {}
    static constexpr void destroy_task(std::coroutine_handle<>) noexcept {}
    static constexpr void leak_task(std::coroutine_handle<>) noexcept {}
    static constexpr void mark_running(std::coroutine_handle<>) noexcept {}
    static constexpr void mark_suspended(std::coroutine_handle<>) noexcept {}
    static constexpr void dump() noexcept {}
};

} // namespace uni20
