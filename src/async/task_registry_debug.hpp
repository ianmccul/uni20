#pragma once

#include <coroutine>

namespace uni20
{

class TaskRegistryDebug;

using TaskRegistry = TaskRegistryDebug;

class TaskRegistryDebug {
  public:
    enum class TaskState {
        Constructed,
        Running,
        Suspended,
        Leaked,
    };

    static void register_task(std::coroutine_handle<> h);
    static void destroy_task(std::coroutine_handle<> h);
    static void leak_task(std::coroutine_handle<> h);
    static void mark_running(std::coroutine_handle<> h);
    static void mark_suspended(std::coroutine_handle<> h);
    static void dump();
};

} // namespace uni20
