#pragma once

#include <coroutine>
#include <vector>

namespace uni20::async
{
class EpochContext;
}

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

    enum class EpochTaskRole {
        Reader,
        Writer,
    };

    static void register_task(std::coroutine_handle<> h);
    static void destroy_task(std::coroutine_handle<> h);
    static void leak_task(std::coroutine_handle<> h);
    static void mark_running(std::coroutine_handle<> h);
    static void mark_suspended(std::coroutine_handle<> h);
    static void register_epoch_context(async::EpochContext const* epoch_context);
    static void destroy_epoch_context(async::EpochContext const* epoch_context);
    static void bind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h, EpochTaskRole role);
    static void unbind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                  EpochTaskRole role);
    static std::vector<std::coroutine_handle<>> epoch_reader_tasks(async::EpochContext const* epoch_context);
    static std::vector<std::coroutine_handle<>> epoch_writer_tasks(async::EpochContext const* epoch_context);
    static void dump();
};

} // namespace uni20
