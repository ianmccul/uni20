#include "config.hpp"
#include "task_registry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <mutex>
#if UNI20_HAS_STACKTRACE
#include <stacktrace>
#endif
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

using TaskState = uni20::TaskRegistry::TaskState;

char const* to_string(TaskState state) noexcept
{
  switch (state)
  {
  case TaskState::Constructed:
    return "constructed";
  case TaskState::Running:
    return "running";
  case TaskState::Suspended:
    return "suspended";
  case TaskState::Leaked:
    return "leaked";
  }

  return "unknown";
}

std::string format_timestamp(std::chrono::system_clock::time_point timestamp)
{
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(timestamp.time_since_epoch()) % std::chrono::seconds(1);
  auto const time = std::chrono::system_clock::to_time_t(timestamp);
  auto const local_time = fmt::localtime(time);
  return fmt::format("{:%F %T}.{:06} {:%z}", local_time, us.count(), local_time);
}

#if UNI20_HAS_STACKTRACE
void print_stacktrace(std::stacktrace const& trace)
{
  for (auto const& frame : trace)
  {
    if (frame.source_line() > 0)
      fmt::print(stderr, "    {} ({}:{})\n", frame.description(), frame.source_file(), frame.source_line());
    else
      fmt::print(stderr, "    {}\n", frame.description());
  }
}
#endif

struct TaskDebugInfo
{
    TaskState state{TaskState::Constructed};
    std::size_t transition_count{0};
    std::chrono::system_clock::time_point creation_timestamp{};
    std::chrono::system_clock::time_point last_state_change_timestamp{};
    std::string waiting_on{};
#if UNI20_HAS_STACKTRACE
    std::stacktrace creation_trace{};
    std::stacktrace last_state_change_trace{};
#endif
};

class TaskRegistryImpl {
  public:
    void register_task(std::coroutine_handle<> h)
    {
      if (!h) return;
      std::lock_guard lock(mutex_);
      auto [it, inserted] = tasks_.try_emplace(h.address());
      if (inserted)
      {
        auto const now = std::chrono::system_clock::now();
        it->second.creation_timestamp = now;
#if UNI20_HAS_STACKTRACE
        auto const trace = std::stacktrace::current(2);
        it->second.creation_trace = trace;
        this->update_state_locked(it->second, TaskState::Constructed, now, trace);
#else
        this->update_state_locked(it->second, TaskState::Constructed, now);
#endif
      }
    }

    void destroy_task(std::coroutine_handle<> h)
    {
      if (!h) return;
      std::lock_guard lock(mutex_);
      tasks_.erase(h.address());
    }

    void leak_task(std::coroutine_handle<> h) { this->set_state(h, TaskState::Leaked); }

    void mark_running(std::coroutine_handle<> h) { this->set_state(h, TaskState::Running); }

    void mark_suspended(std::coroutine_handle<> h) { this->set_state(h, TaskState::Suspended); }

    void dump()
    {
      std::lock_guard lock(mutex_);

      fmt::print(stderr, "\n========== Async Task Registry Dump ==========\n");
      fmt::print(stderr, "Total tracked tasks: {}\n\n", tasks_.size());
#if !UNI20_HAS_STACKTRACE
      fmt::print(stderr,
                 "WARNING: std::stacktrace is unavailable; dump output is degraded to state-only information.\n\n");
#endif

      std::vector<std::pair<void const*, TaskDebugInfo const*>> sorted_tasks;
      sorted_tasks.reserve(tasks_.size());
      for (auto const& [addr, info] : tasks_)
        sorted_tasks.emplace_back(addr, &info);

      std::sort(sorted_tasks.begin(), sorted_tasks.end(), [](auto const& lhs, auto const& rhs) {
        return reinterpret_cast<std::uintptr_t>(lhs.first) < reinterpret_cast<std::uintptr_t>(rhs.first);
      });

      std::size_t task_number = 1;
      for (auto const& [addr, info_ptr] : sorted_tasks)
      {
        auto const& info = *info_ptr;
        fmt::print(stderr, "Task {}:\n", task_number);
        fmt::print(stderr, "  task pointer: {}\n", addr);
        fmt::print(stderr, "  transition count: {}\n", info.transition_count);
        fmt::print(stderr, "  current state: {}\n", to_string(info.state));
        fmt::print(stderr, "  creation timestamp: {}\n", format_timestamp(info.creation_timestamp));

        if (!info.waiting_on.empty()) fmt::print(stderr, "  waiting on: {}\n", info.waiting_on);

#if UNI20_HAS_STACKTRACE
        fmt::print(stderr, "  creation stacktrace:\n");
        print_stacktrace(info.creation_trace);
        fmt::print(stderr, "  last state-change: {}\n", to_string(info.state));
        fmt::print(stderr, "  last state-change timestamp: {}\n", format_timestamp(info.last_state_change_timestamp));
        fmt::print(stderr, "  last state-change stacktrace:\n");
        print_stacktrace(info.last_state_change_trace);
#else
        fmt::print(stderr, "  creation stacktrace: unavailable\n");
        fmt::print(stderr, "  last state-change: {}\n", to_string(info.state));
        fmt::print(stderr, "  last state-change timestamp: {}\n", format_timestamp(info.last_state_change_timestamp));
        fmt::print(stderr, "  last state-change stacktrace: unavailable\n");
#endif
        fmt::print(stderr, "\n");
        ++task_number;
      }

      fmt::print(stderr, "================================================\n");
    }

    static TaskRegistryImpl& instance()
    {
      static TaskRegistryImpl* inst = new TaskRegistryImpl(); // intentional leak
      return *inst;
    }

  private:
    void set_state(std::coroutine_handle<> h, TaskState state)
    {
      if (!h) return;
      std::lock_guard lock(mutex_);
      auto it = tasks_.find(h.address());
      if (it == tasks_.end()) return;
      auto const now = std::chrono::system_clock::now();
#if UNI20_HAS_STACKTRACE
      auto const trace = std::stacktrace::current(2);
      this->update_state_locked(it->second, state, now, trace);
#else
      this->update_state_locked(it->second, state, now);
#endif
    }

    void update_state_locked(TaskDebugInfo& info, TaskState state, std::chrono::system_clock::time_point timestamp
#if UNI20_HAS_STACKTRACE
                             ,
                             std::stacktrace const& trace
#endif
    )
    {
      info.state = state;
      ++info.transition_count;
      info.last_state_change_timestamp = timestamp;
#if UNI20_HAS_STACKTRACE
      info.last_state_change_trace = trace;
#endif
    }

    std::mutex mutex_;
    std::unordered_map<void*, TaskDebugInfo> tasks_;
};

} // anonymous namespace

namespace uni20
{

void TaskRegistry::register_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().register_task(h); }

void TaskRegistry::destroy_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().destroy_task(h); }

void TaskRegistry::leak_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().leak_task(h); }

void TaskRegistry::mark_running(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_running(h); }

void TaskRegistry::mark_suspended(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_suspended(h); }

void TaskRegistry::dump() { TaskRegistryImpl::instance().dump(); }

} // namespace uni20
