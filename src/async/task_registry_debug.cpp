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
#include <unordered_set>
#include <vector>

namespace
{

using TaskState = uni20::TaskRegistry::TaskState;
using EpochTaskRole = uni20::TaskRegistry::EpochTaskRole;

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

char const* to_string(EpochTaskRole role) noexcept
{
  switch (role)
  {
  case EpochTaskRole::Reader:
    return "reader";
  case EpochTaskRole::Writer:
    return "writer";
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

struct EpochTaskBinding
{
    void const* epoch_context{nullptr};
    EpochTaskRole role{EpochTaskRole::Reader};
};

struct EpochDebugInfo
{
    std::chrono::system_clock::time_point creation_timestamp{};
    std::unordered_set<void*> reader_tasks{};
    std::unordered_set<void*> writer_tasks{};
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
      void* task_addr = h.address();
      tasks_.erase(task_addr);

      auto binding_it = task_bindings_.find(task_addr);
      if (binding_it != task_bindings_.end())
      {
        for (auto const& binding : binding_it->second)
        {
          auto context_it = epoch_contexts_.find(binding.epoch_context);
          if (context_it != epoch_contexts_.end())
            this->task_set_for_role(context_it->second, binding.role).erase(task_addr);
        }
        task_bindings_.erase(binding_it);
      }
    }

    void leak_task(std::coroutine_handle<> h) { this->set_state(h, TaskState::Leaked); }

    void mark_running(std::coroutine_handle<> h) { this->set_state(h, TaskState::Running); }

    void mark_suspended(std::coroutine_handle<> h) { this->set_state(h, TaskState::Suspended); }

    void register_epoch_context(void const* epoch_context)
    {
      if (!epoch_context) return;
      std::lock_guard lock(mutex_);
      auto [it, inserted] = epoch_contexts_.try_emplace(epoch_context);
      if (inserted) it->second.creation_timestamp = std::chrono::system_clock::now();
    }

    void destroy_epoch_context(void const* epoch_context)
    {
      if (!epoch_context) return;
      std::lock_guard lock(mutex_);
      auto context_it = epoch_contexts_.find(epoch_context);
      if (context_it == epoch_contexts_.end()) return;

      for (auto const& task_addr : context_it->second.reader_tasks)
        this->erase_task_epoch_binding_locked(task_addr, epoch_context, EpochTaskRole::Reader);
      for (auto const& task_addr : context_it->second.writer_tasks)
        this->erase_task_epoch_binding_locked(task_addr, epoch_context, EpochTaskRole::Writer);

      epoch_contexts_.erase(context_it);
    }

    void bind_epoch_task(void const* epoch_context, std::coroutine_handle<> h, EpochTaskRole role)
    {
      if (!epoch_context || !h) return;
      std::lock_guard lock(mutex_);
      auto& context = epoch_contexts_[epoch_context];
      if (context.creation_timestamp == std::chrono::system_clock::time_point{})
        context.creation_timestamp = std::chrono::system_clock::now();

      void* task_addr = h.address();
      auto& task_set = this->task_set_for_role(context, role);
      task_set.insert(task_addr);

      auto& bindings = task_bindings_[task_addr];
      if (!this->has_task_epoch_binding(bindings, epoch_context, role))
        bindings.push_back(EpochTaskBinding{epoch_context, role});
    }

    void unbind_epoch_task(void const* epoch_context, std::coroutine_handle<> h, EpochTaskRole role)
    {
      if (!epoch_context || !h) return;
      std::lock_guard lock(mutex_);
      auto context_it = epoch_contexts_.find(epoch_context);
      if (context_it != epoch_contexts_.end())
      {
        this->task_set_for_role(context_it->second, role).erase(h.address());
      }
      this->erase_task_epoch_binding_locked(h.address(), epoch_context, role);
    }

    std::vector<std::coroutine_handle<>> epoch_reader_tasks(void const* epoch_context)
    {
      std::lock_guard lock(mutex_);
      return this->epoch_tasks_locked(epoch_context, EpochTaskRole::Reader);
    }

    std::vector<std::coroutine_handle<>> epoch_writer_tasks(void const* epoch_context)
    {
      std::lock_guard lock(mutex_);
      return this->epoch_tasks_locked(epoch_context, EpochTaskRole::Writer);
    }

    void dump()
    {
      std::lock_guard lock(mutex_);

      fmt::print(stderr, "\n========== Async Task Registry Dump ==========\n");
      fmt::print(stderr, "Total tracked tasks: {}\n", tasks_.size());
      fmt::print(stderr, "Total tracked epoch contexts: {}\n\n", epoch_contexts_.size());
#if !UNI20_HAS_STACKTRACE
      fmt::print(stderr,
                 "WARNING: std::stacktrace is unavailable; dump output is degraded to state-only information.\n\n");
#endif

      std::vector<std::pair<void*, TaskDebugInfo const*>> sorted_tasks;
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
        if (info.state == TaskState::Suspended)
          this->print_task_epoch_bindings_locked(addr);

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
    static bool has_task_epoch_binding(std::vector<EpochTaskBinding> const& bindings, void const* epoch_context,
                                       EpochTaskRole role)
    {
      return std::any_of(bindings.begin(), bindings.end(), [&](EpochTaskBinding const& binding) {
        return binding.epoch_context == epoch_context && binding.role == role;
      });
    }

    static std::unordered_set<void*>& task_set_for_role(EpochDebugInfo& info, EpochTaskRole role)
    {
      return (role == EpochTaskRole::Reader) ? info.reader_tasks : info.writer_tasks;
    }

    static std::unordered_set<void*> const& task_set_for_role(EpochDebugInfo const& info, EpochTaskRole role)
    {
      return (role == EpochTaskRole::Reader) ? info.reader_tasks : info.writer_tasks;
    }

    void erase_task_epoch_binding_locked(void* task_addr, void const* epoch_context, EpochTaskRole role)
    {
      auto binding_it = task_bindings_.find(task_addr);
      if (binding_it == task_bindings_.end()) return;

      auto& bindings = binding_it->second;
      bindings.erase(std::remove_if(bindings.begin(), bindings.end(), [&](EpochTaskBinding const& binding) {
                       return binding.epoch_context == epoch_context && binding.role == role;
                     }),
                     bindings.end());

      if (bindings.empty()) task_bindings_.erase(binding_it);
    }

    std::vector<std::coroutine_handle<>> epoch_tasks_locked(void const* epoch_context, EpochTaskRole role) const
    {
      std::vector<std::coroutine_handle<>> handles;
      auto context_it = epoch_contexts_.find(epoch_context);
      if (context_it == epoch_contexts_.end()) return handles;

      auto const& task_set = this->task_set_for_role(context_it->second, role);
      handles.reserve(task_set.size());
      for (auto const& task_addr : task_set)
      {
        if (tasks_.contains(task_addr)) handles.push_back(std::coroutine_handle<>::from_address(task_addr));
      }

      std::sort(handles.begin(), handles.end(), [](auto lhs, auto rhs) {
        return reinterpret_cast<std::uintptr_t>(lhs.address()) < reinterpret_cast<std::uintptr_t>(rhs.address());
      });
      return handles;
    }

    void print_task_epoch_bindings_locked(void* task_addr) const
    {
      auto binding_it = task_bindings_.find(task_addr);
      if (binding_it == task_bindings_.end() || binding_it->second.empty()) return;

      auto bindings = binding_it->second;
      std::sort(bindings.begin(), bindings.end(), [](EpochTaskBinding const& lhs, EpochTaskBinding const& rhs) {
        auto const lhs_epoch = reinterpret_cast<std::uintptr_t>(lhs.epoch_context);
        auto const rhs_epoch = reinterpret_cast<std::uintptr_t>(rhs.epoch_context);
        if (lhs_epoch != rhs_epoch) return lhs_epoch < rhs_epoch;
        return static_cast<int>(lhs.role) < static_cast<int>(rhs.role);
      });

      fmt::print(stderr, "  held by epoch contexts:\n");
      for (auto const& binding : bindings)
      {
        fmt::print(stderr, "    {} ({})\n", binding.epoch_context, to_string(binding.role));
      }
    }

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
    std::unordered_map<void const*, EpochDebugInfo> epoch_contexts_;
    std::unordered_map<void*, std::vector<EpochTaskBinding>> task_bindings_;
};

} // anonymous namespace

namespace uni20
{

void TaskRegistry::register_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().register_task(h); }

void TaskRegistry::destroy_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().destroy_task(h); }

void TaskRegistry::leak_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().leak_task(h); }

void TaskRegistry::mark_running(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_running(h); }

void TaskRegistry::mark_suspended(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_suspended(h); }

void TaskRegistry::register_epoch_context(void const* epoch_context)
{
  TaskRegistryImpl::instance().register_epoch_context(epoch_context);
}

void TaskRegistry::destroy_epoch_context(void const* epoch_context)
{
  TaskRegistryImpl::instance().destroy_epoch_context(epoch_context);
}

void TaskRegistry::bind_epoch_task(void const* epoch_context, std::coroutine_handle<> h, EpochTaskRole role)
{
  TaskRegistryImpl::instance().bind_epoch_task(epoch_context, h, role);
}

void TaskRegistry::unbind_epoch_task(void const* epoch_context, std::coroutine_handle<> h, EpochTaskRole role)
{
  TaskRegistryImpl::instance().unbind_epoch_task(epoch_context, h, role);
}

std::vector<std::coroutine_handle<>> TaskRegistry::epoch_reader_tasks(void const* epoch_context)
{
  return TaskRegistryImpl::instance().epoch_reader_tasks(epoch_context);
}

std::vector<std::coroutine_handle<>> TaskRegistry::epoch_writer_tasks(void const* epoch_context)
{
  return TaskRegistryImpl::instance().epoch_writer_tasks(epoch_context);
}

void TaskRegistry::dump() { TaskRegistryImpl::instance().dump(); }

} // namespace uni20
