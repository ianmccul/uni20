#include "config.hpp"
#include "epoch_context.hpp"
#include "task_registry.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <mutex>
#if UNI20_HAS_STACKTRACE
#include <stacktrace>
#endif
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace
{

using TaskState = uni20::TaskRegistry::TaskState;
using EpochTaskRole = uni20::TaskRegistry::EpochTaskRole;
using DumpMode = uni20::TaskRegistry::DumpMode;
using EpochContext = uni20::async::EpochContext;

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

std::string_view to_string(EpochContext::Phase phase) noexcept { return uni20::async::format_as(phase); }

std::string format_timestamp(std::chrono::system_clock::time_point timestamp)
{
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(timestamp.time_since_epoch()) % std::chrono::seconds(1);
  auto const time = std::chrono::system_clock::to_time_t(timestamp);
  auto const local_time = fmt::localtime(time);
  return fmt::format("{:%F %T}.{:06} {:%z}", local_time, us.count(), local_time);
}

DumpMode parse_dump_mode(char const* raw_value) noexcept
{
  if (!raw_value || *raw_value == '\0') return DumpMode::Basic;

  std::string value(raw_value);
  auto const is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if_not(value.begin(), value.end(), is_space));
  value.erase(std::find_if_not(value.rbegin(), value.rend(), is_space).base(), value.end());
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (value == "0" || value == "none" || value == "off" || value == "false" || value == "no")
    return DumpMode::None;
  if (value == "2" || value == "full" || value == "all" || value == "verbose")
    return DumpMode::Full;
  if (value == "1" || value == "basic" || value == "on" || value == "true" || value == "yes")
    return DumpMode::Basic;

  return DumpMode::Basic;
}

DumpMode runtime_dump_mode() noexcept
{
  static DumpMode const mode = parse_dump_mode(std::getenv("UNI20_DEBUG_ASYNC_TASKS"));
  return mode;
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
    std::size_t id{0};
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

struct EpochDebugInfo
{
    std::size_t id{0};
    std::chrono::system_clock::time_point creation_timestamp{};
#if UNI20_HAS_STACKTRACE
    std::stacktrace creation_trace{};
#endif
};

struct TaskEpochAssociation
{
    std::size_t epoch_id{0};
    EpochTaskRole role{EpochTaskRole::Reader};
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
        it->second.id = next_task_id_++;
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

    void register_epoch_context(EpochContext const* epoch_context)
    {
      if (!epoch_context) return;
      std::lock_guard lock(mutex_);
      auto [it, inserted] = epoch_contexts_.try_emplace(epoch_context);
      if (inserted)
      {
        it->second.id = next_epoch_id_++;
        it->second.creation_timestamp = std::chrono::system_clock::now();
#if UNI20_HAS_STACKTRACE
        it->second.creation_trace = std::stacktrace::current(2);
#endif
      }
    }

    void destroy_epoch_context(EpochContext const* epoch_context)
    {
      if (!epoch_context) return;
      std::lock_guard lock(mutex_);
      epoch_contexts_.erase(epoch_context);
    }

    void bind_epoch_task(EpochContext const*, std::coroutine_handle<>, EpochTaskRole) {}

    void unbind_epoch_task(EpochContext const*, std::coroutine_handle<>, EpochTaskRole) {}

    std::vector<std::coroutine_handle<>> epoch_reader_tasks(EpochContext const* epoch_context) const
    {
      if (!epoch_context) return {};
      return epoch_context->reader_task_handles();
    }

    std::vector<std::coroutine_handle<>> epoch_writer_tasks(EpochContext const* epoch_context) const
    {
      if (!epoch_context) return {};
      return epoch_context->writer_task_handles();
    }

    void dump()
    {
      struct EpochDumpRecord
      {
          EpochContext const* epoch{nullptr};
          EpochDebugInfo info{};
          EpochContext::DebugSnapshot snapshot{};
      };

      std::unordered_map<void*, TaskDebugInfo> tasks_copy;
      std::vector<EpochDumpRecord> epochs;
      {
        std::lock_guard lock(mutex_);
        tasks_copy = tasks_;
        epochs.reserve(epoch_contexts_.size());
        for (auto const& [epoch, info] : epoch_contexts_)
        {
          if (!epoch) continue;
          epochs.push_back(EpochDumpRecord{epoch, info, epoch->debug_snapshot()});
        }
      }

      std::sort(epochs.begin(), epochs.end(), [](EpochDumpRecord const& lhs, EpochDumpRecord const& rhs) {
        return lhs.info.id < rhs.info.id;
      });

      std::unordered_map<EpochContext const*, std::size_t> epoch_id_by_ptr;
      epoch_id_by_ptr.reserve(epochs.size());
      for (auto const& epoch : epochs)
        epoch_id_by_ptr.emplace(epoch.epoch, epoch.info.id);

      std::unordered_map<void*, std::vector<TaskEpochAssociation>> task_associations;
      auto add_association = [&](std::coroutine_handle<> h, std::size_t epoch_id, EpochTaskRole role) {
        if (!h) return;
        task_associations[h.address()].push_back(TaskEpochAssociation{epoch_id, role});
      };

      for (auto const& epoch : epochs)
      {
        for (auto const& reader : epoch.snapshot.reader_tasks)
          add_association(reader, epoch.info.id, EpochTaskRole::Reader);
        for (auto const& writer : epoch.snapshot.writer_tasks)
          add_association(writer, epoch.info.id, EpochTaskRole::Writer);
      }

      for (auto& [task_addr, associations] : task_associations)
      {
        (void)task_addr;
        std::sort(associations.begin(), associations.end(), [](TaskEpochAssociation const& lhs, TaskEpochAssociation const& rhs) {
          if (lhs.epoch_id != rhs.epoch_id) return lhs.epoch_id < rhs.epoch_id;
          return static_cast<int>(lhs.role) < static_cast<int>(rhs.role);
        });
        associations.erase(std::unique(associations.begin(), associations.end(), [](TaskEpochAssociation const& lhs,
                                                                                     TaskEpochAssociation const& rhs) {
                             return lhs.epoch_id == rhs.epoch_id && lhs.role == rhs.role;
                           }),
                           associations.end());
      }

      std::vector<std::pair<void*, TaskDebugInfo const*>> sorted_tasks;
      sorted_tasks.reserve(tasks_copy.size());
      for (auto const& [addr, info] : tasks_copy)
        sorted_tasks.emplace_back(addr, &info);

      std::sort(sorted_tasks.begin(), sorted_tasks.end(), [](auto const& lhs, auto const& rhs) {
        return lhs.second->id < rhs.second->id;
      });

      fmt::print(stderr, "\n========== Async Task Registry Dump ==========\n");
      fmt::print(stderr, "Total tracked epoch contexts: {}\n", epochs.size());
      fmt::print(stderr, "Total tracked tasks: {}\n\n", sorted_tasks.size());
#if !UNI20_HAS_STACKTRACE
      fmt::print(stderr,
                 "WARNING: std::stacktrace is unavailable; dump output is degraded to state-only information.\n\n");
#endif

      fmt::print(stderr, "EpochContext objects:\n");
      if (epochs.empty())
      {
        fmt::print(stderr, "  (none)\n\n");
      }
      else
      {
        std::size_t epoch_number = 1;
        for (auto const& epoch : epochs)
        {
          fmt::print(stderr, "EpochContext {}:\n", epoch_number);
          fmt::print(stderr, "  epoch id: {}\n", epoch.info.id);
          fmt::print(stderr, "  epoch pointer: {}\n", static_cast<void const*>(epoch.epoch));
          fmt::print(stderr, "  creation timestamp: {}\n", format_timestamp(epoch.info.creation_timestamp));
          fmt::print(stderr, "  generation: {}\n", epoch.snapshot.generation);
          fmt::print(stderr, "  phase: {}\n", to_string(epoch.snapshot.phase));
          if (epoch.snapshot.next_epoch)
          {
            auto next_it = epoch_id_by_ptr.find(epoch.snapshot.next_epoch);
            if (next_it != epoch_id_by_ptr.end())
              fmt::print(stderr, "  next epoch id: {}\n", next_it->second);
            else
              fmt::print(stderr, "  next epoch id: unknown ({})\n", static_cast<void const*>(epoch.snapshot.next_epoch));
          }
          else
          {
            fmt::print(stderr, "  next epoch id: none\n");
          }
#if UNI20_HAS_STACKTRACE
          fmt::print(stderr, "  creation stacktrace:\n");
          print_stacktrace(epoch.info.creation_trace);
#else
          fmt::print(stderr, "  creation stacktrace: unavailable\n");
#endif
          fmt::print(stderr, "\n");
          ++epoch_number;
        }
      }

      fmt::print(stderr, "Coroutine tasks:\n");
      if (sorted_tasks.empty())
      {
        fmt::print(stderr, "  (none)\n");
      }
      else
      {
        std::size_t task_number = 1;
        for (auto const& [addr, info_ptr] : sorted_tasks)
        {
          auto const& info = *info_ptr;
          fmt::print(stderr, "Task {}:\n", task_number);
          fmt::print(stderr, "  task id: {}\n", info.id);
          fmt::print(stderr, "  task pointer: {}\n", addr);
          fmt::print(stderr, "  transition count: {}\n", info.transition_count);
          fmt::print(stderr, "  current state: {}\n", to_string(info.state));
          fmt::print(stderr, "  creation timestamp: {}\n", format_timestamp(info.creation_timestamp));

          if (!info.waiting_on.empty()) fmt::print(stderr, "  waiting on: {}\n", info.waiting_on);

          auto association_it = task_associations.find(addr);
          if (association_it == task_associations.end() || association_it->second.empty())
          {
            fmt::print(stderr, "  associated epoch contexts: none\n");
          }
          else
          {
            fmt::print(stderr, "  associated epoch contexts:\n");
            for (auto const& association : association_it->second)
              fmt::print(stderr, "    {} ({})\n", association.epoch_id, to_string(association.role));
          }

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
      }

      fmt::print(stderr, "================================================\n");
    }

    DumpMode dump_mode() const noexcept { return runtime_dump_mode(); }

    void dump_epoch_context(EpochContext const* epoch_context, char const* reason)
    {
      fmt::print(stderr, "\n========== Async Task Registry Diagnostic ==========\n");
      if (reason && reason[0] != '\0') fmt::print(stderr, "reason: {}\n", reason);

      if (!epoch_context)
      {
        fmt::print(stderr, "epoch: null\n");
        fmt::print(stderr, "====================================================\n");
        return;
      }

      EpochContext::DebugSnapshot snapshot = epoch_context->debug_snapshot();
      TaskDebugInfoMap tasks_copy;
      bool found_epoch = false;
      EpochDebugInfo epoch_info{};
      {
        std::lock_guard lock(mutex_);
        tasks_copy = tasks_;
        auto const epoch_it = epoch_contexts_.find(epoch_context);
        if (epoch_it != epoch_contexts_.end())
        {
          found_epoch = true;
          epoch_info = epoch_it->second;
        }
      }

      fmt::print(stderr, "epoch pointer: {}\n", static_cast<void const*>(epoch_context));
      if (found_epoch)
      {
        fmt::print(stderr, "epoch id: {}\n", epoch_info.id);
        fmt::print(stderr, "epoch creation timestamp: {}\n", format_timestamp(epoch_info.creation_timestamp));
      }
      else
      {
        fmt::print(stderr, "epoch id: unknown\n");
        fmt::print(stderr, "epoch creation timestamp: unknown\n");
      }

      fmt::print(stderr, "epoch generation: {}\n", snapshot.generation);
      fmt::print(stderr, "epoch phase: {}\n", to_string(snapshot.phase));
      fmt::print(stderr, "next epoch pointer: {}\n", static_cast<void const*>(snapshot.next_epoch));

      auto print_task_list = [&](char const* label, std::vector<std::coroutine_handle<>> const& handles) {
        fmt::print(stderr, "{}: {}\n", label, handles.size());
        for (auto const& handle : handles)
        {
          if (!handle)
          {
            fmt::print(stderr, "  - null handle\n");
            continue;
          }

          auto const it = tasks_copy.find(handle.address());
          if (it == tasks_copy.end())
          {
            fmt::print(stderr, "  - {} (untracked)\n", static_cast<void const*>(handle.address()));
            continue;
          }

          fmt::print(stderr, "  - id={} ptr={} state={}\n", it->second.id, static_cast<void const*>(handle.address()),
                     to_string(it->second.state));
        }
      };

      print_task_list("reader tasks", snapshot.reader_tasks);
      print_task_list("writer tasks", snapshot.writer_tasks);

      if (found_epoch)
      {
#if UNI20_HAS_STACKTRACE
        fmt::print(stderr, "epoch creation stacktrace:\n");
        print_stacktrace(epoch_info.creation_trace);
#else
        fmt::print(stderr, "epoch creation stacktrace: unavailable\n");
#endif
      }
      else
      {
        fmt::print(stderr, "epoch creation stacktrace: unknown\n");
      }

      fmt::print(stderr, "====================================================\n");
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

    using TaskDebugInfoMap = std::unordered_map<void*, TaskDebugInfo>;

    std::mutex mutex_;
    TaskDebugInfoMap tasks_;
    std::unordered_map<EpochContext const*, EpochDebugInfo> epoch_contexts_;
    std::size_t next_task_id_{1};
    std::size_t next_epoch_id_{1};
};

} // anonymous namespace

namespace uni20
{

void TaskRegistry::register_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().register_task(h); }

void TaskRegistry::destroy_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().destroy_task(h); }

void TaskRegistry::leak_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().leak_task(h); }

void TaskRegistry::mark_running(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_running(h); }

void TaskRegistry::mark_suspended(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_suspended(h); }

void TaskRegistry::register_epoch_context(async::EpochContext const* epoch_context)
{
  TaskRegistryImpl::instance().register_epoch_context(epoch_context);
}

void TaskRegistry::destroy_epoch_context(async::EpochContext const* epoch_context)
{
  TaskRegistryImpl::instance().destroy_epoch_context(epoch_context);
}

void TaskRegistry::bind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h, EpochTaskRole role)
{
  TaskRegistryImpl::instance().bind_epoch_task(epoch_context, h, role);
}

void TaskRegistry::unbind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                     EpochTaskRole role)
{
  TaskRegistryImpl::instance().unbind_epoch_task(epoch_context, h, role);
}

std::vector<std::coroutine_handle<>> TaskRegistry::epoch_reader_tasks(async::EpochContext const* epoch_context)
{
  return TaskRegistryImpl::instance().epoch_reader_tasks(epoch_context);
}

std::vector<std::coroutine_handle<>> TaskRegistry::epoch_writer_tasks(async::EpochContext const* epoch_context)
{
  return TaskRegistryImpl::instance().epoch_writer_tasks(epoch_context);
}

TaskRegistry::DumpMode TaskRegistry::dump_mode() noexcept { return TaskRegistryImpl::instance().dump_mode(); }

void TaskRegistry::dump_epoch_context(async::EpochContext const* epoch_context, char const* reason)
{
  TaskRegistryImpl::instance().dump_epoch_context(epoch_context, reason);
}

void TaskRegistry::dump() { TaskRegistryImpl::instance().dump(); }

} // namespace uni20
