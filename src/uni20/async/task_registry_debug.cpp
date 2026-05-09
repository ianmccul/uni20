#include "epoch_context.hpp"
#include "task_registry.hpp"
#include <uni20/config.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <iterator>
#include <limits>
#include <mutex>
#if UNI20_HAS_STACKTRACE
#include <stacktrace>
#endif
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace
{

using TaskState = uni20::TaskRegistry::TaskState;
using EpochTaskRole = uni20::TaskRegistry::EpochTaskRole;
using DumpMode = uni20::TaskRegistry::DumpMode;
using EpochContext = uni20::async::EpochContext;
using NodeInfo = uni20::async::NodeInfo;
using GraphvizDumpOptions = uni20::TaskRegistry::GraphvizDumpOptions;
using DiagnosticsServiceOptions = uni20::TaskRegistry::DiagnosticsServiceOptions;

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
  auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(timestamp.time_since_epoch()) % std::chrono::seconds(1);
  auto const time = std::chrono::system_clock::to_time_t(timestamp);
  auto const local_time = fmt::localtime(time);
  return fmt::format("{:%F %T}.{:06} {:%z}", local_time, us.count(), local_time);
}

std::string dot_escape(std::string_view text)
{
  std::string escaped;
  escaped.reserve(text.size());
  for (char c : text)
  {
    switch (c)
    {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

void append_unique_node(std::vector<NodeInfo const*>& nodes, NodeInfo const* node)
{
  if (!node) return;
  if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) nodes.push_back(node);
}

void append_unique_edge(std::vector<std::string>& edges, std::string edge)
{
  if (std::find(edges.begin(), edges.end(), edge) == edges.end()) edges.push_back(std::move(edge));
}

std::atomic<unsigned> pending_graphviz_dump_requests{0};
std::atomic<unsigned long long> graphviz_dump_sequence{0};

long current_process_id() noexcept
{
#if defined(__unix__) || defined(__APPLE__)
  return static_cast<long>(::getpid());
#else
  return 0;
#endif
}

std::string trim_trailing_slash(std::string path)
{
  while (path.size() > 1 && path.back() == '/')
    path.pop_back();
  return path;
}

std::string default_dump_path(GraphvizDumpOptions const& options)
{
  auto output_dir = trim_trailing_slash(options.output_dir.empty() ? std::string(".") : options.output_dir);
  auto const prefix = options.file_prefix.empty() ? std::string("uni20-dag") : options.file_prefix;
  auto const sequence = graphviz_dump_sequence.fetch_add(1, std::memory_order_relaxed);
  return fmt::format("{}/{}.{}.{}.dot", output_dir, prefix, current_process_id(), sequence);
}

bool write_text_file(std::string const& path, std::string_view contents)
{
  auto* stream = std::fopen(path.c_str(), "w");
  if (!stream) return false;
  auto const write_ok = std::fwrite(contents.data(), 1, contents.size(), stream) == contents.size();
  auto const close_ok = std::fclose(stream) == 0;
  return write_ok && close_ok;
}

bool consume_request_file(std::string const& request_file)
{
  if (request_file.empty()) return false;
  std::error_code ec;
  if (!std::filesystem::exists(request_file, ec) || ec) return false;
  std::filesystem::remove(request_file, ec);
  return true;
}

void ensure_output_dir(std::string const& output_dir)
{
  if (output_dir.empty() || output_dir == ".") return;
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
}

char const* nonempty_env(char const* name) noexcept
{
  auto const* value = std::getenv(name);
  return value && value[0] != '\0' ? value : nullptr;
}

int parse_positive_int(char const* value, int fallback) noexcept
{
  if (!value || value[0] == '\0') return fallback;
  char* end = nullptr;
  errno = 0;
  auto parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed <= 0) return fallback;
  if (parsed > static_cast<long>(std::numeric_limits<int>::max())) return fallback;
  return static_cast<int>(parsed);
}

int parse_signal_number(char const* value) noexcept
{
  if (!value || value[0] == '\0') return 0;
  std::string signal(value);
  std::transform(signal.begin(), signal.end(), signal.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (signal == "0" || signal == "NONE" || signal == "OFF" || signal == "FALSE" || signal == "NO") return 0;
#if defined(SIGUSR1)
  if (signal == "USR1" || signal == "SIGUSR1") return SIGUSR1;
#endif
#if defined(SIGUSR2)
  if (signal == "USR2" || signal == "SIGUSR2") return SIGUSR2;
#endif
  return parse_positive_int(value, 0);
}

GraphvizDumpOptions default_graphviz_dump_options()
{
  GraphvizDumpOptions options;
  if (auto const* output_dir = nonempty_env("UNI20_DEBUG_DAG_OUTPUT_DIR")) options.output_dir = output_dir;
  if (auto const* prefix = nonempty_env("UNI20_DEBUG_DAG_FILE_PREFIX")) options.file_prefix = prefix;
  return options;
}

DiagnosticsServiceOptions default_diagnostics_service_options()
{
  DiagnosticsServiceOptions options;
  options.dump_options = default_graphviz_dump_options();
  if (auto const* request_file = nonempty_env("UNI20_DEBUG_DAG_REQUEST_FILE")) options.request_file = request_file;
  if (auto const* poll_ms = nonempty_env("UNI20_DEBUG_DAG_POLL_MS"))
    options.poll_interval_ms = parse_positive_int(poll_ms, options.poll_interval_ms);
  if (auto const* signal = nonempty_env("UNI20_DEBUG_DAG_SIGNAL")) options.signal_number = parse_signal_number(signal);
  return options;
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

  if (value == "0" || value == "none" || value == "off" || value == "false" || value == "no") return DumpMode::None;
  if (value == "2" || value == "full" || value == "all" || value == "verbose") return DumpMode::Full;
  if (value == "1" || value == "basic" || value == "on" || value == "true" || value == "yes") return DumpMode::Basic;

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

struct TaskNodeDependency
{
    NodeInfo const* node{nullptr};
    EpochTaskRole role{EpochTaskRole::Reader};
};

struct TaskDebugInfo
{
    std::size_t id{0};
    TaskState state{TaskState::Constructed};
    std::size_t transition_count{0};
    std::chrono::system_clock::time_point creation_timestamp{};
    std::chrono::system_clock::time_point last_state_change_timestamp{};
    std::string display_name{};
    std::string waiting_on{};
    std::vector<NodeInfo const*> read_dependencies{};
    std::vector<NodeInfo const*> write_dependencies{};
    std::vector<TaskNodeDependency> await_dependencies{};
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

    void record_task_dependencies(std::coroutine_handle<> h, std::vector<NodeInfo const*> const& read_dependencies,
                                  std::vector<NodeInfo const*> const& write_dependencies)
    {
      if (!h) return;
      std::lock_guard lock(mutex_);
      auto it = tasks_.find(h.address());
      if (it == tasks_.end()) return;
      for (auto* node : read_dependencies)
        append_unique_node(it->second.read_dependencies, node);
      for (auto* node : write_dependencies)
        append_unique_node(it->second.write_dependencies, node);
    }

    void record_await_dependency(std::coroutine_handle<> h, NodeInfo const* node, EpochTaskRole role)
    {
      if (!h || !node) return;
      std::lock_guard lock(mutex_);
      auto it = tasks_.find(h.address());
      if (it == tasks_.end()) return;
      auto& dependencies = it->second.await_dependencies;
      auto const found = std::find_if(dependencies.begin(), dependencies.end(), [&](TaskNodeDependency const& dep) {
        return dep.node == node && dep.role == role;
      });
      if (found == dependencies.end()) dependencies.push_back(TaskNodeDependency{node, role});
    }

    void name_task(std::coroutine_handle<> h, std::string const& label)
    {
      if (!h) return;
      std::lock_guard lock(mutex_);
      auto it = tasks_.find(h.address());
      if (it == tasks_.end()) return;
      it->second.display_name = label;
    }

    void name_async_value(NodeInfo const* node, std::string const& label)
    {
      if (!node) return;
      std::lock_guard lock(mutex_);
      if (label.empty())
        node_names_.erase(node);
      else
        node_names_[node] = label;
    }

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

      std::sort(epochs.begin(), epochs.end(),
                [](EpochDumpRecord const& lhs, EpochDumpRecord const& rhs) { return lhs.info.id < rhs.info.id; });

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
        std::sort(associations.begin(), associations.end(),
                  [](TaskEpochAssociation const& lhs, TaskEpochAssociation const& rhs) {
                    if (lhs.epoch_id != rhs.epoch_id) return lhs.epoch_id < rhs.epoch_id;
                    return static_cast<int>(lhs.role) < static_cast<int>(rhs.role);
                  });
        associations.erase(std::unique(associations.begin(), associations.end(),
                                       [](TaskEpochAssociation const& lhs, TaskEpochAssociation const& rhs) {
                                         return lhs.epoch_id == rhs.epoch_id && lhs.role == rhs.role;
                                       }),
                           associations.end());
      }

      std::vector<std::pair<void*, TaskDebugInfo const*>> sorted_tasks;
      sorted_tasks.reserve(tasks_copy.size());
      for (auto const& [addr, info] : tasks_copy)
        sorted_tasks.emplace_back(addr, &info);

      std::sort(sorted_tasks.begin(), sorted_tasks.end(),
                [](auto const& lhs, auto const& rhs) { return lhs.second->id < rhs.second->id; });

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
              fmt::print(stderr, "  next epoch id: unknown ({})\n",
                         static_cast<void const*>(epoch.snapshot.next_epoch));
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
          if (!info.display_name.empty()) fmt::print(stderr, "  task name: {}\n", info.display_name);
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

    std::string graphviz_dot(bool best_effort)
    {
      struct EpochDotRecord
      {
          EpochContext const* epoch{nullptr};
          EpochDebugInfo info{};
          EpochContext::DebugSnapshot snapshot{};
          bool snapshot_available{true};
      };

      std::unordered_map<void*, TaskDebugInfo> tasks_copy;
      std::unordered_map<NodeInfo const*, std::string> node_names_copy;
      std::vector<std::pair<EpochContext const*, EpochDebugInfo>> epoch_records;
      {
        std::unique_lock<std::mutex> lock;
        if (best_effort)
          lock = std::unique_lock<std::mutex>(mutex_, std::try_to_lock);
        else
          lock = std::unique_lock<std::mutex>(mutex_);
        if (!lock.owns_lock())
        {
          return "digraph uni20_async_dag {\n"
                 "  rankdir=LR;\n"
                 "  registry_locked [shape=note, label=\"TaskRegistry snapshot unavailable: registry lock busy\"];\n"
                 "}\n";
        }
        tasks_copy = tasks_;
        node_names_copy = node_names_;
        epoch_records.reserve(epoch_contexts_.size());
        for (auto const& [epoch, info] : epoch_contexts_)
        {
          if (!epoch) continue;
          epoch_records.emplace_back(epoch, info);
        }
      }

      std::vector<EpochDotRecord> epochs;
      epochs.reserve(epoch_records.size());
      for (auto const& [epoch, info] : epoch_records)
      {
        EpochDotRecord record{epoch, info, {}, true};
        if (best_effort)
          record.snapshot_available = epoch->try_debug_snapshot(record.snapshot);
        else
          record.snapshot = epoch->debug_snapshot();
        epochs.push_back(std::move(record));
      }

      std::sort(epochs.begin(), epochs.end(),
                [](EpochDotRecord const& lhs, EpochDotRecord const& rhs) { return lhs.info.id < rhs.info.id; });

      std::unordered_map<EpochContext const*, std::size_t> epoch_id_by_ptr;
      epoch_id_by_ptr.reserve(epochs.size());
      for (auto const& epoch : epochs)
        epoch_id_by_ptr.emplace(epoch.epoch, epoch.info.id);

      std::vector<std::pair<void*, TaskDebugInfo const*>> sorted_tasks;
      sorted_tasks.reserve(tasks_copy.size());
      for (auto const& [addr, info] : tasks_copy)
        sorted_tasks.emplace_back(addr, &info);
      std::sort(sorted_tasks.begin(), sorted_tasks.end(),
                [](auto const& lhs, auto const& rhs) { return lhs.second->id < rhs.second->id; });

      std::vector<NodeInfo const*> data_nodes;
      for (auto const& [addr, info_ptr] : sorted_tasks)
      {
        (void)addr;
        for (auto* node : info_ptr->read_dependencies)
          append_unique_node(data_nodes, node);
        for (auto* node : info_ptr->write_dependencies)
          append_unique_node(data_nodes, node);
        for (auto const& dependency : info_ptr->await_dependencies)
          append_unique_node(data_nodes, dependency.node);
      }
#if UNI20_DEBUG_DAG
      for (auto const& epoch : epochs)
        append_unique_node(data_nodes, epoch.snapshot.node);
#endif
      std::sort(data_nodes.begin(), data_nodes.end(),
                [](NodeInfo const* lhs, NodeInfo const* rhs) { return lhs->global_index() < rhs->global_index(); });

      std::vector<std::string> edges;
      auto add_edge = [&](std::string edge) { append_unique_edge(edges, std::move(edge)); };

      std::string dot;
      fmt::format_to(std::back_inserter(dot), "digraph uni20_async_dag {{\n");
      fmt::format_to(std::back_inserter(dot), "  rankdir=LR;\n");
      fmt::format_to(std::back_inserter(dot), "  graph [fontname=\"monospace\"];\n");
      fmt::format_to(std::back_inserter(dot), "  node [fontname=\"monospace\"];\n");
      fmt::format_to(std::back_inserter(dot), "  edge [fontname=\"monospace\"];\n\n");

      for (auto* node : data_nodes)
      {
        auto const name_it = node_names_copy.find(node);
        auto const label =
            name_it == node_names_copy.end()
                ? fmt::format("data {}\n{}\naddr={}", node->global_index(), node->type(), node->address())
                : fmt::format("{}\ndata {}\n{}\naddr={}", name_it->second, node->global_index(), node->type(),
                              node->address());
        fmt::format_to(std::back_inserter(dot),
                       "  data_{} [shape=cylinder, style=filled, fillcolor=\"#eef7ff\", label=\"{}\"];\n",
                       node->global_index(), dot_escape(label));
      }

      if (!data_nodes.empty()) dot += "\n";

      for (auto const& epoch : epochs)
      {
        auto const label = epoch.snapshot_available
                               ? fmt::format("epoch {}\ngen={}\nphase={}", epoch.info.id, epoch.snapshot.generation,
                                             to_string(epoch.snapshot.phase))
                               : fmt::format("epoch {}\nsnapshot unavailable\nlock busy", epoch.info.id);
        fmt::format_to(std::back_inserter(dot),
                       "  epoch_{} [shape=oval, style=filled, fillcolor=\"#f7f7f7\", label=\"{}\"];\n", epoch.info.id,
                       dot_escape(label));
      }

      if (!epochs.empty()) dot += "\n";

      for (auto const& [addr, info_ptr] : sorted_tasks)
      {
        auto const& info = *info_ptr;
        auto const label =
            info.display_name.empty()
                ? fmt::format("task {}\n{}\ntransitions={}\nptr={}", info.id, to_string(info.state),
                              info.transition_count, addr)
                : fmt::format("{}\ntask {}\n{}\ntransitions={}\nptr={}", info.display_name, info.id,
                              to_string(info.state), info.transition_count, addr);
        fmt::format_to(std::back_inserter(dot),
                       "  task_{} [shape=box, style=\"rounded,filled\", fillcolor=\"#fff7e8\", label=\"{}\"];\n",
                       info.id, dot_escape(label));
      }

      if (!sorted_tasks.empty()) dot += "\n";

      for (auto const& [addr, info_ptr] : sorted_tasks)
      {
        (void)addr;
        auto const& info = *info_ptr;
        for (auto* node : info.read_dependencies)
        {
          if (!node) continue;
          add_edge(fmt::format("  data_{} -> task_{} [label=\"arg read\", style=dashed, color=\"#4c78a8\"];\n",
                               node->global_index(), info.id));
        }
        for (auto* node : info.write_dependencies)
        {
          if (!node) continue;
          add_edge(fmt::format("  task_{} -> data_{} [label=\"arg write\", style=dashed, color=\"#f58518\"];\n",
                               info.id, node->global_index()));
        }
        for (auto const& dependency : info.await_dependencies)
        {
          if (!dependency.node) continue;
          if (dependency.role == EpochTaskRole::Reader)
          {
            add_edge(fmt::format("  data_{} -> task_{} [label=\"co_await read\", color=\"#4c78a8\"];\n",
                                 dependency.node->global_index(), info.id));
          }
          else
          {
            add_edge(fmt::format("  task_{} -> data_{} [label=\"co_await write\", color=\"#f58518\"];\n", info.id,
                                 dependency.node->global_index()));
          }
        }
      }

      for (auto const& epoch : epochs)
      {
        if (!epoch.snapshot_available) continue;
#if UNI20_DEBUG_DAG
        if (epoch.snapshot.node)
        {
          add_edge(fmt::format("  data_{} -> epoch_{} [label=\"epochs\", style=dotted, color=\"#888888\"];\n",
                               epoch.snapshot.node->global_index(), epoch.info.id));
        }
#endif
        if (epoch.snapshot.next_epoch)
        {
          auto const next_it = epoch_id_by_ptr.find(epoch.snapshot.next_epoch);
          if (next_it != epoch_id_by_ptr.end())
          {
            add_edge(fmt::format("  epoch_{} -> epoch_{} [label=\"next\", color=\"#888888\"];\n", epoch.info.id,
                                 next_it->second));
          }
        }
        for (auto const& reader : epoch.snapshot.reader_tasks)
        {
          if (!reader) continue;
          auto const task_it = tasks_copy.find(reader.address());
          if (task_it == tasks_copy.end()) continue;
          add_edge(fmt::format("  epoch_{} -> task_{} [label=\"await read\", color=\"#4c78a8\"];\n", epoch.info.id,
                               task_it->second.id));
        }
        for (auto const& writer : epoch.snapshot.writer_tasks)
        {
          if (!writer) continue;
          auto const task_it = tasks_copy.find(writer.address());
          if (task_it == tasks_copy.end()) continue;
          add_edge(fmt::format("  task_{} -> epoch_{} [label=\"await write\", color=\"#f58518\"];\n",
                               task_it->second.id, epoch.info.id));
        }
      }

      for (auto const& edge : edges)
        dot += edge;

      dot += "}\n";
      return dot;
    }

    std::string graphviz_dot() { return this->graphviz_dot(false); }

    std::string graphviz_dot_best_effort() { return this->graphviz_dot(true); }

    void dump_graphviz(std::FILE* stream)
    {
      if (!stream) return;
      auto dot = this->graphviz_dot();
      std::fputs(dot.c_str(), stream);
    }

    bool dump_graphviz_file(std::string const& path) { return write_text_file(path, this->graphviz_dot()); }

    bool dump_graphviz_file_best_effort(std::string const& path)
    {
      return write_text_file(path, this->graphviz_dot_best_effort());
    }

    bool service_debug_requests(GraphvizDumpOptions const& options)
    {
      auto requests = pending_graphviz_dump_requests.exchange(0, std::memory_order_acq_rel);
      ensure_output_dir(options.output_dir);
      bool serviced = false;
      for (unsigned i = 0; i < requests; ++i)
      {
        serviced = this->dump_graphviz_file_best_effort(default_dump_path(options)) || serviced;
      }
      return serviced;
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
    std::unordered_map<NodeInfo const*, std::string> node_names_;
    std::unordered_map<EpochContext const*, EpochDebugInfo> epoch_contexts_;
    std::size_t next_task_id_{1};
    std::size_t next_epoch_id_{1};
};

class DiagnosticServiceState {
  public:
    bool start(DiagnosticsServiceOptions options)
    {
      std::lock_guard lock(mutex_);
      if (thread_.joinable()) return true;

#if defined(__linux__)
      if (options.signal_number > 0 && options.block_signal_in_calling_thread)
      {
        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, options.signal_number);
        if (pthread_sigmask(SIG_BLOCK, &set, nullptr) != 0) return false;
      }
#else
      if (options.signal_number > 0) return false;
#endif

      stop_.store(false, std::memory_order_release);
      signal_number_ = options.signal_number;
      thread_ = std::thread([this, options = std::move(options)] { this->run(options); });
      return true;
    }

    void stop() noexcept
    {
      std::thread thread;
      int signal_number = 0;
      {
        std::lock_guard lock(mutex_);
        stop_.store(true, std::memory_order_release);
        signal_number = signal_number_;
        thread = std::move(thread_);
      }

#if defined(__linux__)
      if (thread.joinable() && signal_number > 0) pthread_kill(thread.native_handle(), signal_number);
#endif
      if (thread.joinable()) thread.join();
    }

    ~DiagnosticServiceState() { this->stop(); }

  private:
    static bool wait_for_signal(int signal_number, int poll_interval_ms) noexcept
    {
#if defined(__linux__)
      if (signal_number <= 0)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        return false;
      }

      sigset_t set;
      sigemptyset(&set);
      sigaddset(&set, signal_number);
      pthread_sigmask(SIG_BLOCK, &set, nullptr);

      timespec timeout{};
      timeout.tv_sec = poll_interval_ms / 1000;
      timeout.tv_nsec = static_cast<long>(poll_interval_ms % 1000) * 1000000L;

      errno = 0;
      auto const received = sigtimedwait(&set, nullptr, &timeout);
      return received == signal_number;
#else
      static_cast<void>(signal_number);
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
      return false;
#endif
    }

    void run(DiagnosticsServiceOptions options)
    {
      auto const poll_interval_ms = options.poll_interval_ms > 0 ? options.poll_interval_ms : 250;
      ensure_output_dir(options.dump_options.output_dir);

      while (!stop_.load(std::memory_order_acquire))
      {
        bool triggered = wait_for_signal(options.signal_number, poll_interval_ms);
        if (stop_.load(std::memory_order_acquire)) break;
        if (consume_request_file(options.request_file)) triggered = true;
        if (triggered) pending_graphviz_dump_requests.fetch_add(1, std::memory_order_release);
        TaskRegistryImpl::instance().service_debug_requests(options.dump_options);
      }
    }

    std::mutex mutex_;
    std::thread thread_{};
    std::atomic<bool> stop_{false};
    int signal_number_{0};
};

DiagnosticServiceState& diagnostic_service()
{
  static DiagnosticServiceState* service = new DiagnosticServiceState(); // intentional leak
  return *service;
}

} // anonymous namespace

namespace uni20
{

void TaskRegistry::register_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().register_task(h); }

void TaskRegistry::destroy_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().destroy_task(h); }

void TaskRegistry::leak_task(std::coroutine_handle<> h) { TaskRegistryImpl::instance().leak_task(h); }

void TaskRegistry::mark_running(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_running(h); }

void TaskRegistry::mark_suspended(std::coroutine_handle<> h) { TaskRegistryImpl::instance().mark_suspended(h); }

void TaskRegistry::record_task_dependencies(std::coroutine_handle<> h,
                                            std::vector<async::NodeInfo const*> const& read_dependencies,
                                            std::vector<async::NodeInfo const*> const& write_dependencies)
{
  TaskRegistryImpl::instance().record_task_dependencies(h, read_dependencies, write_dependencies);
}

void TaskRegistry::record_await_dependency(std::coroutine_handle<> h, async::NodeInfo const* node, EpochTaskRole role)
{
  TaskRegistryImpl::instance().record_await_dependency(h, node, role);
}

void TaskRegistry::name_task(std::coroutine_handle<> h, std::string const& label)
{
  TaskRegistryImpl::instance().name_task(h, label);
}

void TaskRegistry::name_async_value(async::NodeInfo const* node, std::string const& label)
{
  TaskRegistryImpl::instance().name_async_value(node, label);
}

void TaskRegistry::register_epoch_context(async::EpochContext const* epoch_context)
{
  TaskRegistryImpl::instance().register_epoch_context(epoch_context);
}

void TaskRegistry::destroy_epoch_context(async::EpochContext const* epoch_context)
{
  TaskRegistryImpl::instance().destroy_epoch_context(epoch_context);
}

void TaskRegistry::bind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                   EpochTaskRole role)
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

std::string TaskRegistry::graphviz_dot() { return TaskRegistryImpl::instance().graphviz_dot(); }

std::string TaskRegistry::graphviz_dot_best_effort() { return TaskRegistryImpl::instance().graphviz_dot_best_effort(); }

void TaskRegistry::dump_graphviz(std::FILE* stream) { TaskRegistryImpl::instance().dump_graphviz(stream); }

bool TaskRegistry::dump_graphviz_file(std::string const& path)
{
  return TaskRegistryImpl::instance().dump_graphviz_file(path);
}

bool TaskRegistry::dump_graphviz_file_best_effort(std::string const& path)
{
  return TaskRegistryImpl::instance().dump_graphviz_file_best_effort(path);
}

TaskRegistry::GraphvizDumpOptions TaskRegistry::default_graphviz_dump_options()
{
  return ::default_graphviz_dump_options();
}

TaskRegistry::DiagnosticsServiceOptions TaskRegistry::default_diagnostics_service_options()
{
  return ::default_diagnostics_service_options();
}

std::string TaskRegistry::default_graphviz_dump_path() { return default_dump_path(default_graphviz_dump_options()); }

std::string TaskRegistry::default_graphviz_dump_path(GraphvizDumpOptions const& options)
{
  return default_dump_path(options);
}

void TaskRegistry::request_graphviz_dump() noexcept
{
  pending_graphviz_dump_requests.fetch_add(1, std::memory_order_release);
}

bool TaskRegistry::service_debug_requests()
{
  return TaskRegistry::service_debug_requests(default_graphviz_dump_options());
}

bool TaskRegistry::service_debug_requests(GraphvizDumpOptions const& options)
{
  return TaskRegistryImpl::instance().service_debug_requests(options);
}

bool TaskRegistry::start_diagnostics_service()
{
  return TaskRegistry::start_diagnostics_service(default_diagnostics_service_options());
}

bool TaskRegistry::start_diagnostics_service(DiagnosticsServiceOptions const& options)
{
  return diagnostic_service().start(options);
}

void TaskRegistry::stop_diagnostics_service() noexcept { diagnostic_service().stop(); }

void TaskRegistry::dump() { TaskRegistryImpl::instance().dump(); }

} // namespace uni20
