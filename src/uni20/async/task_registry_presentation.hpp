#pragma once

/**
 * \file task_registry_presentation.hpp
 * \ingroup async
 * \brief Presentation-layer reports for async task-registry snapshots.
 */

#include "task_registry_snapshot.hpp"

#include <uni20/common/presentation.hpp>
#include <uni20/config.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace uni20
{

/// \brief Options controlling an async task-registry summary report.
struct TaskRegistryReportOptions
{
    std::string title{"Async task registry"};
    std::string reason{};
};

namespace detail
{

template <typename Value> [[nodiscard]] std::string task_registry_join_ids(std::vector<Value> const& values)
{
  if (values.empty()) return "-";

  std::string result;
  for (auto const value : values)
  {
    if (!result.empty()) result += ", ";
    result += fmt::format("{}", value);
  }
  return result;
}

[[nodiscard]] inline std::string task_registry_awaits(std::vector<TaskRegistryGraphAwaitDependency> const& dependencies)
{
  if (dependencies.empty()) return "-";

  std::string result;
  for (auto const& dependency : dependencies)
  {
    if (!result.empty()) result += ", ";
    result +=
        fmt::format("{}{}", dependency.role == TaskRegistryGraphRole::Reader ? "read " : "write ", dependency.node_id);
  }
  return result;
}

[[nodiscard]] inline std::string task_registry_task_source(TaskRegistryGraphTask const& task)
{
  if (task.state == "suspended" && !task.last_await_site.location.empty())
    return "await: " + task.last_await_site.location;
  if (!task.schedule_site.location.empty()) return "scheduled: " + task.schedule_site.location;
  if (!task.creation_site.location.empty()) return "created: " + task.creation_site.location;
  if (!task.creation_timestamp.empty()) return task.creation_timestamp;
  return "-";
}

[[nodiscard]] inline std::string task_registry_active_awaits(TaskRegistryGraphTask const& task)
{
  auto awaits = task_registry_awaits(task.await_dependencies);
  if (task.waiting_on.empty()) return awaits;
  return awaits == "-" ? task.waiting_on : fmt::format("{}; {}", awaits, task.waiting_on);
}

[[nodiscard]] inline std::string task_registry_data_state(TaskRegistryGraphDataNode const& node,
                                                          TaskRegistryGraphDiagnostics const& diagnostics)
{
  std::string state = node.state;
  if (std::ranges::find(diagnostics.missing_writer_node_ids, node.id) != diagnostics.missing_writer_node_ids.end())
    state += "; missing writer";
  if (std::ranges::find(diagnostics.cycle_node_ids, node.id) != diagnostics.cycle_node_ids.end())
    state += "; dependency cycle";
  return state;
}

[[nodiscard]] inline std::string task_registry_task_state(TaskRegistryGraphTask const& task,
                                                          TaskRegistryGraphDiagnostics const& diagnostics)
{
  std::string state = task.state;
  if (std::ranges::find(diagnostics.blocked_read_task_ids, task.id) != diagnostics.blocked_read_task_ids.end())
    state += "; blocked read";
  if (std::ranges::find(diagnostics.blocked_write_task_ids, task.id) != diagnostics.blocked_write_task_ids.end())
    state += "; blocked write";
  if (std::ranges::find(diagnostics.cycle_task_ids, task.id) != diagnostics.cycle_task_ids.end())
    state += "; dependency cycle";
  return state;
}

} // namespace detail

/// \brief Build a presentation-native summary of one async task-registry snapshot.
/// \details The report contains the same structured tasks, epochs, values, and
///          diagnostic findings used by the Graphviz renderer. Callers choose
///          terminal, plain-text, strict-ASCII, or another presentation policy
///          when rendering the returned document.
/// \param snapshot Structured async runtime snapshot.
/// \param diagnostics Findings derived from `snapshot`.
/// \param options Report title and optional trigger reason.
/// \return Presentation report suitable for any presentation renderer.
[[nodiscard]] inline presentation::report_builder task_registry_report(TaskRegistryGraphSnapshot const& snapshot,
                                                                       TaskRegistryGraphDiagnostics const& diagnostics,
                                                                       TaskRegistryReportOptions const& options = {})
{
  presentation::report_builder report(options.title);
  if (!options.reason.empty()) report.status(presentation::semantic_glyph::info, options.reason);

  if (!snapshot.snapshot_available)
  {
    report.status(presentation::semantic_glyph::failure,
                  snapshot.unavailable_reason.empty() ? "snapshot unavailable" : snapshot.unavailable_reason);
    return report;
  }

  if (!diagnostics.failed_task_ids.empty())
  {
    report.status(presentation::semantic_glyph::failure, "failed coroutine task captured");
  }
  else if (!diagnostics.cycle_task_ids.empty() || !diagnostics.cycle_node_ids.empty())
  {
    report.status(presentation::semantic_glyph::failure, "dependency cycle detected");
  }
  else if (!diagnostics.missing_writer_node_ids.empty() || !diagnostics.missing_writer_epoch_ids.empty())
  {
    report.status(presentation::semantic_glyph::failure, "missing writer detected");
  }
  else if (!diagnostics.blocked_read_task_ids.empty() || !diagnostics.blocked_write_task_ids.empty())
  {
    report.status(presentation::semantic_glyph::warning, "blocked async work captured");
  }
  else
  {
    report.status(presentation::semantic_glyph::info, "snapshot captured");
  }

#if !UNI20_HAS_STACKTRACE
  report.status(presentation::semantic_glyph::warning, "std::stacktrace is unavailable; source provenance is degraded");
#endif

  report.field("tracked async values", snapshot.data_nodes.size())
      .field("tracked epoch contexts", snapshot.epochs.size())
      .field("tracked coroutine tasks", snapshot.tasks.size())
      .field("failed coroutine tasks", diagnostics.failed_task_ids.size())
      .field("blocked readers", diagnostics.blocked_read_task_ids.size())
      .field("blocked writers", diagnostics.blocked_write_task_ids.size());

  if (!diagnostics.notes.empty())
  {
    auto& table = report.table("Diagnostics");
    table.column("finding", presentation::table_alignment::left);
    for (auto const& note : diagnostics.notes)
      table.row(note);
  }

  if (!snapshot.data_nodes.empty())
  {
    auto& table = report.table("Async values");
    table.column("value", presentation::table_alignment::left)
        .column("type", presentation::table_alignment::left)
        .column("state", presentation::table_alignment::left)
        .column("content", presentation::table_alignment::left)
        .column("storage", presentation::table_alignment::left);

    for (auto const& node : snapshot.data_nodes)
    {
      auto name =
          node.label.empty() ? fmt::format("data {}", node.id) : fmt::format("data {}: {}", node.id, node.label);
      table.row(name, node.type.empty() ? "-" : node.type, detail::task_registry_data_state(node, diagnostics),
                node.value.empty() ? "-" : node.value, node.storage_address.empty() ? "-" : node.storage_address);
    }
  }

  if (!snapshot.epochs.empty())
  {
    auto& table = report.table("Epoch contexts");
    table.column("epoch", presentation::table_alignment::left)
        .column("phase", presentation::table_alignment::left)
        .column("value", presentation::table_alignment::left)
        .column("readers", presentation::table_alignment::left)
        .column("writers", presentation::table_alignment::left)
        .column("next", presentation::table_alignment::left)
        .column("created", presentation::table_alignment::left);

    for (auto const& epoch : snapshot.epochs)
    {
      auto const epoch_label = fmt::format("{} (gen {})", epoch.id, epoch.generation);
      auto const phase = epoch.snapshot_available ? epoch.phase : "snapshot unavailable";
      auto const value = epoch.has_node ? fmt::format("data {}", epoch.node_id) : "-";
      auto const next = epoch.has_next_epoch ? fmt::format("epoch {}", epoch.next_epoch_id) : "-";
      auto const created = !epoch.creation_site.location.empty()
                               ? epoch.creation_site.location
                               : (!epoch.creation_timestamp.empty() ? epoch.creation_timestamp : "-");
      table.row(epoch_label, phase, value, detail::task_registry_join_ids(epoch.reader_task_ids),
                detail::task_registry_join_ids(epoch.writer_task_ids), next, created);
    }
  }

  if (!snapshot.tasks.empty())
  {
    if (std::ranges::any_of(snapshot.tasks, [](auto const& task) { return !task.failure.empty(); }))
    {
      auto& failures = report.table("Coroutine failures");
      failures.column("task", presentation::table_alignment::left)
          .column("exception", presentation::table_alignment::left);
      for (auto const& task : snapshot.tasks)
      {
        if (task.failure.empty()) continue;
        auto name =
            task.label.empty() ? fmt::format("task {}", task.id) : fmt::format("task {}: {}", task.id, task.label);
        failures.row(name, task.failure);
      }
    }

    auto& table = report.table("Coroutine tasks");
    table.column("task", presentation::table_alignment::left)
        .column("state", presentation::table_alignment::left)
        .column("transitions")
        .column("argument dependencies", presentation::table_alignment::left)
        .column("active awaits", presentation::table_alignment::left)
        .column("source", presentation::table_alignment::left);

    for (auto const& task : snapshot.tasks)
    {
      auto name =
          task.label.empty() ? fmt::format("task {}", task.id) : fmt::format("task {}: {}", task.id, task.label);
      auto dependencies = fmt::format("read: {}; write: {}", detail::task_registry_join_ids(task.read_dependencies),
                                      detail::task_registry_join_ids(task.write_dependencies));
      table.row(name, detail::task_registry_task_state(task, diagnostics), task.transition_count, dependencies,
                detail::task_registry_active_awaits(task), detail::task_registry_task_source(task));
    }
  }

  return report;
}

} // namespace uni20
