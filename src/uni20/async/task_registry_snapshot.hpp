#pragma once

/**
 * \file task_registry_snapshot.hpp
 * \brief Structured async task-registry graph snapshots and diagnostics.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace uni20
{

/// \brief Reader/writer role for a task-registry graph dependency.
enum class TaskRegistryGraphRole
{
  Reader,
  Writer,
};

/// \brief Optional source provenance attached to a task, epoch, or dependency edge.
struct TaskRegistryGraphProvenance
{
    std::string location{};
    std::string function{};
    std::string stacktrace{};
};

/// \brief Runtime controls for formatting captured task-registry stacktraces.
struct TaskRegistryStacktraceOptions
{
    std::size_t max_frames{32};
    bool include_internal_frames{true};
};

/// \brief Async value node captured in a task-registry graph snapshot.
struct TaskRegistryGraphDataNode
{
    std::uint64_t id{0};
    std::string label{};
    std::string type{};
    std::string storage_address{};
    std::string address{};
    std::string state{};
    std::string value{};
    bool value_constructed{true};
};

/// \brief Concrete awaited dependency captured for a task.
struct TaskRegistryGraphAwaitDependency
{
    std::uint64_t node_id{0};
    TaskRegistryGraphRole role{TaskRegistryGraphRole::Reader};
    TaskRegistryGraphProvenance await_site{};
};

/// \brief Coroutine task captured in a task-registry graph snapshot.
struct TaskRegistryGraphTask
{
    std::size_t id{0};
    std::string address{};
    std::string label{};
    std::string state{};
    std::size_t transition_count{0};
    std::string creation_timestamp{};
    std::string last_transition_timestamp{};
    std::string waiting_on{};
    TaskRegistryGraphProvenance creation_site{};
    TaskRegistryGraphProvenance schedule_site{};
    TaskRegistryGraphProvenance last_transition_site{};
    TaskRegistryGraphProvenance last_await_site{};
    std::vector<std::uint64_t> read_dependencies{};
    std::vector<std::uint64_t> write_dependencies{};
    std::vector<TaskRegistryGraphAwaitDependency> await_dependencies{};
    std::string failure{};
};

/// \brief Epoch context captured in a task-registry graph snapshot.
struct TaskRegistryGraphEpoch
{
    std::size_t id{0};
    std::string address{};
    bool snapshot_available{true};
    std::string creation_timestamp{};
    int generation{0};
    std::string phase{};
    bool has_next_epoch{false};
    std::size_t next_epoch_id{0};
    bool has_node{false};
    std::uint64_t node_id{0};
    int total_writers{0};
    int num_writers{0};
    int num_readers{0};
    bool writer_active{false};
    TaskRegistryGraphProvenance creation_site{};
    std::vector<std::size_t> reader_task_ids{};
    std::vector<std::size_t> writer_task_ids{};
};

/// \brief Structured task-registry graph snapshot.
struct TaskRegistryGraphSnapshot
{
    bool snapshot_available{true};
    std::string unavailable_reason{};
    std::vector<TaskRegistryGraphDataNode> data_nodes{};
    std::vector<TaskRegistryGraphEpoch> epochs{};
    std::vector<TaskRegistryGraphTask> tasks{};
};

/// \brief Diagnostics derived from a structured task-registry graph snapshot.
struct TaskRegistryGraphDiagnostics
{
    std::vector<std::string> notes{};
    std::vector<std::size_t> blocked_read_task_ids{};
    std::vector<std::size_t> blocked_write_task_ids{};
    std::vector<std::size_t> cycle_task_ids{};
    std::vector<std::uint64_t> cycle_node_ids{};
    std::vector<std::uint64_t> missing_writer_node_ids{};
    std::vector<std::size_t> missing_writer_epoch_ids{};
    std::vector<std::size_t> failed_task_ids{};
};

} // namespace uni20
