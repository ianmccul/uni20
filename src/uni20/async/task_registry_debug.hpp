#pragma once

/**
 * \file task_registry_debug.hpp
 * \brief Declares the debug task registry used for async coroutine diagnostics.
 */

#include "task_registry_snapshot.hpp"

#include <coroutine>
#include <cstdio>
#include <string>
#include <vector>

namespace uni20::async
{
class EpochContext;
class NodeInfo;
} // namespace uni20::async

namespace uni20
{

/// \brief Forward declaration for the debug task registry type.
class TaskRegistryDebug;

/// \brief Alias selecting the active task registry implementation.
using TaskRegistry = TaskRegistryDebug;

/// \brief Tracks coroutine and epoch-context lifecycle events in debug builds.
class TaskRegistryDebug {
  public:
    /// \brief Verbosity levels for registry dump output.
    enum class DumpMode
    {
      None,
      Basic,
      Full,
    };

    /// \brief Logical lifecycle states tracked for each coroutine handle.
    enum class TaskState
    {
      Constructed,
      Running,
      Suspended,
      Leaked,
    };

    /// \brief Role of a coroutine relative to an epoch context.
    enum class EpochTaskRole
    {
      Reader,
      Writer,
    };

    /// \brief File-output options for Graphviz DAG dumps.
    struct GraphvizDumpOptions
    {
        std::string output_dir{"/tmp"};
        std::string file_prefix{"uni20-dag"};
    };

    /// \brief Options for the opt-in background DAG diagnostics service.
    struct DiagnosticsServiceOptions
    {
        GraphvizDumpOptions dump_options{};
        int signal_number{0};
        std::string request_file{};
        int poll_interval_ms{250};
        bool block_signal_in_calling_thread{true};
    };

    /// \brief Runtime controls for stacktrace formatting in debug output.
    using StacktraceOptions = TaskRegistryStacktraceOptions;

    /// \brief Structured async DAG snapshot type.
    using GraphSnapshot = TaskRegistryGraphSnapshot;

    /// \brief Structured async DAG diagnostic result type.
    using GraphDiagnostics = TaskRegistryGraphDiagnostics;

    /// \brief Registers a newly-created coroutine handle.
    /// \param h Coroutine handle to register.
    static void register_task(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as destroyed.
    /// \param h Coroutine handle being destroyed.
    static void destroy_task(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as leaked.
    /// \param h Coroutine handle intentionally leaked.
    static void leak_task(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as currently running.
    /// \param h Coroutine handle that resumed execution.
    static void mark_running(std::coroutine_handle<> h);
    /// \brief Marks a coroutine handle as suspended.
    /// \param h Coroutine handle that suspended execution.
    static void mark_suspended(std::coroutine_handle<> h);
    /// \brief Records the call stack where a coroutine was submitted to a scheduler.
    /// \param h Coroutine handle being scheduled.
    static void record_task_scheduled(std::coroutine_handle<> h);
    /// \brief Records constructor-discovered coarse DAG dependencies for a coroutine.
    /// \param h Coroutine handle whose promise collected the dependencies.
    /// \param read_dependencies Async value nodes read by the coroutine.
    /// \param write_dependencies Async value nodes written by the coroutine.
    static void record_task_dependencies(std::coroutine_handle<> h,
                                         std::vector<async::NodeInfo const*> const& read_dependencies,
                                         std::vector<async::NodeInfo const*> const& write_dependencies);
    /// \brief Records a buffer dependency discovered at a concrete co_await site.
    /// \param h Coroutine handle that is awaiting the dependency.
    /// \param node Async value node being awaited.
    /// \param role Reader or writer role for the await dependency.
    static void record_await_dependency(std::coroutine_handle<> h, async::NodeInfo const* node, EpochTaskRole role);
    /// \brief Assigns an optional human-readable label to a coroutine task.
    /// \param h Coroutine handle to label.
    /// \param label Label to show in diagnostic graph output.
    static void name_task(std::coroutine_handle<> h, std::string const& label);
    /// \brief Assigns an optional human-readable label to an async value node.
    /// \param node Async value node to label.
    /// \param label Label to show in diagnostic graph output.
    static void name_async_value(async::NodeInfo const* node, std::string const& label);
    /// \brief Registers an epoch context for debug tracking.
    /// \param epoch_context Epoch context pointer to register.
    static void register_epoch_context(async::EpochContext const* epoch_context);
    /// \brief Removes an epoch context from debug tracking.
    /// \param epoch_context Epoch context pointer to remove.
    static void destroy_epoch_context(async::EpochContext const* epoch_context);
    /// \brief Associates a coroutine with an epoch context as reader or writer.
    /// \param epoch_context Epoch context being accessed.
    /// \param h Coroutine handle to bind.
    /// \param role Reader or writer role for the binding.
    static void bind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                EpochTaskRole role);
    /// \brief Removes a coroutine-role binding from an epoch context.
    /// \param epoch_context Epoch context that owns the binding.
    /// \param h Coroutine handle to unbind.
    /// \param role Reader or writer role being removed.
    static void unbind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                  EpochTaskRole role);
    /// \brief Returns coroutine handles currently bound as readers.
    /// \param epoch_context Epoch context to inspect.
    /// \return Reader coroutine handles for the requested epoch context.
    static std::vector<std::coroutine_handle<>> epoch_reader_tasks(async::EpochContext const* epoch_context);
    /// \brief Returns coroutine handles currently bound as writers.
    /// \param epoch_context Epoch context to inspect.
    /// \return Writer coroutine handles for the requested epoch context.
    static std::vector<std::coroutine_handle<>> epoch_writer_tasks(async::EpochContext const* epoch_context);
    /// \brief Reports the current debug dump mode.
    /// \return Active dump-mode setting.
    static DumpMode dump_mode() noexcept;
    /// \brief Prints diagnostics for a specific epoch context.
    /// \param epoch_context Epoch context to print.
    /// \param reason Optional label for the dump trigger.
    static void dump_epoch_context(async::EpochContext const* epoch_context, char const* reason = nullptr);
    /// \brief Captures the current task/epoch/value DAG as structured data.
    /// \return Structured graph snapshot.
    static GraphSnapshot snapshot();
    /// \brief Captures a best-effort task/epoch/value DAG snapshot without indefinite lock waits.
    /// \return Structured graph snapshot, possibly marked unavailable or partial.
    static GraphSnapshot snapshot_best_effort();
    /// \brief Diagnoses blocked tasks, missing writers, and dependency cycles in a graph snapshot.
    /// \param snapshot Structured graph snapshot to inspect.
    /// \return Diagnostic annotations for the snapshot.
    static GraphDiagnostics diagnose_snapshot(GraphSnapshot const& snapshot);
    /// \brief Returns a Graphviz DOT snapshot of the currently tracked async DAG.
    /// \return Graphviz DOT document describing tasks, epochs, and tracked dependencies.
    static std::string graphviz_dot();
    /// \brief Returns a best-effort Graphviz DOT snapshot without waiting indefinitely for locks.
    /// \return Graphviz DOT document, possibly partial if debug locks are currently held.
    static std::string graphviz_dot_best_effort();
    /// \brief Renders a graph snapshot as Graphviz DOT.
    /// \param snapshot Structured graph snapshot to render.
    /// \return Graphviz DOT document.
    static std::string graphviz_dot(GraphSnapshot const& snapshot);
    /// \brief Renders a graph snapshot and diagnostics as Graphviz DOT.
    /// \param snapshot Structured graph snapshot to render.
    /// \param diagnostics Diagnostic annotations for the snapshot.
    /// \return Graphviz DOT document.
    static std::string graphviz_dot(GraphSnapshot const& snapshot, GraphDiagnostics const& diagnostics);
    /// \brief Prints a Graphviz DOT snapshot to a C stream.
    /// \param stream Destination stream. Defaults to stderr.
    static void dump_graphviz(std::FILE* stream = stderr);
    /// \brief Writes a Graphviz DOT snapshot to a file.
    /// \param path Destination path.
    /// \return true if the file was written successfully.
    static bool dump_graphviz_file(std::string const& path);
    /// \brief Writes a best-effort Graphviz DOT snapshot to a file.
    /// \param path Destination path.
    /// \return true if the file was written successfully.
    static bool dump_graphviz_file_best_effort(std::string const& path);
    /// \brief Returns default Graphviz file-output options.
    /// \return Options derived from environment variables when present.
    static GraphvizDumpOptions default_graphviz_dump_options();
    /// \brief Returns default diagnostics-service options.
    /// \return Service options derived from environment variables when present.
    static DiagnosticsServiceOptions default_diagnostics_service_options();
    /// \brief Returns default stacktrace formatting options.
    /// \return Stacktrace options derived from environment variables when present.
    static StacktraceOptions default_stacktrace_options();
    /// \brief Returns active stacktrace formatting options.
    /// \return Active stacktrace formatting options.
    static StacktraceOptions stacktrace_options();
    /// \brief Replaces active stacktrace formatting options.
    /// \param options New stacktrace formatting options.
    static void set_stacktrace_options(StacktraceOptions const& options);
    /// \brief Resets active stacktrace formatting options from environment defaults.
    static void reset_stacktrace_options();
    /// \brief Creates a default DOT output path for the current process.
    /// \return Path containing output directory, prefix, process id, and sequence number.
    static std::string default_graphviz_dump_path();
    /// \brief Creates a default DOT output path for the current process.
    /// \param options File-output options.
    /// \return Path containing output directory, prefix, process id, and sequence number.
    static std::string default_graphviz_dump_path(GraphvizDumpOptions const& options);
    /// \brief Queues one Graphviz dump request for scheduler or service processing.
    static void request_graphviz_dump() noexcept;
    /// \brief Processes pending dump requests on the calling thread.
    /// \return true if at least one request was serviced.
    static bool service_debug_requests();
    /// \brief Processes pending dump requests on the calling thread.
    /// \param options File-output options for generated DOT files.
    /// \return true if at least one request was serviced.
    static bool service_debug_requests(GraphvizDumpOptions const& options);
    /// \brief Starts the optional background DAG diagnostics service.
    /// \return true if a service was started or was already running.
    static bool start_diagnostics_service();
    /// \brief Starts the optional background DAG diagnostics service.
    /// \param options Signal/control-file and output configuration.
    /// \return true if a service was started or was already running.
    static bool start_diagnostics_service(DiagnosticsServiceOptions const& options);
    /// \brief Stops the background DAG diagnostics service if it is running.
    static void stop_diagnostics_service() noexcept;
    /// \brief Prints global task-registry diagnostics.
    static void dump();
};

} // namespace uni20
