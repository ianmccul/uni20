#pragma once

/**
 * \file task_registry_dummy.hpp
 * \brief Provides a no-op task registry used when async debug tracking is disabled.
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

/// \brief No-op task registry implementation for non-debug async builds.
class TaskRegistry {
  public:
    /// \brief Verbosity levels mirrored from the debug registry interface.
    enum class DumpMode
    {
      None,
      Basic,
      Full,
    };

    /// \brief Lifecycle states mirrored from the debug registry interface.
    enum class TaskState
    {
      Constructed,
      Running,
      Suspended,
      Leaked,
    };

    /// \brief Reader/writer role mirrored from the debug registry interface.
    enum class EpochTaskRole
    {
      Reader,
      Writer,
    };

    /// \brief File-output options mirrored from the debug registry interface.
    struct GraphvizDumpOptions
    {
        std::string output_dir{"/tmp"};
        std::string file_prefix{"uni20-dag"};
    };

    /// \brief Diagnostics-service options mirrored from the debug registry interface.
    struct DiagnosticsServiceOptions
    {
        GraphvizDumpOptions dump_options{};
        int signal_number{0};
        std::string request_file{};
        int poll_interval_ms{250};
        bool block_signal_in_calling_thread{true};
    };

    /// \brief Runtime controls for stacktrace formatting mirrored from the debug registry.
    using StacktraceOptions = TaskRegistryStacktraceOptions;

    /// \brief Structured async DAG snapshot type mirrored from the debug registry.
    using GraphSnapshot = TaskRegistryGraphSnapshot;

    /// \brief Structured async DAG diagnostic result type mirrored from the debug registry.
    using GraphDiagnostics = TaskRegistryGraphDiagnostics;

    /// \brief No-op task registration hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void register_task(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op task destruction hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void destroy_task(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op task leak hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void leak_task(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op running-state hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void mark_running(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op suspended-state hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void mark_suspended(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op scheduler submission provenance hook.
    /// \param h Coroutine handle ignored in dummy mode.
    static constexpr void record_task_scheduled(std::coroutine_handle<> h) noexcept { static_cast<void>(h); }
    /// \brief No-op coarse DAG dependency hook.
    /// \param h Coroutine handle ignored in dummy mode.
    /// \param read_dependencies Read dependency nodes ignored in dummy mode.
    /// \param write_dependencies Write dependency nodes ignored in dummy mode.
    static constexpr void
    record_task_dependencies(std::coroutine_handle<> h, std::vector<async::NodeInfo const*> const& read_dependencies,
                             std::vector<async::NodeInfo const*> const& write_dependencies) noexcept
    {
      static_cast<void>(h);
      static_cast<void>(read_dependencies);
      static_cast<void>(write_dependencies);
    }
    /// \brief No-op concrete await DAG dependency hook.
    /// \param h Coroutine handle ignored in dummy mode.
    /// \param node Dependency node ignored in dummy mode.
    /// \param role Dependency role ignored in dummy mode.
    static constexpr void record_await_dependency(std::coroutine_handle<> h, async::NodeInfo const* node,
                                                  EpochTaskRole role) noexcept
    {
      static_cast<void>(h);
      static_cast<void>(node);
      static_cast<void>(role);
    }
    /// \brief No-op task label hook.
    /// \param h Coroutine handle ignored in dummy mode.
    /// \param label Label ignored in dummy mode.
    static void name_task(std::coroutine_handle<> h, std::string const& label) noexcept
    {
      static_cast<void>(h);
      static_cast<void>(label);
    }
    /// \brief No-op async value label hook.
    /// \param node Async value node ignored in dummy mode.
    /// \param label Label ignored in dummy mode.
    static void name_async_value(async::NodeInfo const* node, std::string const& label) noexcept
    {
      static_cast<void>(node);
      static_cast<void>(label);
    }
    /// \brief No-op epoch-context registration hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    static constexpr void register_epoch_context(async::EpochContext const* epoch_context) noexcept
    {
      static_cast<void>(epoch_context);
    }
    /// \brief No-op epoch-context destruction hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    static constexpr void destroy_epoch_context(async::EpochContext const* epoch_context) noexcept
    {
      static_cast<void>(epoch_context);
    }
    /// \brief No-op epoch-task binding hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \param h Coroutine handle ignored in dummy mode.
    /// \param role Role value ignored in dummy mode.
    static constexpr void bind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                          EpochTaskRole role) noexcept
    {
      static_cast<void>(epoch_context);
      static_cast<void>(h);
      static_cast<void>(role);
    }
    /// \brief No-op epoch-task unbinding hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \param h Coroutine handle ignored in dummy mode.
    /// \param role Role value ignored in dummy mode.
    static constexpr void unbind_epoch_task(async::EpochContext const* epoch_context, std::coroutine_handle<> h,
                                            EpochTaskRole role) noexcept
    {
      static_cast<void>(epoch_context);
      static_cast<void>(h);
      static_cast<void>(role);
    }
    /// \brief Returns an empty reader list in dummy mode.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \return Empty vector.
    static std::vector<std::coroutine_handle<>> epoch_reader_tasks(async::EpochContext const* epoch_context)
    {
      static_cast<void>(epoch_context);
      return {};
    }
    /// \brief Returns an empty writer list in dummy mode.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \return Empty vector.
    static std::vector<std::coroutine_handle<>> epoch_writer_tasks(async::EpochContext const* epoch_context)
    {
      static_cast<void>(epoch_context);
      return {};
    }
    /// \brief Returns `DumpMode::None` in dummy mode.
    /// \return Always `DumpMode::None`.
    static constexpr DumpMode dump_mode() noexcept { return DumpMode::None; }
    /// \brief No-op epoch-context dump hook.
    /// \param epoch_context Epoch context pointer ignored in dummy mode.
    /// \param reason Optional reason string ignored in dummy mode.
    static constexpr void dump_epoch_context(async::EpochContext const* epoch_context,
                                             char const* reason = nullptr) noexcept
    {
      static_cast<void>(epoch_context);
      static_cast<void>(reason);
    }
    /// \brief No-op global dump hook.
    static constexpr void dump() noexcept {}
    /// \brief Returns an empty structured graph snapshot in dummy mode.
    /// \return Empty graph snapshot.
    static GraphSnapshot snapshot() { return GraphSnapshot{}; }
    /// \brief Returns an empty best-effort structured graph snapshot in dummy mode.
    /// \return Empty graph snapshot.
    static GraphSnapshot snapshot_best_effort() { return snapshot(); }
    /// \brief Returns empty diagnostics for a graph snapshot in dummy mode.
    /// \param snapshot Graph snapshot ignored in dummy mode.
    /// \return Empty diagnostics.
    static GraphDiagnostics diagnose_snapshot(GraphSnapshot const& snapshot)
    {
      static_cast<void>(snapshot);
      return GraphDiagnostics{};
    }
    /// \brief Returns an empty Graphviz DOT document in dummy mode.
    /// \return Empty async DAG graph.
    static std::string graphviz_dot() { return "digraph uni20_async_dag {\n  rankdir=LR;\n}\n"; }
    /// \brief Returns an empty best-effort Graphviz DOT document in dummy mode.
    /// \return Empty async DAG graph.
    static std::string graphviz_dot_best_effort() { return graphviz_dot(); }
    /// \brief Renders a graph snapshot as Graphviz DOT in dummy mode.
    /// \param snapshot Graph snapshot ignored in dummy mode.
    /// \return Empty async DAG graph.
    static std::string graphviz_dot(GraphSnapshot const& snapshot)
    {
      static_cast<void>(snapshot);
      return graphviz_dot();
    }
    /// \brief Renders a graph snapshot and diagnostics as Graphviz DOT in dummy mode.
    /// \param snapshot Graph snapshot ignored in dummy mode.
    /// \param diagnostics Diagnostics ignored in dummy mode.
    /// \return Empty async DAG graph.
    static std::string graphviz_dot(GraphSnapshot const& snapshot, GraphDiagnostics const& diagnostics)
    {
      static_cast<void>(snapshot);
      static_cast<void>(diagnostics);
      return graphviz_dot();
    }
    /// \brief Prints an empty Graphviz DOT document in dummy mode.
    /// \param stream Destination stream.
    static void dump_graphviz(std::FILE* stream = stderr)
    {
      if (!stream) return;
      auto dot = graphviz_dot();
      std::fputs(dot.c_str(), stream);
    }
    /// \brief Writes an empty Graphviz DOT document in dummy mode.
    /// \param path Destination path.
    /// \return true if the file was written successfully.
    static bool dump_graphviz_file(std::string const& path)
    {
      if (auto* stream = std::fopen(path.c_str(), "w"); stream)
      {
        auto dot = graphviz_dot();
        auto const ok = std::fputs(dot.c_str(), stream) >= 0;
        return std::fclose(stream) == 0 && ok;
      }
      return false;
    }
    /// \brief Writes an empty best-effort Graphviz DOT document in dummy mode.
    /// \param path Destination path.
    /// \return true if the file was written successfully.
    static bool dump_graphviz_file_best_effort(std::string const& path) { return dump_graphviz_file(path); }
    /// \brief Builds default file-output options in dummy mode.
    /// \return Default Graphviz dump options.
    static GraphvizDumpOptions default_graphviz_dump_options() { return GraphvizDumpOptions{}; }
    /// \brief Builds default diagnostics-service options in dummy mode.
    /// \return Default diagnostics-service options.
    static DiagnosticsServiceOptions default_diagnostics_service_options() { return DiagnosticsServiceOptions{}; }
    /// \brief Builds default stacktrace formatting options in dummy mode.
    /// \return Default stacktrace formatting options.
    static StacktraceOptions default_stacktrace_options() { return StacktraceOptions{}; }
    /// \brief Returns stacktrace formatting options in dummy mode.
    /// \return Default stacktrace formatting options.
    static StacktraceOptions stacktrace_options() { return default_stacktrace_options(); }
    /// \brief Ignores stacktrace formatting options in dummy mode.
    /// \param options Stacktrace options ignored in dummy mode.
    static void set_stacktrace_options(StacktraceOptions const& options) noexcept { static_cast<void>(options); }
    /// \brief Resets stacktrace formatting options in dummy mode.
    static constexpr void reset_stacktrace_options() noexcept {}
    /// \brief Builds a dummy default Graphviz dump path.
    /// \return Path using the default directory and prefix.
    static std::string default_graphviz_dump_path()
    {
      return default_graphviz_dump_path(default_graphviz_dump_options());
    }
    /// \brief Builds a dummy default Graphviz dump path.
    /// \param options File-output options.
    /// \return Path using the configured directory and prefix.
    static std::string default_graphviz_dump_path(GraphvizDumpOptions const& options)
    {
      auto dir = options.output_dir.empty() ? std::string(".") : options.output_dir;
      if (!dir.empty() && dir.back() == '/') dir.pop_back();
      return dir + "/" + options.file_prefix + ".dot";
    }
    /// \brief No-op dump request hook in dummy mode.
    static constexpr void request_graphviz_dump() noexcept {}
    /// \brief No-op request service hook in dummy mode.
    /// \return Always false.
    static bool service_debug_requests() noexcept { return false; }
    /// \brief No-op request service hook in dummy mode.
    /// \param options File-output options ignored in dummy mode.
    /// \return Always false.
    static bool service_debug_requests(GraphvizDumpOptions const& options) noexcept
    {
      static_cast<void>(options);
      return false;
    }
    /// \brief No-op diagnostics service start hook in dummy mode.
    /// \return Always false.
    static bool start_diagnostics_service() noexcept { return false; }
    /// \brief No-op diagnostics service start hook in dummy mode.
    /// \param options Service options ignored in dummy mode.
    /// \return Always false.
    static bool start_diagnostics_service(DiagnosticsServiceOptions const& options) noexcept
    {
      static_cast<void>(options);
      return false;
    }
    /// \brief No-op diagnostics service stop hook in dummy mode.
    static constexpr void stop_diagnostics_service() noexcept {}
};

} // namespace uni20
