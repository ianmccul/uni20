/**
 * \file async_dag_deadlock_tbb_example.cpp
 * \brief Demonstrates best-effort async DAG debugging for a deliberate TBB dataflow deadlock.
 */

#include <uni20/async/async.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/config.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <string>
#include <thread>
#include <utility>

using namespace uni20::async;

namespace
{

AsyncTask copy_after_read(ReadBuffer<int> input, WriteBuffer<int> output, int offset)
{
  int const value = co_await input;
  input.release();
  co_await output = value + offset;
  co_return;
}

std::filesystem::path dot_path(std::filesystem::path const& output_dir, char const* name)
{
  return output_dir / fmt::format("async-dag-deadlock-tbb-{}.dot", name);
}

struct SnapshotWriteResult
{
    bool wrote{false};
    uni20::TaskRegistry::GraphSnapshot snapshot{};
    uni20::TaskRegistry::GraphDiagnostics diagnostics{};
};

SnapshotWriteResult write_dot_snapshot(std::filesystem::path const& path)
{
  SnapshotWriteResult result;
  result.snapshot = uni20::TaskRegistry::snapshot_best_effort();
  result.diagnostics = uni20::TaskRegistry::diagnose_snapshot(result.snapshot);
  auto const dot = uni20::TaskRegistry::graphviz_dot(result.snapshot, result.diagnostics);

  std::ofstream output(path);
  if (output)
  {
    output << dot;
    result.wrote = output.good();
  }
  fmt::print("{} {}\n", result.wrote ? "wrote" : "failed to write", path.string());
  return result;
}

int parse_sleep_seconds(char const* raw_value, int fallback)
{
  if (!raw_value || raw_value[0] == '\0') return fallback;
  char* end = nullptr;
  auto const parsed = std::strtol(raw_value, &end, 10);
  if (end == raw_value || *end != '\0' || parsed < 0 || parsed > 3600) return fallback;
  return static_cast<int>(parsed);
}

void print_build_mode()
{
#if UNI20_DEBUG_DAG
  fmt::print("UNI20_DEBUG_DAG=ON: DOT includes async value nodes, dependency edges, and diagnostics.\n");
#elif UNI20_DEBUG_ASYNC_TASKS
  fmt::print("UNI20_DEBUG_ASYNC_TASKS=ON, UNI20_DEBUG_DAG=OFF: cycle diagnostics will be incomplete.\n");
  fmt::print("Rebuild with -DUNI20_DEBUG_DAG=ON to include async value nodes and dependency edges.\n");
#else
  fmt::print("UNI20_DEBUG_ASYNC_TASKS=OFF: TaskRegistry is a dummy and Graphviz DOT output would be empty.\n");
#endif
}

bool dag_debug_enabled() noexcept
{
#if UNI20_DEBUG_DAG
  return true;
#else
  return false;
#endif
}

void print_rebuild_hint()
{
  fmt::print("No useful deadlock DOT files were written.\n");
  fmt::print("Configure and build a DAG-instrumented TBB example with:\n\n");
  fmt::print("  cmake -S . -B ./build_codex/build_gcc13_debug_dag \\\n");
  fmt::print("    -DCMAKE_BUILD_TYPE=Debug \\\n");
  fmt::print("    -DUNI20_DEBUG_DAG=ON\n");
  fmt::print("  cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_deadlock_tbb_example\n\n");
  fmt::print("Then run:\n");
  fmt::print(
      "  ./build_codex/build_gcc13_debug_dag/examples/async_dag_deadlock_tbb_example /tmp/uni20-dag-deadlock 2\n");
}

void print_diagnostics(uni20::TaskRegistry::GraphDiagnostics const& diagnostics)
{
  fmt::print("\nDAG diagnostic summary:\n");
  if (diagnostics.notes.empty())
  {
    fmt::print("  notes: none\n");
  }
  else
  {
    fmt::print("  notes:\n");
    for (std::string const& note : diagnostics.notes)
      fmt::print("    - {}\n", note);
  }

  fmt::print("  blocked read tasks: {}\n", diagnostics.blocked_read_task_ids.size());
  fmt::print("  blocked write tasks: {}\n", diagnostics.blocked_write_task_ids.size());
  fmt::print("  cycle tasks: {}\n", diagnostics.cycle_task_ids.size());
  fmt::print("  cycle data nodes: {}\n", diagnostics.cycle_node_ids.size());
  fmt::print("  missing-writer nodes: {}\n", diagnostics.missing_writer_node_ids.size());
}

void print_graph_legend(std::filesystem::path const& output_dir)
{
  fmt::print("\nHow to read this example:\n");
  fmt::print("  left and right start unconstructed.\n");
  fmt::print("  task 'left = right + 1' reads right before it can write left.\n");
  fmt::print("  task 'right = left + 1' reads left before it can write right.\n");
  fmt::print("  after the sleep, both tasks should be suspended on reads and the graph should highlight a cycle.\n");
  fmt::print("Render the blocked snapshot with:\n");
  fmt::print("  dot -Tsvg {} -o async-dag-deadlock-tbb.svg\n", dot_path(output_dir, "02-after-sleep").string());
  fmt::print("For more detail see docs/async/dag_debug_examples.md\n");
}

} // namespace

int main(int argc, char** argv)
{
  print_build_mode();

  if (!dag_debug_enabled())
  {
    print_rebuild_hint();
    return 1;
  }

  auto output_dir = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("async-dag-deadlock-tbb-output");
  int const sleep_seconds = argc > 2 ? parse_sleep_seconds(argv[2], 2) : 2;
  std::filesystem::create_directories(output_dir);

  fmt::print("Graphviz DOT output directory: {}\n", output_dir.string());
  fmt::print("Main thread sleep before debug snapshot: {} second(s)\n", sleep_seconds);

  TbbScheduler scheduler{2};
  scheduler.pause();

  Async<int> left;
  Async<int> right;
  left.debug_name("left");
  right.debug_name("right");

  auto left_task = copy_after_read(right.read(), left.write(), 1);
  auto right_task = copy_after_read(left.read(), right.write(), 1);
  left_task.debug_name("left = right + 1");
  right_task.debug_name("right = left + 1");

  scheduler.schedule(std::move(left_task));
  scheduler.schedule(std::move(right_task));

  write_dot_snapshot(dot_path(output_dir, "01-scheduled"));

  scheduler.resume();
  fmt::print("TBB workers released. Sleeping before taking a best-effort debug snapshot.\n");
  std::this_thread::sleep_for(std::chrono::seconds(sleep_seconds));

  auto const after_sleep = write_dot_snapshot(dot_path(output_dir, "02-after-sleep"));
  print_diagnostics(after_sleep.diagnostics);

  fmt::print(
      "\nThis example intentionally does not call get_wait() or run_all(); the dataflow cycle cannot complete.\n");
  print_graph_legend(output_dir);
  return after_sleep.diagnostics.cycle_task_ids.empty() ? 2 : 0;
}
