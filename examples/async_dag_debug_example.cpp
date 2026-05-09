#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/config.hpp>

#include <chrono>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <string>
#include <thread>
#include <utility>

using namespace uni20::async;

namespace
{

AsyncTask add_values(ReadBuffer<int> lhs, ReadBuffer<int> rhs, WriteBuffer<int> out)
{
  auto const left = co_await lhs;
  auto const right = co_await rhs;
  co_await out = left + right;
  co_return;
}

AsyncTask scale_value(ReadBuffer<int> input, WriteBuffer<int> out, int factor)
{
  auto const value = co_await input;
  co_await out = value * factor;
  co_return;
}

AsyncTask write_value(WriteBuffer<int> out, int value)
{
  co_await out = value;
  co_return;
}

std::filesystem::path dot_path(std::filesystem::path const& output_dir, char const* name)
{
  return output_dir / fmt::format("async-dag-{}.dot", name);
}

void write_dot(std::filesystem::path const& path, bool best_effort = false)
{
  auto const ok = best_effort ? uni20::TaskRegistry::dump_graphviz_file_best_effort(path.string())
                              : uni20::TaskRegistry::dump_graphviz_file(path.string());
  fmt::print("{} {}\n", ok ? "wrote" : "failed to write", path.string());
}

bool wait_for_service_file(std::filesystem::path const& output_dir)
{
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline)
  {
    for (auto const& entry : std::filesystem::directory_iterator(output_dir))
    {
      auto const filename = entry.path().filename().string();
      if (entry.path().extension() == ".dot" && filename.find("async-dag-service.") == 0) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

void print_build_mode()
{
#if UNI20_DEBUG_DAG
  fmt::print("UNI20_DEBUG_DAG=ON: DOT includes async value nodes plus coarse and concrete buffer edges.\n");
#elif UNI20_DEBUG_ASYNC_TASKS
  fmt::print("UNI20_DEBUG_ASYNC_TASKS=ON, UNI20_DEBUG_DAG=OFF: DOT includes task/epoch state but no value edges.\n");
  fmt::print("Rebuild with -DUNI20_DEBUG_DAG=ON to include async value nodes and dependency edges.\n");
#else
  fmt::print("UNI20_DEBUG_ASYNC_TASKS=OFF: TaskRegistry is a dummy and Graphviz DOT output would be empty.\n");
#endif
}

bool task_registry_enabled() noexcept
{
#if UNI20_DEBUG_ASYNC_TASKS
  return true;
#else
  return false;
#endif
}

void print_rebuild_hint()
{
  fmt::print("No DOT files were written.\n");
  fmt::print("Configure and build a DAG-instrumented example with:\n\n");
  fmt::print("  cmake -S . -B ./build_codex/build_gcc13_debug_dag \\\n");
  fmt::print("    -DCMAKE_BUILD_TYPE=Debug \\\n");
  fmt::print("    -DUNI20_DEBUG_DAG=ON\n");
  fmt::print("  cmake --build ./build_codex/build_gcc13_debug_dag --target async_dag_debug_example\n\n");
  fmt::print("Then run:\n");
  fmt::print("  ./build_codex/build_gcc13_debug_dag/examples/async_dag_debug_example /tmp/uni20-dag-example\n");
}

void print_graph_legend(std::filesystem::path const& output_dir)
{
  fmt::print("\nHow to read the DAG snapshots:\n");
  fmt::print("  data_N  : Async<T> value node; labels show storage identity, state, and optional value.\n");
  fmt::print("            state=unconstructed means shared_storage has no current T object.\n");
  fmt::print("  task_N  : coroutine task; state is constructed/running/suspended/leaked.\n");
  fmt::print("  epoch_N : read/write ordering generation for one async value.\n");
  fmt::print("  arg read/write     : coarse constructor-time ReadBuffer/WriteBuffer dependency.\n");
  fmt::print("  co_await read/write: dependency observed when the coroutine actually awaited.\n");
  fmt::print("  await read/write   : task currently queued on an epoch.\n");
  fmt::print("  red or pink        : diagnostic highlight, such as missing writer or dependency cycle.\n");
  fmt::print("  later snapshots may be sparse after completed tasks and epochs are destroyed.\n");
  fmt::print("Start with: {}\n", dot_path(output_dir, "02-suspended").string());
  fmt::print("For more detail see docs/async/dag_debug_examples.md\n");
}

} // namespace

int main(int argc, char** argv)
{
  print_build_mode();

  if (!task_registry_enabled())
  {
    print_rebuild_hint();
    return 1;
  }

  auto output_dir = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("async-dag-example-output");
  std::filesystem::create_directories(output_dir);

  fmt::print("Graphviz DOT output directory: {}\n", output_dir.string());

  DebugScheduler scheduler;
  Async<int> a = 2;
  Async<int> b = 5;
  Async<int> sum;
  Async<int> scaled_sum;
  Async<int> late_input;
  Async<int> late_scaled;
  a.debug_name("a");
  b.debug_name("b");
  sum.debug_name("sum");
  scaled_sum.debug_name("scaled_sum");
  late_input.debug_name("late_input");
  late_scaled.debug_name("late_scaled");

  auto sum_task = add_values(a.read(), b.read(), sum.write());
  auto scale_task = scale_value(sum.read(), scaled_sum.write(), 3);
  auto blocked_task = scale_value(late_input.read(), late_scaled.write(), 10);
  sum_task.debug_name("sum = a + b");
  scale_task.debug_name("scaled_sum = sum * 3");
  blocked_task.debug_name("late_scaled = late_input * 10");

  write_dot(dot_path(output_dir, "01-constructed"));

  scheduler.schedule(std::move(sum_task));
  scheduler.schedule(std::move(scale_task));
  scheduler.schedule(std::move(blocked_task));
  scheduler.run();

  write_dot(dot_path(output_dir, "02-suspended"));

  scheduler.run_all();
  write_dot(dot_path(output_dir, "03-partial"), true);

  auto late_writer = write_value(late_input.write(), 7);
  late_writer.debug_name("late_input = 7");
  scheduler.schedule(std::move(late_writer));
  scheduler.run_all();
  write_dot(dot_path(output_dir, "04-complete"), true);

  uni20::TaskRegistry::GraphvizDumpOptions request_options;
  request_options.output_dir = output_dir.string();
  request_options.file_prefix = "async-dag-request";
  uni20::TaskRegistry::request_graphviz_dump();
  uni20::TaskRegistry::service_debug_requests(request_options);

  uni20::TaskRegistry::DiagnosticsServiceOptions service_options;
  service_options.dump_options.output_dir = output_dir.string();
  service_options.dump_options.file_prefix = "async-dag-service";
  service_options.request_file = (output_dir / "dump.request").string();
  service_options.poll_interval_ms = 10;

  if (uni20::TaskRegistry::start_diagnostics_service(service_options))
  {
    {
      std::ofstream request(service_options.request_file);
      request << "dump\n";
    }
    auto const produced = wait_for_service_file(output_dir);
    uni20::TaskRegistry::stop_diagnostics_service();
    fmt::print("{} diagnostics-service DOT request\n", produced ? "serviced" : "timed out waiting for");
  }

  fmt::print("Final value: {}\n", late_scaled.get_wait(scheduler));
  fmt::print("Render one DOT file with: dot -Tsvg {} -o async-dag.svg\n", dot_path(output_dir, "02-suspended").string());
  print_graph_legend(output_dir);
  return 0;
}
