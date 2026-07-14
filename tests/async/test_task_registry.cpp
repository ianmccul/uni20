#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/buffers.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/config.hpp>
#include <uni20/core/scalar_precision.hpp>

#include "../common/env_var_guard.hpp"

#if defined(__linux__)
#include <signal.h>
#include <unistd.h>
#endif

using namespace uni20;
using namespace uni20::async;

#if UNI20_DEBUG_ASYNC_TASKS
namespace
{
using uni20::test::EnvVarGuard;

class StacktraceOptionsGuard {
  public:
    StacktraceOptionsGuard() : old_(TaskRegistry::stacktrace_options()) {}

    ~StacktraceOptionsGuard() { TaskRegistry::set_stacktrace_options(old_); }

  private:
    TaskRegistry::StacktraceOptions old_;
};

AsyncTask make_suspended_task() { co_return; }

AsyncTask wait_for_reader(ReadBuffer<int> reader)
{
  auto const& value = co_await reader;
  (void)value;
}

AsyncTask write_value(WriteBuffer<int> writer, int value)
{
  co_await writer = value;
  co_return;
}

AsyncTask copy_value(ReadBuffer<int> reader, WriteBuffer<int> writer)
{
  auto const& value = co_await reader;
  co_await writer = value;
  co_return;
}

AsyncTask sum_all_values(ReadBuffer<int> lhs, ReadBuffer<int> rhs, WriteBuffer<int> writer)
{
  auto [left, right] = co_await all(lhs, rhs);
  co_await writer = left + right;
  co_return;
}

struct ShapeLikeExtents
{
    static constexpr std::size_t rank() noexcept { return 3; }

    [[nodiscard]] std::size_t extent(std::size_t axis) const noexcept { return axis == 0 ? 2U : axis == 1 ? 3U : 4U; }
};

struct ShapeLikeValue
{
    [[nodiscard]] ShapeLikeExtents extents() const noexcept { return {}; }
};

struct CustomDebugValue
{
    int id{0};
};

struct OpaqueValue
{};

std::string uni20_async_debug_value(CustomDebugValue const& value) { return "custom id=" + std::to_string(value.id); }

std::filesystem::path make_temp_dir(std::string_view name)
{
  auto const stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  auto path = std::filesystem::temp_directory_path() / (std::string(name) + "-" + std::to_string(stamp));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string read_file(std::filesystem::path const& path)
{
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::filesystem::path find_dot_file(std::filesystem::path const& dir)
{
  if (!std::filesystem::exists(dir)) return {};
  for (auto const& entry : std::filesystem::directory_iterator(dir))
  {
    if (entry.path().extension() == ".dot") return entry.path();
  }
  return {};
}

bool wait_for_dot_file(std::filesystem::path const& dir, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline)
  {
    auto path = find_dot_file(dir);
    if (!path.empty() && read_file(path).find("digraph uni20_async_dag") != std::string::npos) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

int dump_mode_probe(char const* value)
{
  if (value)
    ::setenv("UNI20_DEBUG_ASYNC_TASKS", value, 1);
  else
    ::unsetenv("UNI20_DEBUG_ASYNC_TASKS");

  switch (TaskRegistry::dump_mode())
  {
    case TaskRegistry::DumpMode::None:
      return 0;
    case TaskRegistry::DumpMode::Basic:
      return 1;
    case TaskRegistry::DumpMode::Full:
      return 2;
  }

  return 99;
}

#if UNI20_DEBUG_DAG
int dependency_cycle_probe()
{
  DebugScheduler sched;
  Async<int> a;
  Async<int> b;
  a.debug_name("cycle a");
  b.debug_name("cycle b");

  auto task1 = copy_value(b.read(), a.write());
  auto task2 = copy_value(a.read(), b.write());
  task1.debug_name("cycle task 1");
  task2.debug_name("cycle task 2");

  sched.schedule(std::move(task1));
  sched.schedule(std::move(task2));
  sched.run();

  auto const dot = TaskRegistry::graphviz_dot();
  bool const ok = dot.find("DAG diagnostics") != std::string::npos &&
                  dot.find("dependency cycle") != std::string::npos &&
                  dot.find("diagnostic: dependency cycle") != std::string::npos &&
                  dot.find("cycle task 1") != std::string::npos && dot.find("cycle task 2") != std::string::npos &&
                  dot.find("cycle a") != std::string::npos && dot.find("cycle b") != std::string::npos;
  return ok ? 0 : 1;
}
#endif

} // namespace
#endif

#if UNI20_DEBUG_ASYNC_TASKS
TEST(TaskRegistryDebugDeathTest, DumpModeDefaultsToBasicWhenUnset)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe(nullptr)); }, ::testing::ExitedWithCode(1), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeParsesNoneSynonyms)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("0")); }, ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("off")); }, ::testing::ExitedWithCode(0), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeParsesFullSynonyms)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("2")); }, ::testing::ExitedWithCode(2), "");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("verbose")); }, ::testing::ExitedWithCode(2), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeTrimsAndNormalizesCase)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("  YeS  ")); }, ::testing::ExitedWithCode(1), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeFallsBackToBasicForUnknownValue)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("not-a-mode")); }, ::testing::ExitedWithCode(1), "");
}

#if UNI20_DEBUG_DAG
TEST(TaskRegistryDebugDeathTest, GraphvizDotHighlightsDependencyCycle)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dependency_cycle_probe()); }, ::testing::ExitedWithCode(0), "");
}
#endif

TEST(TaskRegistryDebugTest, DumpShowsTaskStateAndTransitions)
{
  auto task = make_suspended_task();

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const dump = testing::internal::GetCapturedStderr();

  EXPECT_NE(dump.find("Total tracked tasks: 1"), std::string::npos);
  EXPECT_NE(dump.find("Task 1:"), std::string::npos);
  EXPECT_NE(dump.find("task pointer:"), std::string::npos);
  EXPECT_NE(dump.find("transition count:"), std::string::npos);
  EXPECT_NE(dump.find("current state: suspended"), std::string::npos);
  EXPECT_NE(dump.find("creation timestamp:"), std::string::npos);
  EXPECT_NE(dump.find("last state-change: suspended"), std::string::npos);
  EXPECT_NE(dump.find("last state-change timestamp:"), std::string::npos);
  auto const task_pos = dump.find("Task 1:");
  auto const pointer_pos = dump.find("task pointer:");
  auto const transition_pos = dump.find("transition count:");
  auto const state_pos = dump.find("current state:");
  auto const creation_time_pos = dump.find("creation timestamp:");
  EXPECT_LT(task_pos, pointer_pos);
  EXPECT_LT(pointer_pos, transition_pos);
  EXPECT_LT(transition_pos, state_pos);
  EXPECT_LT(state_pos, creation_time_pos);
#if UNI20_HAS_STACKTRACE
  auto const creation_trace_pos = dump.find("creation stacktrace:");
  auto const last_state_pos = dump.find("last state-change:");
  auto const last_time_pos = dump.find("last state-change timestamp:");
  auto const last_trace_pos = dump.find("last state-change stacktrace:");
  EXPECT_NE(dump.find("creation stacktrace:"), std::string::npos);
  EXPECT_NE(dump.find("last state-change stacktrace:"), std::string::npos);
  EXPECT_LT(creation_time_pos, creation_trace_pos);
  EXPECT_LT(creation_trace_pos, last_state_pos);
  EXPECT_LT(last_state_pos, last_time_pos);
  EXPECT_LT(last_time_pos, last_trace_pos);
#else
  EXPECT_NE(dump.find("WARNING: std::stacktrace is unavailable"), std::string::npos);
  EXPECT_NE(dump.find("creation stacktrace: unavailable"), std::string::npos);
  EXPECT_NE(dump.find("last state-change stacktrace: unavailable"), std::string::npos);
  auto const last_state_pos = dump.find("last state-change:");
  auto const last_time_pos = dump.find("last state-change timestamp:");
  auto const creation_trace_pos = dump.find("creation stacktrace: unavailable");
  auto const last_trace_pos = dump.find("last state-change stacktrace: unavailable");
  EXPECT_LT(creation_time_pos, creation_trace_pos);
  EXPECT_LT(creation_trace_pos, last_state_pos);
  EXPECT_LT(last_state_pos, last_time_pos);
  EXPECT_LT(last_time_pos, last_trace_pos);
#endif

  task.resume();

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const after_resume_dump = testing::internal::GetCapturedStderr();
  EXPECT_NE(after_resume_dump.find("Total tracked tasks: 0"), std::string::npos);
}

TEST(TaskRegistryDebugTest, DumpShowsEpochContextBindingsForSuspendedTask)
{
  DebugScheduler sched;
  Async<int> value;

  sched.schedule(wait_for_reader(value.read()));
  sched.run();

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const dump = testing::internal::GetCapturedStderr();

  EXPECT_NE(dump.find("Total tracked epoch contexts:"), std::string::npos);
  EXPECT_NE(dump.find("EpochContext objects:"), std::string::npos);
  EXPECT_NE(dump.find("associated epoch contexts:"), std::string::npos);
  EXPECT_NE(dump.find("(reader)"), std::string::npos);

  sched.schedule(write_value(value.write(), 7));
  sched.run_all();

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const after_completion_dump = testing::internal::GetCapturedStderr();
  EXPECT_NE(after_completion_dump.find("Total tracked tasks: 0"), std::string::npos);
}

TEST(TaskRegistryDebugTest, GraphvizDotShowsSuspendedTaskAndEpochReadEdges)
{
  DebugScheduler sched;
  Async<int> value;

  sched.schedule(wait_for_reader(value.read()));
  sched.run();

  auto const dot = TaskRegistry::graphviz_dot();
  EXPECT_NE(dot.find("digraph uni20_async_dag"), std::string::npos);
  EXPECT_NE(dot.find("rankdir=LR"), std::string::npos);
  EXPECT_NE(dot.find("task_"), std::string::npos);
  EXPECT_NE(dot.find("epoch_"), std::string::npos);
  EXPECT_NE(dot.find("suspended"), std::string::npos);
  EXPECT_NE(dot.find("await read"), std::string::npos);
#if UNI20_DEBUG_DAG
  EXPECT_NE(dot.find("data_"), std::string::npos);
  EXPECT_NE(dot.find("co_await read"), std::string::npos);
#endif

  sched.schedule(write_value(value.write(), 7));
  sched.run_all();
}

#if UNI20_DEBUG_DAG
TEST(TaskRegistryDebugTest, GraphvizDotShowsCoarseBufferArgumentDependencies)
{
  DebugScheduler sched;
  Async<int> input(3);
  Async<int> output;

  auto task = copy_value(input.read(), output.write());
  auto const dot = TaskRegistry::graphviz_dot();

  EXPECT_NE(dot.find("data_"), std::string::npos);
  EXPECT_NE(dot.find("arg read"), std::string::npos);
  EXPECT_NE(dot.find("arg write"), std::string::npos);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(output.read().get_wait(sched), 3);
}

TEST(TaskRegistryDebugTest, GraphvizDotShowsOptionalDebugLabels)
{
  DebugScheduler sched;
  Async<int> input(3);
  Async<int> output;
  input.debug_name("named input");
  output.debug_name("named output");

  auto task = copy_value(input.read(), output.write());
  task.debug_name("copy kernel");

  auto const dot = TaskRegistry::graphviz_dot();
  EXPECT_NE(dot.find("named input"), std::string::npos);
  EXPECT_NE(dot.find("named output"), std::string::npos);
  EXPECT_NE(dot.find("copy kernel"), std::string::npos);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(output.read().get_wait(sched), 3);
}

TEST(TaskRegistryDebugTest, GraphSnapshotCarriesOptionalStacktraceProvenance)
{
  DebugScheduler sched;
  Async<int> value;

  auto task = wait_for_reader(value.read());
  task.debug_name("provenance reader");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const task_it = std::find_if(snapshot.tasks.begin(), snapshot.tasks.end(),
                                    [](auto const& record) { return record.label == "provenance reader"; });

  ASSERT_NE(task_it, snapshot.tasks.end());
  EXPECT_EQ(task_it->state, "suspended");
  ASSERT_FALSE(task_it->await_dependencies.empty());

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("provenance reader"), std::string::npos);
  EXPECT_NE(dot.find("co_await read"), std::string::npos);

#if UNI20_HAS_STACKTRACE
  EXPECT_FALSE(task_it->creation_site.stacktrace.empty());
  EXPECT_FALSE(task_it->schedule_site.stacktrace.empty());
  EXPECT_FALSE(task_it->last_transition_site.stacktrace.empty());
  EXPECT_FALSE(task_it->last_await_site.stacktrace.empty());
  EXPECT_FALSE(task_it->await_dependencies.front().await_site.stacktrace.empty());
  EXPECT_NE(dot.find("tooltip=\""), std::string::npos);
  if (!task_it->creation_site.location.empty())
  {
    EXPECT_NE(dot.find("created_at="), std::string::npos);
  }
  if (!task_it->schedule_site.location.empty())
  {
    EXPECT_NE(dot.find("scheduled_at="), std::string::npos);
  }
  if (!task_it->last_await_site.location.empty())
  {
    EXPECT_NE(dot.find("awaiting_at="), std::string::npos);
  }
#else
  EXPECT_TRUE(task_it->creation_site.stacktrace.empty());
  EXPECT_TRUE(task_it->schedule_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_transition_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_await_site.stacktrace.empty());
  EXPECT_TRUE(task_it->await_dependencies.front().await_site.stacktrace.empty());
  EXPECT_EQ(dot.find("created_at="), std::string::npos);
  EXPECT_EQ(dot.find("scheduled_at="), std::string::npos);
  EXPECT_EQ(dot.find("awaiting_at="), std::string::npos);
#endif

  sched.schedule(write_value(value.write(), 7));
  sched.run_all();
}

#if UNI20_HAS_STACKTRACE
TEST(TaskRegistryDebugTest, StacktraceOptionsSuppressSnapshotStacktraceText)
{
  StacktraceOptionsGuard guard;
  auto options = TaskRegistry::stacktrace_options();
  options.max_frames = 0;
  TaskRegistry::set_stacktrace_options(options);

  DebugScheduler sched;
  Async<int> value;

  auto task = wait_for_reader(value.read());
  task.debug_name("hidden stacktrace reader");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const task_it = std::find_if(snapshot.tasks.begin(), snapshot.tasks.end(),
                                    [](auto const& record) { return record.label == "hidden stacktrace reader"; });

  ASSERT_NE(task_it, snapshot.tasks.end());
  EXPECT_TRUE(task_it->creation_site.stacktrace.empty());
  EXPECT_TRUE(task_it->schedule_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_transition_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_await_site.stacktrace.empty());
  ASSERT_FALSE(task_it->await_dependencies.empty());
  EXPECT_TRUE(task_it->await_dependencies.front().await_site.stacktrace.empty());

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_EQ(dot.find("stacktrace:"), std::string::npos);

  sched.schedule(write_value(value.write(), 8));
  sched.run_all();
}
#endif

TEST(TaskRegistryDebugTest, GraphSnapshotCapturesAllAwaiterDependencies)
{
  DebugScheduler sched;
  Async<int> lhs;
  Async<int> rhs(6);
  Async<int> output;

  auto task = sum_all_values(lhs.read(), rhs.read(), output.write());
  task.debug_name("all awaiter kernel");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const task_it = std::find_if(snapshot.tasks.begin(), snapshot.tasks.end(),
                                    [](auto const& record) { return record.label == "all awaiter kernel"; });

  ASSERT_NE(task_it, snapshot.tasks.end());
  auto const read_await_count =
      std::count_if(task_it->await_dependencies.begin(), task_it->await_dependencies.end(),
                    [](auto const& dependency) { return dependency.role == TaskRegistryGraphRole::Reader; });
  EXPECT_GE(read_await_count, 2);

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("all awaiter kernel"), std::string::npos);
  EXPECT_NE(dot.find("co_await read"), std::string::npos);

  sched.schedule(write_value(lhs.write(), 4));
  sched.run_all();
  EXPECT_EQ(output.read().get_wait(sched), 10);
}

TEST(TaskRegistryDebugTest, GraphSnapshotExposesStructuredRecords)
{
  DebugScheduler sched;
  Async<int> input(3);
  Async<int> output;
  input.debug_name("structured input");
  output.debug_name("structured output");

  auto task = copy_value(input.read(), output.write());
  task.debug_name("structured copy");

  auto const snapshot = TaskRegistry::snapshot();
  auto const diagnostics = TaskRegistry::diagnose_snapshot(snapshot);
  auto const find_named_data = [&](std::string_view label) {
    return std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                        [&](auto const& node) { return node.label == label; });
  };
  auto const has_named_task = std::any_of(snapshot.tasks.begin(), snapshot.tasks.end(), [](auto const& record) {
    return record.label == "structured copy" && record.state == "suspended" && !record.read_dependencies.empty() &&
           !record.write_dependencies.empty();
  });
  auto const input_node = find_named_data("structured input");
  auto const output_node = find_named_data("structured output");

  EXPECT_TRUE(snapshot.snapshot_available);
  EXPECT_TRUE(diagnostics.notes.empty());
  ASSERT_NE(input_node, snapshot.data_nodes.end());
  ASSERT_NE(output_node, snapshot.data_nodes.end());
  EXPECT_FALSE(input_node->type.empty());
  EXPECT_FALSE(input_node->storage_address.empty());
  EXPECT_FALSE(input_node->address.empty());
  EXPECT_EQ(input_node->state, "constructed");
  EXPECT_EQ(input_node->value, "3");
  EXPECT_TRUE(input_node->value_constructed);
  EXPECT_FALSE(output_node->type.empty());
  EXPECT_FALSE(output_node->storage_address.empty());
  EXPECT_EQ(output_node->address, "(unconstructed)");
  EXPECT_EQ(output_node->state, "unconstructed");
  EXPECT_TRUE(output_node->value.empty());
  EXPECT_FALSE(output_node->value_constructed);
  EXPECT_TRUE(has_named_task);

  auto const dot = TaskRegistry::graphviz_dot(snapshot, diagnostics);
  EXPECT_NE(dot.find("structured input"), std::string::npos);
  EXPECT_NE(dot.find("structured output"), std::string::npos);
  EXPECT_NE(dot.find("structured copy"), std::string::npos);
  EXPECT_NE(dot.find("storage=0x"), std::string::npos);
  EXPECT_NE(dot.find("state=unconstructed"), std::string::npos);
  EXPECT_NE(dot.find("value=3"), std::string::npos);
  EXPECT_EQ(dot.find("value=unconstructed"), std::string::npos);
  EXPECT_EQ(dot.find("value=0x"), std::string::npos);
  EXPECT_EQ(dot.find("addr=0x0"), std::string::npos);

  sched.schedule(std::move(task));
  sched.run_all();

  auto const after_snapshot = TaskRegistry::snapshot();
  auto const after_output_node = std::find_if(after_snapshot.data_nodes.begin(), after_snapshot.data_nodes.end(),
                                              [](auto const& node) { return node.label == "structured output"; });
  ASSERT_NE(after_output_node, after_snapshot.data_nodes.end());
  EXPECT_TRUE(after_output_node->value_constructed);
  EXPECT_NE(after_output_node->address, "(unconstructed)");
  EXPECT_EQ(after_output_node->state, "constructed");
  EXPECT_EQ(after_output_node->value, "3");

  EXPECT_EQ(output.read().get_wait(sched), 3);
}

TEST(TaskRegistryDebugTest, GraphSnapshotSummarizesShapeLikeValues)
{
  Async<ShapeLikeValue> tensor(ShapeLikeValue{});
  tensor.debug_name("shape-like value");

  auto const snapshot = TaskRegistry::snapshot();
  auto const tensor_node = std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                                        [](auto const& node) { return node.label == "shape-like value"; });

  ASSERT_NE(tensor_node, snapshot.data_nodes.end());
  EXPECT_EQ(tensor_node->state, "constructed");
  EXPECT_EQ(tensor_node->value, "shape=(2, 3, 4)");

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("shape-like value"), std::string::npos);
  EXPECT_NE(dot.find("state=constructed"), std::string::npos);
  EXPECT_NE(dot.find("value=shape=(2, 3, 4)"), std::string::npos);
}

TEST(TaskRegistryDebugTest, GraphSnapshotUsesCustomDebugValueHook)
{
  Async<CustomDebugValue> value(CustomDebugValue{7});
  value.debug_name("custom value");

  auto const snapshot = TaskRegistry::snapshot();
  auto const value_node = std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                                       [](auto const& node) { return node.label == "custom value"; });

  ASSERT_NE(value_node, snapshot.data_nodes.end());
  EXPECT_EQ(value_node->state, "constructed");
  EXPECT_EQ(value_node->value, "custom id=7");

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("custom value"), std::string::npos);
  EXPECT_NE(dot.find("value=custom id=7"), std::string::npos);
}

TEST(TaskRegistryDebugTest, GraphSnapshotOmitsValueForOpaqueConstructedTypes)
{
  Async<OpaqueValue> value(OpaqueValue{});
  value.debug_name("opaque value");

  auto const snapshot = TaskRegistry::snapshot();
  auto const value_node = std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                                       [](auto const& node) { return node.label == "opaque value"; });

  ASSERT_NE(value_node, snapshot.data_nodes.end());
  EXPECT_EQ(value_node->state, "constructed");
  EXPECT_TRUE(value_node->value.empty());

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("opaque value"), std::string::npos);
  EXPECT_NE(dot.find("state=constructed"), std::string::npos);
  EXPECT_EQ(dot.find("value=constructed"), std::string::npos);
}

TEST(TaskRegistryDebugTest, FormatsEveryConfiguredRealScalar)
{
  auto check = []<typename Scalar>() {
    EXPECT_EQ(uni20::async::detail::scalar_debug_info(Scalar{5} / Scalar{4}), "1.25");
  };

  uni20::visit_scalar_precision(uni20::ScalarPrecision::fp32, check);
  uni20::visit_scalar_precision(uni20::ScalarPrecision::fp64, check);
  if (uni20::has_float128)
  {
    uni20::visit_scalar_precision(uni20::ScalarPrecision::fp128, check);
  }
}

TEST(TaskRegistryDebugTest, GraphvizDotDiagnosesMissingWriter)
{
  DebugScheduler sched;
  Async<int> value;
  value.debug_name("unwritten value");

  auto task = wait_for_reader(value.read());
  task.debug_name("blocked reader");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const diagnostics = TaskRegistry::diagnose_snapshot(snapshot);
  EXPECT_EQ(diagnostics.blocked_read_task_ids.size(), 1U);
  EXPECT_EQ(diagnostics.missing_writer_epoch_ids.size(), 1U);
  EXPECT_EQ(diagnostics.missing_writer_node_ids.size(), 1U);

  auto const dot_from_snapshot = TaskRegistry::graphviz_dot(snapshot, diagnostics);
  EXPECT_NE(dot_from_snapshot.find("DAG diagnostics"), std::string::npos);
  EXPECT_NE(dot_from_snapshot.find("missing writer"), std::string::npos);

  auto const dot = TaskRegistry::graphviz_dot();
  EXPECT_NE(dot.find("DAG diagnostics"), std::string::npos);
  EXPECT_NE(dot.find("blocked tasks: 1"), std::string::npos);
  EXPECT_NE(dot.find("missing writer"), std::string::npos);
  EXPECT_NE(dot.find("blocked: read"), std::string::npos);
  EXPECT_NE(dot.find("unwritten value"), std::string::npos);
  EXPECT_NE(dot.find("blocked reader"), std::string::npos);

  sched.schedule(write_value(value.write(), 5));
  sched.run_all();
  EXPECT_EQ(value.read().get_wait(sched), 5);
}

#endif

TEST(TaskRegistryDebugTest, GraphvizDumpFileWritesDot)
{
  auto dir = make_temp_dir("uni20-dag-file-test");
  TaskRegistry::GraphvizDumpOptions options;
  options.output_dir = dir.string();
  options.file_prefix = "registry";
  auto const path = TaskRegistry::default_graphviz_dump_path(options);

  EXPECT_TRUE(TaskRegistry::dump_graphviz_file(path));
  EXPECT_NE(read_file(path).find("digraph uni20_async_dag"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DefaultGraphvizDumpOptionsReadEnvironment)
{
  auto dir = make_temp_dir("uni20-dag-default-options-test");
  EnvVarGuard output_dir("UNI20_DEBUG_DAG_OUTPUT_DIR", dir.string());
  EnvVarGuard prefix("UNI20_DEBUG_DAG_FILE_PREFIX", "env-default");

  auto const options = TaskRegistry::default_graphviz_dump_options();
  EXPECT_EQ(options.output_dir, dir.string());
  EXPECT_EQ(options.file_prefix, "env-default");

  auto const path = TaskRegistry::default_graphviz_dump_path();
  EXPECT_NE(path.find(dir.string()), std::string::npos);
  EXPECT_NE(path.find("env-default"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DefaultStacktraceOptionsReadEnvironment)
{
  EnvVarGuard frames("UNI20_DEBUG_DAG_STACKTRACE_FRAMES", "3");
  EnvVarGuard internal_frames("UNI20_DEBUG_DAG_STACKTRACE_INTERNAL_FRAMES", "false");

  auto const options = TaskRegistry::default_stacktrace_options();
  EXPECT_EQ(options.max_frames, 3U);
  EXPECT_FALSE(options.include_internal_frames);
}

TEST(TaskRegistryDebugTest, DefaultStacktraceOptionsAcceptUnlimitedFrames)
{
  EnvVarGuard frames("UNI20_DEBUG_DAG_STACKTRACE_FRAMES", "all");

  auto const options = TaskRegistry::default_stacktrace_options();
  EXPECT_EQ(options.max_frames, std::numeric_limits<std::size_t>::max());
}

TEST(TaskRegistryDebugTest, ResetStacktraceOptionsReadsEnvironment)
{
  StacktraceOptionsGuard guard;
  EnvVarGuard frames("UNI20_DEBUG_DAG_STACKTRACE_FRAMES", "0");

  TaskRegistry::reset_stacktrace_options();
  EXPECT_EQ(TaskRegistry::stacktrace_options().max_frames, 0U);
}

TEST(TaskRegistryDebugTest, RequestedGraphvizDumpIsServicedByCaller)
{
  auto dir = make_temp_dir("uni20-dag-request-test");
  TaskRegistry::GraphvizDumpOptions options;
  options.output_dir = dir.string();
  options.file_prefix = "requested";

  TaskRegistry::request_graphviz_dump();
  EXPECT_TRUE(TaskRegistry::service_debug_requests(options));
  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(200)));
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DebugSchedulerServicesQueuedGraphvizDumpRequests)
{
  auto dir = make_temp_dir("uni20-dag-scheduler-request-test");
  EnvVarGuard output_dir("UNI20_DEBUG_DAG_OUTPUT_DIR", dir.string());
  EnvVarGuard prefix("UNI20_DEBUG_DAG_FILE_PREFIX", "scheduler");

  TaskRegistry::request_graphviz_dump();
  DebugScheduler sched;
  sched.run_all();

  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(200)));
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DiagnosticsServicePollsControlFile)
{
  TaskRegistry::stop_diagnostics_service();
  auto dir = make_temp_dir("uni20-dag-control-test");
  auto request_file = dir / "dump.request";

  TaskRegistry::DiagnosticsServiceOptions options;
  options.dump_options.output_dir = dir.string();
  options.dump_options.file_prefix = "control";
  options.request_file = request_file.string();
  options.poll_interval_ms = 10;

  ASSERT_TRUE(TaskRegistry::start_diagnostics_service(options));
  {
    std::ofstream request(request_file);
    request << "dump\n";
  }
  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(1000)));
  TaskRegistry::stop_diagnostics_service();
  std::filesystem::remove_all(dir);
}

#if defined(__linux__)
TEST(TaskRegistryDebugTest, DiagnosticsServiceHandlesSignalTrigger)
{
  TaskRegistry::stop_diagnostics_service();
  auto dir = make_temp_dir("uni20-dag-signal-test");

  TaskRegistry::DiagnosticsServiceOptions options;
  options.dump_options.output_dir = dir.string();
  options.dump_options.file_prefix = "signal";
  options.signal_number = SIGUSR1;
  options.poll_interval_ms = 50;

  ASSERT_TRUE(TaskRegistry::start_diagnostics_service(options));
  ASSERT_EQ(::kill(::getpid(), SIGUSR1), 0);
  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(1000)));
  TaskRegistry::stop_diagnostics_service();
  std::filesystem::remove_all(dir);
}
#endif
#endif
