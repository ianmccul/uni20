#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/buffers.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/config.hpp>

#if defined(__linux__)
#include <signal.h>
#include <unistd.h>
#endif

using namespace uni20;
using namespace uni20::async;

#if UNI20_DEBUG_ASYNC_TASKS
namespace
{

class EnvVarGuard {
  public:
    EnvVarGuard(char const* name, std::string value) : name_(name)
    {
      if (auto const* old_value = std::getenv(name_.c_str())) old_value_ = old_value;
      ::setenv(name_.c_str(), value.c_str(), 1);
    }

    ~EnvVarGuard()
    {
      if (old_value_)
        ::setenv(name_.c_str(), old_value_->c_str(), 1);
      else
        ::unsetenv(name_.c_str());
    }

  private:
    std::string name_;
    std::optional<std::string> old_value_;
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
