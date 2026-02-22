#include "async/async.hpp"
#include "async/async_task.hpp"
#include "async/buffers.hpp"
#include "async/debug_scheduler.hpp"
#include "async/task_registry.hpp"
#include "config.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

using namespace uni20;
using namespace uni20::async;

namespace
{

AsyncTask make_suspended_task() { co_return; }

AsyncTask wait_for_reader(ReadBuffer<int> reader)
{
  auto const& value = co_await reader;
  (void)value;
}

AsyncTask write_value(WriteBuffer<int> writer, int value)
{
  co_await writer.emplace(value);
  co_return;
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

#if UNI20_DEBUG_ASYNC_TASKS
TEST(TaskRegistryDebugTest, DumpModeDefaultsToBasicWhenUnset)
{
  EXPECT_EXIT({ std::_Exit(dump_mode_probe(nullptr)); }, ::testing::ExitedWithCode(1), "");
}

TEST(TaskRegistryDebugTest, DumpModeParsesNoneSynonyms)
{
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("0")); }, ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("off")); }, ::testing::ExitedWithCode(0), "");
}

TEST(TaskRegistryDebugTest, DumpModeParsesFullSynonyms)
{
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("2")); }, ::testing::ExitedWithCode(2), "");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("verbose")); }, ::testing::ExitedWithCode(2), "");
}

TEST(TaskRegistryDebugTest, DumpModeTrimsAndNormalizesCase)
{
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("  YeS  ")); }, ::testing::ExitedWithCode(1), "");
}

TEST(TaskRegistryDebugTest, DumpModeFallsBackToBasicForUnknownValue)
{
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
#endif
