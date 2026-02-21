#include "async/async_task.hpp"
#include "async/task_registry.hpp"
#include "config.hpp"
#include <gtest/gtest.h>
#include <string>

using namespace uni20;
using namespace uni20::async;

namespace
{

AsyncTask make_suspended_task() { co_return; }

} // namespace

#if UNI20_DEBUG_ASYNC_TASKS
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
#endif
