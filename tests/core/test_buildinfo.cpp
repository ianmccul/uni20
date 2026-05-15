#include <uni20/buildinfo.hpp>

#include "gtest/gtest.h"

#include <algorithm>
#include <span>
#include <string_view>

namespace
{
bool contains_entry(std::span<uni20::build_info::Entry const> entries, std::string_view key)
{
  return std::ranges::any_of(entries, [key](auto const& entry) { return entry.key == key; });
}
} // namespace

TEST(BuildInfoTest, CurrentReportsCoreFields)
{
  auto const info = uni20::build_info::current();

  EXPECT_FALSE(info.generator.empty());
  EXPECT_FALSE(info.build_type.empty());
  EXPECT_FALSE(info.system_name.empty());
  EXPECT_FALSE(info.system_version.empty());
  EXPECT_FALSE(info.system_processor.empty());
  EXPECT_FALSE(info.cxx_compiler_id.empty());
  EXPECT_FALSE(info.cxx_compiler_version.empty());
  EXPECT_FALSE(info.cxx_compiler_path.empty());
  EXPECT_TRUE(info.generator == uni20::build_info::kGenerator);
}

TEST(BuildInfoTest, CurrentReportsUni20CacheEntries)
{
  auto const info = uni20::build_info::current();

  EXPECT_EQ(info.build_options.size(), uni20::build_info::kBuildOptions.size());
  EXPECT_EQ(info.detected_environment.size(), uni20::build_info::kDetectedEnvironment.size());
  EXPECT_FALSE(info.build_options.empty());
  EXPECT_TRUE(contains_entry(info.build_options, "UNI20_BUILD_TESTS"));
  EXPECT_TRUE(contains_entry(info.build_options, "UNI20_BUILD_PYTHON"));
}

TEST(BuildInfoTest, EntriesHaveKeys)
{
  auto const info = uni20::build_info::current();

  for (auto const& entry : info.build_options)
  {
    EXPECT_FALSE(entry.key.empty());
  }

  for (auto const& entry : info.detected_environment)
  {
    EXPECT_FALSE(entry.key.empty());
  }
}
