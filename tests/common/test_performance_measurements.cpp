#include <uni20/common/performance_measurements.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>
#include <type_traits>

namespace
{

enum class TestEvent
{
  first,
  second,
  count,
};

using TestMeasurements =
    uni20::performance::DurationMeasurements<TestEvent, static_cast<std::size_t>(TestEvent::count)>;
using TestDetailedMeasurements =
    uni20::performance::DetailedMeasurements<TestEvent, static_cast<std::size_t>(TestEvent::count)>;

static_assert(std::is_empty_v<uni20::performance::NoMeasurements>);
static_assert(uni20::performance::measurement_level_v<uni20::performance::NoMeasurements> ==
              uni20::performance::MeasurementLevel::none);
static_assert(uni20::performance::measurement_level_v<TestMeasurements> ==
              uni20::performance::MeasurementLevel::coarse);
static_assert(uni20::performance::measurement_level_v<TestDetailedMeasurements> ==
              uni20::performance::MeasurementLevel::detailed);

TEST(PerformanceMeasurementsTest, DisabledPolicyPreservesResultsAndReferences)
{
  uni20::performance::NoMeasurements measurements;
  int value = 7;

  auto&& reference =
      uni20::performance::measure_duration(measurements, TestEvent::first, [&]() -> int& { return value; });

  static_assert(std::is_same_v<decltype(reference), int&>);
  EXPECT_EQ(&reference, &value);
}

TEST(PerformanceMeasurementsTest, AggregatesExplicitDurations)
{
  using namespace std::chrono_literals;
  TestMeasurements measurements;

  measurements.record_duration(TestEvent::first, 4ns);
  measurements.record_duration(TestEvent::first, 8ns);
  measurements.record_duration(TestEvent::first, 6ns);

  auto const& statistics = measurements[TestEvent::first];
  EXPECT_EQ(statistics.count, 3U);
  EXPECT_EQ(statistics.total, 18ns);
  EXPECT_EQ(statistics.minimum, 4ns);
  EXPECT_EQ(statistics.maximum, 8ns);
  EXPECT_EQ(statistics.mean(), 6ns);
  EXPECT_EQ(measurements[TestEvent::second].mean(), 0ns);
}

TEST(PerformanceMeasurementsTest, MeasuresSuccessfulAndThrowingCalls)
{
  TestMeasurements measurements;

  EXPECT_EQ(uni20::performance::measure_duration(measurements, TestEvent::first, [] { return 11; }), 11);
  EXPECT_THROW(uni20::performance::measure_duration(measurements, TestEvent::second,
                                                    [] { throw std::runtime_error("expected"); }),
               std::runtime_error);

  EXPECT_EQ(measurements[TestEvent::first].count, 1U);
  EXPECT_EQ(measurements[TestEvent::second].count, 1U);
}

TEST(PerformanceMeasurementsTest, ResetsStatistics)
{
  using namespace std::chrono_literals;
  TestMeasurements measurements;
  measurements.record_duration(TestEvent::first, 5ns);

  measurements.reset();
  EXPECT_EQ(measurements[TestEvent::first].count, 0U);
  EXPECT_EQ(measurements[TestEvent::first].total, 0ns);
}

TEST(PerformanceMeasurementsTest, DisabledBatchInvokesExecutorDirectly)
{
  uni20::performance::NoMeasurements measurements;
  std::size_t calls = 0;
  auto executor = [&](auto& function) {
    for (std::size_t index = 0; index < 3; ++index)
      function(index);
  };

  uni20::performance::measure_batch(measurements, TestEvent::first, 3, executor,
                                    [&](std::size_t index) { calls += index + 1; });

  EXPECT_EQ(calls, 6U);
}

TEST(PerformanceMeasurementsTest, DetailedBatchRecordsItemsAndRethrows)
{
  TestDetailedMeasurements measurements;
  auto executor = [&](auto& function) {
    for (std::size_t index = 0; index < 4; ++index)
      function(index);
  };

  EXPECT_THROW(uni20::performance::measure_batch(measurements, TestEvent::first, 4, executor,
                                                 [](std::size_t index) {
                                                   if (index == 2) throw std::runtime_error("expected");
                                                 }),
               std::runtime_error);

  ASSERT_EQ(measurements.batches(TestEvent::first).size(), 1U);
  auto const& batch = measurements.batches(TestEvent::first).front();
  EXPECT_EQ(batch.requested_items, 4U);
  EXPECT_EQ(batch.started_items, 3U);
  EXPECT_EQ(batch.completed_items, 3U);
  EXPECT_EQ(batch.peak_concurrency, 1U);
  EXPECT_EQ(measurements[TestEvent::first].count, 1U);

  measurements.reset();
  EXPECT_TRUE(measurements.batches(TestEvent::first).empty());
  EXPECT_EQ(measurements[TestEvent::first].count, 0U);
}

} // namespace
