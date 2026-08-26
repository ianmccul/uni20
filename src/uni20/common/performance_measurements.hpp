/**
 * \file performance_measurements.hpp
 * \ingroup common
 * \brief Compile-time-gated performance duration measurements.
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::performance
{

/// \brief Selects how much performance instrumentation an operation instantiates.
enum class MeasurementLevel
{
  none,
  coarse,
  detailed,
};

/// \brief Explicit disabled measurement policy used by ordinary operation overloads.
/// \details Algorithms must select the disabled path with `if constexpr`. Merely
///          accepting a nullable recorder would leave a branch in the hot path.
struct NoMeasurements
{
    static constexpr MeasurementLevel measurement_level = MeasurementLevel::none;
};

/// \brief Duration type shared by performance measurement collectors.
using Duration = std::chrono::nanoseconds;

/// \brief Clock used for host wall-duration measurements.
using WallClock = std::chrono::steady_clock;

/// \brief Aggregate duration statistics for repeated instances of one event.
struct DurationStatistics
{
    /// \brief Number of recorded event instances.
    std::size_t count = 0;
    /// \brief Sum of all recorded durations.
    Duration total = Duration::zero();
    /// \brief Smallest recorded duration, or zero when no instance was recorded.
    Duration minimum = Duration::zero();
    /// \brief Largest recorded duration, or zero when no instance was recorded.
    Duration maximum = Duration::zero();

    /// \brief Return the arithmetic mean duration.
    /// \return Zero when no instance has been recorded; otherwise `total / count`.
    [[nodiscard]] constexpr auto mean() const noexcept -> Duration
    {
      return count == 0 ? Duration::zero() : total / static_cast<Duration::rep>(count);
    }
};

/// \brief Fixed-size coarse wall-duration collector indexed by an event enum.
/// \details The collector is intentionally not thread-safe. Coarse instrumentation
///          records operation-scale phase boundaries from their calling thread.
///          Detailed concurrent work should first record into disjoint per-item
///          slots and aggregate after the batch joins.
/// \tparam Event Enum or enum-like event identifier convertible to `std::size_t`.
/// \tparam EventCount Number of valid event identifiers.
template <class Event, std::size_t EventCount> class DurationMeasurements {
  public:
    static constexpr MeasurementLevel measurement_level = MeasurementLevel::coarse;
    static constexpr std::size_t event_count = EventCount;

    /// \brief Record one completed event duration.
    /// \pre `event` identifies an element of `[0, EventCount)`.
    /// \param event Event identifier.
    /// \param duration Nonnegative elapsed duration to aggregate.
    void record_duration(Event event, Duration duration) noexcept
    {
      auto& statistics = values_[event_index(event)];
      if (statistics.count == 0)
      {
        statistics.minimum = duration;
        statistics.maximum = duration;
      }
      else
      {
        statistics.minimum = std::min(statistics.minimum, duration);
        statistics.maximum = std::max(statistics.maximum, duration);
      }
      ++statistics.count;
      statistics.total += duration;
    }

    /// \brief Return aggregate statistics for one event.
    /// \pre `event` identifies an element of `[0, EventCount)`.
    /// \param event Event identifier.
    /// \return Immutable aggregate statistics.
    [[nodiscard]] auto operator[](Event event) const noexcept -> DurationStatistics const&
    {
      return values_[event_index(event)];
    }

    /// \brief Clear every recorded statistic.
    constexpr void reset() noexcept { values_ = {}; }

  private:
    [[nodiscard]] static constexpr auto event_index(Event event) noexcept -> std::size_t
    {
      return static_cast<std::size_t>(event);
    }

    std::array<DurationStatistics, EventCount> values_{};
};

/// \brief Detailed timing summary for one synchronous lightweight batch.
struct BatchMeasurement
{
    /// \brief Number of items requested from the scheduler.
    std::size_t requested_items = 0;
    /// \brief Number of items whose invocation began.
    std::size_t started_items = 0;
    /// \brief Number of started item invocations which exited, including by exception.
    std::size_t completed_items = 0;
    /// \brief Largest number of measured item intervals overlapping in time.
    std::size_t peak_concurrency = 0;
    /// \brief Duration from entry to the batch executor until its return or exception.
    Duration wall_duration = Duration::zero();
    /// \brief Sum of completed item durations.
    Duration total_item_duration = Duration::zero();
    /// \brief Largest completed item duration.
    Duration maximum_item_duration = Duration::zero();
    /// \brief Delay from batch entry until the first item began.
    Duration first_item_start_delay = Duration::zero();
    /// \brief Difference between the first and last item start times.
    Duration item_start_spread = Duration::zero();
    /// \brief Difference between the first and last item finish times.
    Duration item_finish_spread = Duration::zero();
    /// \brief Delay from the last item finish until the batch executor returned.
    Duration return_after_last_finish = Duration::zero();
};

/// \brief Fixed-size detailed collector retaining every measured batch.
/// \details Duration aggregates use the base coarse interface. Batch records
///          are intentionally retained individually for offline load-balance
///          analysis. This collector is an explicit high-overhead policy.
/// \tparam Event Enum or enum-like event identifier convertible to `std::size_t`.
/// \tparam EventCount Number of valid event identifiers.
template <class Event, std::size_t EventCount>
class DetailedMeasurements : public DurationMeasurements<Event, EventCount> {
  public:
    static constexpr MeasurementLevel measurement_level = MeasurementLevel::detailed;

    /// \brief Retain one completed synchronous batch measurement.
    /// \pre `event` identifies an element of `[0, EventCount)`.
    /// \param event Event identifier.
    /// \param measurement Completed batch timing and work summary.
    void record_batch(Event event, BatchMeasurement measurement)
    {
      batches_[static_cast<std::size_t>(event)].push_back(std::move(measurement));
    }

    /// \brief Return detailed batch records for one event.
    /// \pre `event` identifies an element of `[0, EventCount)`.
    /// \param event Event identifier.
    /// \return Immutable batch records in execution order.
    [[nodiscard]] auto batches(Event event) const noexcept -> std::span<BatchMeasurement const>
    {
      return batches_[static_cast<std::size_t>(event)];
    }

    /// \brief Clear duration aggregates and retained batch measurements.
    void reset()
    {
      DurationMeasurements<Event, EventCount>::reset();
      batches_ = {};
    }

  private:
    std::array<std::vector<BatchMeasurement>, EventCount> batches_{};
};

/// \brief Compile-time instrumentation level declared by a measurement policy.
template <class Measurements>
inline constexpr MeasurementLevel measurement_level_v = std::remove_cvref_t<Measurements>::measurement_level;

/// \brief Measurement policy that either disables instrumentation or records event durations.
template <class Measurements, class Event>
concept DurationMeasurementPolicy = requires { std::remove_cvref_t<Measurements>::measurement_level; } &&
                                    (measurement_level_v<Measurements> == MeasurementLevel::none ||
                                     requires(Measurements& measurements, Event event, Duration duration) {
                                       { measurements.record_duration(event, duration) } noexcept -> std::same_as<void>;
                                     });

/// \brief Measurement policy compatible with detailed synchronous batch records.
template <class Measurements, class Event>
concept BatchMeasurementPolicy = DurationMeasurementPolicy<Measurements, Event> &&
                                 (measurement_level_v<Measurements> != MeasurementLevel::detailed ||
                                  requires(Measurements& measurements, Event event, BatchMeasurement batch) {
                                    { measurements.record_batch(event, std::move(batch)) } -> std::same_as<void>;
                                  });

namespace detail
{

template <class Measurements, class Event> class DurationGuard {
  public:
    DurationGuard(Measurements& measurements, Event event) noexcept
        : measurements_(measurements), event_(event), start_(WallClock::now())
    {}

    DurationGuard(DurationGuard const&) = delete;
    DurationGuard(DurationGuard&&) = delete;
    auto operator=(DurationGuard const&) -> DurationGuard& = delete;
    auto operator=(DurationGuard&&) -> DurationGuard& = delete;

    ~DurationGuard() noexcept
    {
      auto const elapsed = std::chrono::duration_cast<Duration>(WallClock::now() - start_);
      measurements_.record_duration(event_, elapsed);
    }

  private:
    Measurements& measurements_;
    Event event_;
    WallClock::time_point start_;
};

} // namespace detail

/// \brief Invoke a function with compile-time-gated wall-duration measurement.
/// \details `NoMeasurements` instantiates only the direct invocation. Any other
///          compatible policy reads the wall clock on entry and exit and records
///          the elapsed duration, including when the function throws.
/// \tparam Measurements Measurement policy type.
/// \tparam Event Event identifier type.
/// \tparam Function Nullary callable type.
/// \param measurements Explicit measurement policy or collector.
/// \param event Event to record for an enabled collector.
/// \param function Callable whose inclusive wall duration is measured.
/// \return The callable result, preserving references and value category.
template <class Measurements, class Event, class Function>
  requires DurationMeasurementPolicy<Measurements, Event> && std::invocable<Function>
decltype(auto) measure_duration(Measurements& measurements, Event event, Function&& function)
{
  if constexpr (measurement_level_v<Measurements> == MeasurementLevel::none)
  {
    static_cast<void>(measurements);
    static_cast<void>(event);
    return std::invoke(std::forward<Function>(function));
  }
  else
  {
    detail::DurationGuard guard(measurements, event);
    return std::invoke(std::forward<Function>(function));
  }
}

namespace detail
{

struct BatchItemInterval
{
    bool started = false;
    bool completed = false;
    WallClock::time_point start = {};
    WallClock::time_point finish = {};
};

inline auto summarize_batch(std::span<BatchItemInterval const> items, WallClock::time_point batch_start,
                            WallClock::time_point batch_finish) -> BatchMeasurement
{
  BatchMeasurement result{.requested_items = items.size(),
                          .wall_duration = std::chrono::duration_cast<Duration>(batch_finish - batch_start)};
  std::vector<std::pair<WallClock::time_point, int>> boundaries;
  boundaries.reserve(items.size() * 2);
  std::optional<WallClock::time_point> first_start;
  std::optional<WallClock::time_point> last_start;
  std::optional<WallClock::time_point> first_finish;
  std::optional<WallClock::time_point> last_finish;
  for (auto const& item : items)
  {
    if (!item.started) continue;
    ++result.started_items;
    first_start = first_start ? std::min(*first_start, item.start) : item.start;
    last_start = last_start ? std::max(*last_start, item.start) : item.start;
    if (!item.completed) continue;
    ++result.completed_items;
    first_finish = first_finish ? std::min(*first_finish, item.finish) : item.finish;
    last_finish = last_finish ? std::max(*last_finish, item.finish) : item.finish;
    auto const duration = std::chrono::duration_cast<Duration>(item.finish - item.start);
    result.total_item_duration += duration;
    result.maximum_item_duration = std::max(result.maximum_item_duration, duration);
    boundaries.emplace_back(item.start, 1);
    boundaries.emplace_back(item.finish, -1);
  }

  if (first_start)
  {
    result.first_item_start_delay = std::chrono::duration_cast<Duration>(*first_start - batch_start);
    result.item_start_spread = std::chrono::duration_cast<Duration>(*last_start - *first_start);
  }
  if (first_finish)
  {
    result.item_finish_spread = std::chrono::duration_cast<Duration>(*last_finish - *first_finish);
    result.return_after_last_finish = std::chrono::duration_cast<Duration>(batch_finish - *last_finish);
  }

  std::ranges::sort(boundaries, [](auto const& lhs, auto const& rhs) {
    if (lhs.first != rhs.first) return lhs.first < rhs.first;
    return lhs.second > rhs.second;
  });
  std::size_t active = 0;
  for (auto const& [time, delta] : boundaries)
  {
    static_cast<void>(time);
    if (delta < 0)
      --active;
    else
      result.peak_concurrency = std::max(result.peak_concurrency, ++active);
  }
  return result;
}

} // namespace detail

/// \brief Execute a synchronous batch with compile-time-selected timing detail.
/// \details The disabled policy invokes `executor(function)` directly. Coarse
///          policies measure only the inclusive executor duration. Detailed
///          policies additionally read the clock around each item, write one
///          disjoint timing slot per index, and aggregate after the executor
///          joins. A batch record is retained before an executor exception is
///          rethrown.
/// \pre The executor invokes each started index at most once and does not return
///      until all item invocations have joined.
/// \tparam Measurements Measurement policy type.
/// \tparam Event Event identifier type.
/// \tparam Executor Callable accepting the wrapped indexed item function.
/// \tparam Function Callable accepting one index in `[0, size)`.
/// \param measurements Explicit measurement policy or collector.
/// \param event Event identifying this batch.
/// \param size Number of indexed work items.
/// \param executor Synchronous scheduler or serial-loop adapter.
/// \param function Work-item callable.
template <class Measurements, class Event, class Executor, class Function>
  requires BatchMeasurementPolicy<Measurements, Event> && std::invocable<Function&, std::size_t>
void measure_batch(Measurements& measurements, Event event, std::size_t size, Executor&& executor, Function&& function)
{
  if constexpr (measurement_level_v<Measurements> == MeasurementLevel::none)
  {
    static_cast<void>(measurements);
    static_cast<void>(event);
    std::invoke(std::forward<Executor>(executor), function);
  }
  else if constexpr (measurement_level_v<Measurements> == MeasurementLevel::coarse)
  {
    measure_duration(measurements, event, [&] { std::invoke(std::forward<Executor>(executor), function); });
  }
  else
  {
    std::vector<detail::BatchItemInterval> items(size);
    auto wrapped = [&](std::size_t index) {
      auto& item = items[index];
      item.started = true;
      item.start = WallClock::now();
      try
      {
        std::invoke(function, index);
      }
      catch (...)
      {
        item.finish = WallClock::now();
        item.completed = true;
        throw;
      }
      item.finish = WallClock::now();
      item.completed = true;
    };

    auto const start = WallClock::now();
    std::exception_ptr failure;
    try
    {
      std::invoke(std::forward<Executor>(executor), wrapped);
    }
    catch (...)
    {
      failure = std::current_exception();
    }
    auto const finish = WallClock::now();
    auto batch = detail::summarize_batch(items, start, finish);
    measurements.record_duration(event, batch.wall_duration);
    measurements.record_batch(event, std::move(batch));
    if (failure) std::rethrow_exception(failure);
  }
}

} // namespace uni20::performance
