#include "dispatch_diagnostics.hpp"

#include <atomic>
#include <mutex>
#include <utility>

namespace uni20::linalg::dispatch_diagnostics
{
namespace
{
std::mutex& sink_mutex()
{
  static std::mutex value;
  return value;
}

sink& sink_storage()
{
  static sink value;
  return value;
}

} // namespace

namespace detail
{
constinit std::atomic<bool> enabled_flag = false;
} // namespace detail

void set_sink(sink replacement)
{
  bool active = false;
  {
    std::lock_guard lock(sink_mutex());
    sink_storage() = std::move(replacement);
    active = static_cast<bool>(sink_storage());
  }
  detail::enabled_flag.store(active, std::memory_order_relaxed);
}

void reset_sink() { set_sink({}); }

sink current_sink()
{
  std::lock_guard lock(sink_mutex());
  return sink_storage();
}

scoped_sink::scoped_sink(sink replacement) : previous_(current_sink()) { set_sink(std::move(replacement)); }

scoped_sink::scoped_sink(scoped_sink&& other) noexcept
    : previous_(std::move(other.previous_)), active_(std::exchange(other.active_, false))
{}

scoped_sink& scoped_sink::operator=(scoped_sink&& other) noexcept
{
  if (this != &other)
  {
    if (active_) set_sink(std::move(previous_));
    previous_ = std::move(other.previous_);
    active_ = std::exchange(other.active_, false);
  }
  return *this;
}

scoped_sink::~scoped_sink()
{
  if (active_) set_sink(std::move(previous_));
}

namespace detail
{
void emit(event const& diagnostic)
{
  static thread_local bool emitting = false;
  if (emitting) return;

  auto destination = current_sink();
  if (!destination) return;

  emitting = true;
  try
  {
    destination(diagnostic);
  }
  catch (...)
  {
    emitting = false;
    throw;
  }
  emitting = false;
}
} // namespace detail

} // namespace uni20::linalg::dispatch_diagnostics
