#pragma once

/**
 * \file dispatch_diagnostics.hpp
 * \ingroup linalg
 * \brief Opt-in structured diagnostics for kernel-dispatch walks.
 */

#include <uni20/linalg/dispatch_error.hpp>

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace uni20::linalg::dispatch_diagnostics
{

/// \brief Structured result of one ordered kernel-dispatch walk.
struct event
{
    std::string operation;
    std::vector<KernelBackendAttempt> backend_attempts;

    /// \brief Return whether one backend completed the operation.
    [[nodiscard]] bool succeeded() const noexcept
    {
      for (auto const& attempt : backend_attempts)
      {
        if (attempt.runtime_result && kernel_attempt_succeeded(*attempt.runtime_result)) return true;
      }
      return false;
    }

    /// \brief Return the selected backend name, if the dispatch succeeded.
    [[nodiscard]] std::optional<std::string_view> selected_backend() const noexcept
    {
      for (auto const& attempt : backend_attempts)
      {
        if (attempt.runtime_result && kernel_attempt_succeeded(*attempt.runtime_result)) return attempt.backend;
      }
      return std::nullopt;
    }
};

/// \brief Callable receiving structured dispatch events after backend walks.
using sink = std::function<void(event const&)>;

namespace detail
{
extern std::atomic<bool> enabled_flag;
}

/// \brief Return whether runtime kernel-dispatch diagnostics are enabled.
/// \details Disabled dispatches pay one relaxed atomic flag check and do not
///          construct diagnostic records, allocate, lock, or invoke a sink.
[[nodiscard]] inline bool enabled() noexcept { return detail::enabled_flag.load(std::memory_order_relaxed); }

/// \brief Replace the process-wide kernel-dispatch diagnostic sink.
/// \details Passing an empty callable disables diagnostics.
void set_sink(sink replacement);

/// \brief Disable runtime kernel-dispatch diagnostics.
void reset_sink();

/// \brief Return a copy of the active diagnostic sink.
[[nodiscard]] sink current_sink();

/// \brief Temporarily replace the diagnostic sink and restore it on destruction.
class scoped_sink {
  public:
    explicit scoped_sink(sink replacement);
    scoped_sink(scoped_sink const&) = delete;
    scoped_sink& operator=(scoped_sink const&) = delete;
    scoped_sink(scoped_sink&& other) noexcept;
    scoped_sink& operator=(scoped_sink&& other) noexcept;
    ~scoped_sink();

  private:
    sink previous_;
    bool active_ = true;
};

namespace detail
{
void emit(event const& diagnostic);
} // namespace detail

} // namespace uni20::linalg::dispatch_diagnostics
