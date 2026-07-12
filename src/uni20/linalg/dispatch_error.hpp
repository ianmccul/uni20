#pragma once

/**
 * \file dispatch_error.hpp
 * \ingroup linalg
 * \brief Structured kernel-dispatch failure data.
 */

#include <uni20/common/diagnostic_error.hpp>
#include <uni20/linalg/kernel_attempt.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::linalg
{

/// \brief Type-level backend eligibility for an operation and argument set.
enum class KernelTypeAcceptance
{
  no,
  maybe,
  yes
};

/// \brief Type-level result returned by `kernel_accepts_types` customizations.
template <KernelTypeAcceptance Value> using KernelAcceptance = std::integral_constant<KernelTypeAcceptance, Value>;

/// \brief Report that a backend rejects the argument types.
inline constexpr KernelAcceptance<KernelTypeAcceptance::no> kernel_types_no{};

/// \brief Report that runtime operand values determine backend acceptance.
inline constexpr KernelAcceptance<KernelTypeAcceptance::maybe> kernel_types_maybe{};

/// \brief Report that a backend accepts every instance of the argument types.
inline constexpr KernelAcceptance<KernelTypeAcceptance::yes> kernel_types_yes{};

/// \brief Classify why an ordered kernel-dispatch walk failed.
enum class KernelDispatchFailure
{
  no_eligible_backend,
  all_candidates_declined
};

/// \brief Describe one backend's role in a failed dispatch walk.
struct KernelBackendAttempt
{
    std::string backend;
    KernelTypeAcceptance type_acceptance = KernelTypeAcceptance::no;
    std::optional<KernelAttempt> runtime_result = std::nullopt;
};

namespace detail
{
inline std::string_view kernel_dispatch_failure_name(KernelDispatchFailure failure) noexcept
{
  switch (failure)
  {
    case KernelDispatchFailure::no_eligible_backend:
      return "no backend accepts these argument types";
    case KernelDispatchFailure::all_candidates_declined:
      return "all runtime candidates declined";
  }
  return "unknown dispatch failure";
}

inline std::string_view kernel_type_acceptance_name(KernelTypeAcceptance acceptance) noexcept
{
  switch (acceptance)
  {
    case KernelTypeAcceptance::no:
      return "no";
    case KernelTypeAcceptance::maybe:
      return "maybe";
    case KernelTypeAcceptance::yes:
      return "yes";
  }
  return "unknown";
}

inline std::string kernel_dispatch_error_message(std::string_view operation, KernelDispatchFailure failure)
{
  return "kernel dispatch failed for operation '" + std::string(operation) +
         "': " + std::string(kernel_dispatch_failure_name(failure));
}
} // namespace detail

/// \brief Exception carrying structured information about a failed backend walk.
class KernelDispatchError : public uni20::diagnostic_error {
  public:
    /// \brief Construct a dispatch failure from its operation and backend attempts.
    KernelDispatchError(std::string operation, KernelDispatchFailure failure,
                        std::vector<KernelBackendAttempt> backend_attempts)
        : diagnostic_error(detail::kernel_dispatch_error_message(operation, failure)), operation_(std::move(operation)),
          failure_(failure), backend_attempts_(std::move(backend_attempts))
    {}

    /// \brief Return the stable diagnostic operation name.
    [[nodiscard]] std::string const& operation() const noexcept { return operation_; }

    /// \brief Return the dispatch failure category.
    [[nodiscard]] KernelDispatchFailure failure() const noexcept { return failure_; }

    /// \brief Return the ordered backend acceptance and attempt records.
    [[nodiscard]] std::vector<KernelBackendAttempt> const& backend_attempts() const noexcept
    {
      return backend_attempts_;
    }

  private:
    std::string operation_;
    KernelDispatchFailure failure_;
    std::vector<KernelBackendAttempt> backend_attempts_;
};

} // namespace uni20::linalg
