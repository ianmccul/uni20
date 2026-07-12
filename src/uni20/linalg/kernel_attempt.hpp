#pragma once

/**
 * \file kernel_attempt.hpp
 * \ingroup linalg
 * \brief Allocation-free runtime results for kernel backend attempts.
 */

#include <string_view>

namespace uni20::linalg
{

/// \brief Result of one runtime kernel backend attempt.
/// \details Every value other than `success` is a clean decline: the backend
///          preserved all arguments and produced no externally visible side
///          effect. Terminal execution failures are reported through the
///          operation's ordinary error or exception mechanism instead.
enum class KernelAttempt
{
  success,
  unsupported_instance,
  unsupported_shape,
  unsupported_layout,
  unsupported_accessor,
  unsupported_transform,
  unavailable,
  insufficient_resources
};

/// \brief Return whether a backend performed the operation.
constexpr bool kernel_attempt_succeeded(KernelAttempt attempt) noexcept { return attempt == KernelAttempt::success; }

/// \brief Return a stable human-readable kernel-attempt result name.
constexpr std::string_view kernel_attempt_name(KernelAttempt attempt) noexcept
{
  switch (attempt)
  {
    case KernelAttempt::success:
      return "success";
    case KernelAttempt::unsupported_instance:
      return "unsupported runtime instance";
    case KernelAttempt::unsupported_shape:
      return "unsupported shape";
    case KernelAttempt::unsupported_layout:
      return "unsupported layout";
    case KernelAttempt::unsupported_accessor:
      return "unsupported accessor";
    case KernelAttempt::unsupported_transform:
      return "unsupported transform";
    case KernelAttempt::unavailable:
      return "unavailable";
    case KernelAttempt::insufficient_resources:
      return "insufficient resources";
  }
  return "unknown kernel attempt";
}

} // namespace uni20::linalg
