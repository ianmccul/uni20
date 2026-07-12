#pragma once

/**
 * \file dispatch_error_presentation.hpp
 * \ingroup linalg
 * \brief Presentation adapter for structured kernel-dispatch failures.
 */

#include <uni20/common/presentation.hpp>
#include <uni20/linalg/dispatch_diagnostics.hpp>
#include <uni20/linalg/dispatch_error.hpp>

#include <fmt/core.h>
#include <string>
#include <utility>
#include <vector>

namespace uni20::linalg
{

namespace detail
{
inline std::string_view unavailable_backend_heading(KernelDispatchFailure failure) noexcept
{
  switch (failure)
  {
    case KernelDispatchFailure::no_eligible_backend:
      return "No available backend accepts these argument types";
    case KernelDispatchFailure::all_candidates_declined:
      return "No available backend accepted this runtime instance";
  }
  return "No available backend performed the operation";
}

inline std::string_view backend_runtime_result_name(KernelBackendAttempt const& attempt) noexcept
{
  if (attempt.runtime_result) return kernel_attempt_name(*attempt.runtime_result);
  if (attempt.type_acceptance == KernelTypeAcceptance::no) return "not eligible";
  return "not attempted";
}

inline void append_backend_attempts(presentation::report_builder& report, std::string title,
                                    std::vector<KernelBackendAttempt> const& attempts)
{
  auto& table = report.table(std::move(title));
  table.column("Backend", presentation::table_alignment::left)
      .column("Type acceptance", presentation::table_alignment::left)
      .column("Runtime result", presentation::table_alignment::left);

  for (auto const& attempt : attempts)
  {
    table.row(attempt.backend, kernel_type_acceptance_name(attempt.type_acceptance),
              backend_runtime_result_name(attempt));
  }
}
} // namespace detail

/// \brief Build a presentation report for one opt-in kernel-dispatch diagnostic event.
inline presentation::report_builder diagnostic_report(dispatch_diagnostics::event const& diagnostic)
{
  presentation::report_builder report;
  if (auto const selected = diagnostic.selected_backend())
  {
    report.status(presentation::semantic_glyph::success,
                  fmt::format("Kernel dispatch for '{}' selected '{}'", diagnostic.operation, *selected));
  }
  else
  {
    report.status(presentation::semantic_glyph::failure,
                  fmt::format("Kernel dispatch for '{}' did not select a backend", diagnostic.operation));
  }

  detail::append_backend_attempts(report, "Ordered backend candidates", diagnostic.backend_attempts);
  return report;
}

/// \brief Build a presentation report for a structured kernel-dispatch error.
inline presentation::report_builder diagnostic_report(KernelDispatchError const& error)
{
  presentation::report_builder report;
  report.status(presentation::semantic_glyph::failure,
                fmt::format("Kernel dispatch failed for '{}'", error.operation()));

  detail::append_backend_attempts(report, std::string(detail::unavailable_backend_heading(error.failure())),
                                  error.backend_attempts());
  return report;
}

} // namespace uni20::linalg
