#pragma once

/**
 * \file dispatch_error_presentation.hpp
 * \ingroup linalg
 * \brief Presentation adapter for structured kernel-dispatch failures.
 */

#include <uni20/common/presentation.hpp>
#include <uni20/linalg/dispatch_error.hpp>

#include <fmt/core.h>

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
} // namespace detail

/// \brief Build a presentation report for a structured kernel-dispatch error.
inline presentation::report_builder diagnostic_report(KernelDispatchError const& error)
{
  presentation::report_builder report;
  report.status(presentation::semantic_glyph::failure,
                fmt::format("Kernel dispatch failed for '{}'", error.operation()));

  auto& table = report.table(std::string(detail::unavailable_backend_heading(error.failure())));
  table.column("Backend", presentation::table_alignment::left)
      .column("Type acceptance", presentation::table_alignment::left)
      .column("Runtime result", presentation::table_alignment::left);

  for (auto const& attempt : error.backend_attempts())
  {
    table.row(attempt.backend, detail::kernel_type_acceptance_name(attempt.type_acceptance),
              attempt.runtime_result ? kernel_attempt_name(*attempt.runtime_result) : "not eligible");
  }
  return report;
}

} // namespace uni20::linalg
