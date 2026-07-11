#pragma once

/**
 * \file dispatch_error_presentation.hpp
 * \ingroup linalg
 * \brief Presentation adapter for structured kernel-dispatch failures.
 */

#include <uni20/common/presentation.hpp>
#include <uni20/linalg/dispatch_error.hpp>

namespace uni20::linalg
{

/// \brief Build a presentation report for a structured kernel-dispatch error.
inline presentation::report_builder diagnostic_report(KernelDispatchError const& error)
{
  presentation::report_builder report("Kernel dispatch failed");
  report.status(presentation::semantic_glyph::failure, "No backend performed the operation")
      .field("Operation", error.operation())
      .field("Reason", detail::kernel_dispatch_failure_name(error.failure()));

  auto& table = report.table("Backend candidates");
  table.column("Backend", presentation::table_alignment::left)
      .column("Type acceptance", presentation::table_alignment::left)
      .column("Runtime result", presentation::table_alignment::left);

  for (auto const& attempt : error.backend_attempts())
  {
    table.row(attempt.backend, detail::kernel_type_acceptance_name(attempt.type_acceptance),
              attempt.attempted ? "declined" : "not eligible");
  }
  return report;
}

} // namespace uni20::linalg
