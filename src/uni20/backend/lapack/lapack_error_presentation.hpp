#pragma once

/**
 * \file lapack_error_presentation.hpp
 * \ingroup backend_lapack
 * \brief Presentation adapter for structured checked-LAPACK failures.
 */

#include <uni20/backend/lapack/lapack_error.hpp>
#include <uni20/common/presentation.hpp>

#include <fmt/core.h>

namespace uni20::lapack
{

/// \brief Build a presentation report for a terminal LAPACK provider failure.
inline presentation::report_builder diagnostic_report(LapackError const& error)
{
  presentation::report_builder report;
  report.status(presentation::semantic_glyph::failure, fmt::format("LAPACK '{}' failed", error.routine()));
  report.field("INFO", error.info());
  report.field("Reason", error.reason());
  return report;
}

} // namespace uni20::lapack
