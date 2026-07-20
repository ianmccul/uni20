#pragma once

/**
 * \file cublas_error_presentation.hpp
 * \ingroup backend_cublas
 * \brief Presentation adapter for structured cuBLAS failures.
 */

#include <uni20/backend/cublas/cublas_error.hpp>
#include <uni20/common/presentation.hpp>

#include <fmt/core.h>

namespace uni20::cublas
{

/// \brief Build a presentation report for a cuBLAS provider failure.
inline presentation::report_builder diagnostic_report(CublasError const& error)
{
  presentation::report_builder report;
  report.status(presentation::semantic_glyph::failure, fmt::format("cuBLAS '{}' failed", error.operation()));
  report.field("Status", error.error_name());
  report.field("Device", error.device());
  return report;
}

} // namespace uni20::cublas
