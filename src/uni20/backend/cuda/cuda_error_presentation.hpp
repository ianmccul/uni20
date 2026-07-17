#pragma once

/**
 * \file cuda_error_presentation.hpp
 * \ingroup backend_cuda
 * \brief Presentation adapter for structured CUDA runtime failures.
 */

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/common/presentation.hpp>

#include <fmt/core.h>

namespace uni20::cuda
{

/// \brief Build a presentation report for a CUDA runtime failure.
inline presentation::report_builder diagnostic_report(CudaRuntimeError const& error)
{
  presentation::report_builder report;
  report.status(presentation::semantic_glyph::failure, fmt::format("CUDA operation '{}' failed", error.operation()));
  report.field("Status", fmt::format("{} ({})", error.error_name(), static_cast<int>(error.status())));
  report.field("Reason", error.reason());
  if (error.device().has_value())
  {
    report.field("Device", *error.device());
  }
  return report;
}

} // namespace uni20::cuda
