#include <uni20/backend/cusolver/cusolver_error.hpp>

#include <uni20/common/trace.hpp>

#include <fmt/core.h>

#include <utility>

namespace uni20::cusolver
{

std::string_view status_name(cusolverStatus_t status) noexcept
{
  switch (status)
  {
    case CUSOLVER_STATUS_SUCCESS:
      return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
    case CUSOLVER_STATUS_INVALID_WORKSPACE:
      return "CUSOLVER_STATUS_INVALID_WORKSPACE";
    default:
      return "CUSOLVER_STATUS_UNKNOWN";
  }
}

CusolverError::CusolverError(cusolverStatus_t status, std::string operation, int device)
    : diagnostic_error(
          fmt::format("cuSOLVER operation '{}' failed on device {}: {}", operation, device, status_name(status))),
      status_(status), operation_(std::move(operation)), device_(device)
{}

[[noreturn]] void raise_error(cusolverStatus_t status, std::string_view operation, int device,
                              std::source_location where)
{
  trace::raise(CusolverError(status, std::string(operation), device), where);
}

void check(cusolverStatus_t status, std::string_view operation, int device, std::source_location where)
{
  if (status != CUSOLVER_STATUS_SUCCESS) raise_error(status, operation, device, where);
}

} // namespace uni20::cusolver
