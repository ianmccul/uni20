#include <uni20/backend/cublas/cublas_error.hpp>

#include <uni20/backend/cublas/cublas_error_presentation.hpp>
#include <uni20/common/trace.hpp>

#include <fmt/core.h>

#include <utility>

namespace uni20::cublas
{

std::string_view status_name(cublasStatus_t status) noexcept
{
  switch (status)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN";
}

CublasError::CublasError(cublasStatus_t status, std::string operation, int device)
    : diagnostic_error(
          fmt::format("cuBLAS operation '{}' failed on device {}: {}", operation, device, status_name(status))),
      status_(status), operation_(std::move(operation)), device_(device)
{}

[[noreturn]] void raise_error(cublasStatus_t status, std::string_view operation, int device, std::source_location where)
{
  trace::raise(CublasError(status, std::string(operation), device), where);
}

void check(cublasStatus_t status, std::string_view operation, int device, std::source_location where)
{
  if (status != CUBLAS_STATUS_SUCCESS) raise_error(status, operation, device, where);
}

} // namespace uni20::cublas
