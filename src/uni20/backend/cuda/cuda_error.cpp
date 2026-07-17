#include <uni20/backend/cuda/cuda_error.hpp>

#include <uni20/backend/cuda/cuda_error_presentation.hpp>
#include <uni20/common/trace.hpp>

#include <fmt/core.h>

#include <utility>

namespace uni20::cuda
{

namespace
{

std::string make_message(cudaError_t status, std::string const& operation, std::optional<int> device)
{
  char const* const name = cudaGetErrorName(status);
  char const* const reason = cudaGetErrorString(status);
  if (device.has_value())
  {
    return fmt::format("CUDA operation '{}' failed on device {}: {} ({})", operation, *device,
                       reason == nullptr ? "unknown CUDA error" : reason, name == nullptr ? "unknown" : name);
  }
  return fmt::format("CUDA operation '{}' failed: {} ({})", operation,
                     reason == nullptr ? "unknown CUDA error" : reason, name == nullptr ? "unknown" : name);
}

} // namespace

CudaRuntimeError::CudaRuntimeError(cudaError_t status, std::string operation, std::optional<int> device)
    : diagnostic_error(make_message(status, operation, device)), status_(status), operation_(std::move(operation)),
      error_name_(cudaGetErrorName(status) == nullptr ? "unknown" : cudaGetErrorName(status)),
      reason_(cudaGetErrorString(status) == nullptr ? "unknown CUDA error" : cudaGetErrorString(status)),
      device_(device)
{}

[[noreturn]] void raise_runtime_error(cudaError_t status, std::string_view operation, std::optional<int> device,
                                      std::source_location where)
{
  trace::raise(CudaRuntimeError(status, std::string(operation), device), where);
}

void check(cudaError_t status, std::string_view operation, std::optional<int> device, std::source_location where)
{
  if (status != cudaSuccess)
  {
    raise_runtime_error(status, operation, device, where);
  }
}

} // namespace uni20::cuda
