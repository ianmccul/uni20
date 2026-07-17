#pragma once

/**
 * \file cuda_error.hpp
 * \ingroup backend_cuda
 * \brief Structured CUDA runtime failures and checked-call helpers.
 */

#include <cuda_runtime_api.h>

#include <uni20/common/diagnostic_error.hpp>

#include <optional>
#include <source_location>
#include <string>
#include <string_view>

namespace uni20::cuda
{

/// \brief Exception carrying a failed CUDA runtime operation and status.
class CudaRuntimeError : public uni20::diagnostic_error {
  public:
    /// \brief Construct a CUDA runtime error with optional device context.
    CudaRuntimeError(cudaError_t status, std::string operation, std::optional<int> device = std::nullopt);

    /// \brief Return the CUDA runtime status code.
    [[nodiscard]] cudaError_t status() const noexcept { return status_; }

    /// \brief Return the CUDA runtime operation that failed.
    [[nodiscard]] std::string const& operation() const noexcept { return operation_; }

    /// \brief Return the CUDA symbolic error name.
    [[nodiscard]] std::string const& error_name() const noexcept { return error_name_; }

    /// \brief Return the CUDA runtime error description.
    [[nodiscard]] std::string const& reason() const noexcept { return reason_; }

    /// \brief Return the associated CUDA device ordinal, when known.
    [[nodiscard]] std::optional<int> device() const noexcept { return device_; }

  private:
    cudaError_t status_;
    std::string operation_;
    std::string error_name_;
    std::string reason_;
    std::optional<int> device_;
};

/// \brief Raise a structured CUDA runtime failure through Uni20's error boundary.
[[noreturn]] void raise_runtime_error(cudaError_t status, std::string_view operation,
                                      std::optional<int> device = std::nullopt,
                                      std::source_location where = std::source_location::current());

/// \brief Raise when a CUDA runtime operation did not return `cudaSuccess`.
void check(cudaError_t status, std::string_view operation, std::optional<int> device = std::nullopt,
           std::source_location where = std::source_location::current());

} // namespace uni20::cuda
