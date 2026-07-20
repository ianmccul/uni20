#pragma once

/**
 * \file cublas_error.hpp
 * \ingroup backend_cublas
 * \brief Structured cuBLAS failures and checked-call helpers.
 */

#include <uni20/common/diagnostic_error.hpp>

#include <cublas_v2.h>

#include <source_location>
#include <string>
#include <string_view>

namespace uni20::cublas
{

/// \brief Return a stable symbolic name for a cuBLAS status value.
[[nodiscard]] std::string_view status_name(cublasStatus_t status) noexcept;

/// \brief Exception carrying a failed cuBLAS operation and device identity.
class CublasError : public uni20::diagnostic_error {
  public:
    CublasError(cublasStatus_t status, std::string operation, int device);

    [[nodiscard]] cublasStatus_t status() const noexcept { return status_; }
    [[nodiscard]] std::string const& operation() const noexcept { return operation_; }
    [[nodiscard]] std::string_view error_name() const noexcept { return status_name(status_); }
    [[nodiscard]] int device() const noexcept { return device_; }

  private:
    cublasStatus_t status_;
    std::string operation_;
    int device_;
};

/// \brief Raise a structured cuBLAS provider failure.
[[noreturn]] void raise_error(cublasStatus_t status, std::string_view operation, int device,
                              std::source_location where = std::source_location::current());

/// \brief Raise unless a cuBLAS call returned `CUBLAS_STATUS_SUCCESS`.
void check(cublasStatus_t status, std::string_view operation, int device,
           std::source_location where = std::source_location::current());

} // namespace uni20::cublas
