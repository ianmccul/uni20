#pragma once

/**
 * \file cusolver_error.hpp
 * \ingroup backend_cusolver
 * \brief Structured cuSOLVER failures and checked-call helpers.
 */

#include <uni20/common/diagnostic_error.hpp>

#include <cusolverDn.h>

#include <source_location>
#include <string>
#include <string_view>

namespace uni20::cusolver
{

/// \brief Return a stable symbolic name for a cuSOLVER status value.
[[nodiscard]] std::string_view status_name(cusolverStatus_t status) noexcept;

/// \brief Exception carrying a failed cuSOLVER operation and device identity.
class CusolverError : public uni20::diagnostic_error {
  public:
    CusolverError(cusolverStatus_t status, std::string operation, int device);

    [[nodiscard]] cusolverStatus_t status() const noexcept { return status_; }
    [[nodiscard]] std::string const& operation() const noexcept { return operation_; }
    [[nodiscard]] std::string_view error_name() const noexcept { return status_name(status_); }
    [[nodiscard]] int device() const noexcept { return device_; }

  private:
    cusolverStatus_t status_;
    std::string operation_;
    int device_;
};

/// \brief Raise a structured cuSOLVER provider failure.
[[noreturn]] void raise_error(cusolverStatus_t status, std::string_view operation, int device,
                              std::source_location where = std::source_location::current());

/// \brief Raise unless a cuSOLVER call returned `CUSOLVER_STATUS_SUCCESS`.
void check(cusolverStatus_t status, std::string_view operation, int device,
           std::source_location where = std::source_location::current());

} // namespace uni20::cusolver
