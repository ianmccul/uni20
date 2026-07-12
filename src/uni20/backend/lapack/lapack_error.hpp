#pragma once

/**
 * \file lapack_error.hpp
 * \ingroup backend_lapack
 * \brief Structured checked-LAPACK failure data.
 */

#include <uni20/common/diagnostic_error.hpp>
#include <uni20/config.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace uni20::lapack
{

namespace detail
{
inline std::string lapack_error_message(std::string_view routine, std::string_view reason, blas_int info)
{
  return "LAPACK " + std::string(routine) + " " + std::string(reason) + " (INFO=" + std::to_string(info) + ")";
}
} // namespace detail

/// \brief Exception carrying a terminal positive LAPACK `INFO` result.
class LapackError : public uni20::diagnostic_error {
  public:
    /// \brief Construct an error from the provider routine, status, and interpreted reason.
    LapackError(std::string routine, blas_int info, std::string reason)
        : diagnostic_error(detail::lapack_error_message(routine, reason, info)), routine_(std::move(routine)),
          info_(info), reason_(std::move(reason))
    {}

    /// \brief Return the LAPACK routine name without a scalar prefix.
    [[nodiscard]] std::string const& routine() const noexcept { return routine_; }

    /// \brief Return the positive provider `INFO` value.
    [[nodiscard]] blas_int info() const noexcept { return info_; }

    /// \brief Return Uni20's interpretation of the provider failure.
    [[nodiscard]] std::string const& reason() const noexcept { return reason_; }

  private:
    std::string routine_;
    blas_int info_;
    std::string reason_;
};

} // namespace uni20::lapack
