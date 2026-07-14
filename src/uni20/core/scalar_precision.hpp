#pragma once

/**
 * \file scalar_precision.hpp
 * \ingroup core_math
 * \brief Runtime selection of configured Uni20 real scalar precisions.
 */

#include "types.hpp"

#include <uni20/config.hpp>

#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace uni20
{

/// \brief Runtime identifier for the real precision underlying a scalar type.
enum class ScalarPrecision
{
  fp32,
  fp64,
  fp128
};

/// \brief Whether this build provides a concrete `uni20::float128` type.
inline constexpr bool has_float128 = UNI20_HAS_FLOAT128 != 0;

/// \brief Return the canonical command-line and diagnostic spelling of a precision.
/// \param precision Precision to name.
/// \return Stable lowercase precision name.
[[nodiscard]] constexpr std::string_view scalar_precision_name(ScalarPrecision precision)
{
  switch (precision)
  {
    case ScalarPrecision::fp32:
      return "fp32";
    case ScalarPrecision::fp64:
      return "fp64";
    case ScalarPrecision::fp128:
      return "fp128";
  }
  return "unknown";
}

/// \brief Parse a canonical scalar precision name independently of build availability.
/// \param name Name to parse.
/// \return Parsed precision, or `std::nullopt` for an unknown name.
[[nodiscard]] constexpr std::optional<ScalarPrecision> parse_scalar_precision(std::string_view name)
{
  if (name == "fp32") return ScalarPrecision::fp32;
  if (name == "fp64") return ScalarPrecision::fp64;
  if (name == "fp128") return ScalarPrecision::fp128;
  return std::nullopt;
}

/// \brief Report whether a runtime precision has a concrete scalar type in this build.
/// \param precision Precision to query.
/// \return `true` when `visit_scalar_precision` can instantiate that scalar type.
[[nodiscard]] constexpr bool scalar_precision_is_available(ScalarPrecision precision)
{
  switch (precision)
  {
    case ScalarPrecision::fp32:
    case ScalarPrecision::fp64:
      return true;
    case ScalarPrecision::fp128:
      return has_float128;
  }
  return false;
}

namespace detail
{
#if UNI20_HAS_FLOAT128
inline constexpr std::array configured_scalar_precisions{ScalarPrecision::fp32, ScalarPrecision::fp64,
                                                         ScalarPrecision::fp128};
#else
inline constexpr std::array configured_scalar_precisions{ScalarPrecision::fp32, ScalarPrecision::fp64};
#endif
} // namespace detail

/// \brief Return the real scalar precisions configured in this build.
/// \return Stable, ascending sequence of available precisions.
[[nodiscard]] constexpr std::span<ScalarPrecision const> configured_scalar_precisions()
{
  return detail::configured_scalar_precisions;
}

/// \brief Return a compact choice list suitable for command-line help.
/// \return Pipe-separated configured precision names.
[[nodiscard]] constexpr std::string_view configured_scalar_precision_choices()
{
#if UNI20_HAS_FLOAT128
  return "fp32|fp64|fp128";
#else
  return "fp32|fp64";
#endif
}

/// \brief Invoke a templated callable with the concrete real scalar for a runtime precision.
/// \details The callable must provide `operator()<Scalar>()` with a common return type for every configured
///          precision. Requesting a known but unavailable precision throws `std::invalid_argument`. The conditional
///          `uni20::float128` type is confined to this configuration boundary, so callers do not need preprocessor
///          guards.
/// \tparam Function Templated callable type.
/// \param precision Runtime precision selection.
/// \param function Callable invoked with the selected scalar as its template argument.
/// \return Result returned by the selected callable specialization.
template <class Function> decltype(auto) visit_scalar_precision(ScalarPrecision precision, Function&& function)
{
  switch (precision)
  {
    case ScalarPrecision::fp32:
      return std::forward<Function>(function).template operator()<float32>();
    case ScalarPrecision::fp64:
      return std::forward<Function>(function).template operator()<float64>();
    case ScalarPrecision::fp128:
#if UNI20_HAS_FLOAT128
      return std::forward<Function>(function).template operator()<float128>();
#else
      throw std::invalid_argument("fp128 is not available in this Uni20 build");
#endif
  }
  throw std::invalid_argument("invalid Uni20 scalar precision");
}

} // namespace uni20
