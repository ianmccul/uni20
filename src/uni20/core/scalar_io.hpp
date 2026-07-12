#pragma once

#include "numeric_limits.hpp"
#include "scalar_concepts.hpp"
#include "types.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fmt/format.h>
#include <istream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace uni20
{

/// \brief Selects how real scalar values are rendered as text.
/// \ingroup core_math
enum class real_format_notation
{
  general,
  fixed,
  scientific
};

/// \brief Formatting options for Uni20 real and complex scalars.
/// \ingroup core_math
struct scalar_format_options
{
    int precision = -1;
    real_format_notation notation = real_format_notation::general;
    bool normalize_negative_zero = true;
    std::string_view imaginary_unit = "i";
};

namespace detail
{
template <typename T> [[nodiscard]] int effective_precision(scalar_format_options const& options)
{
  if (options.precision >= 0)
  {
    return options.precision;
  }
  return uni20::numeric_limits<std::remove_cv_t<T>>::max_digits10;
}

[[nodiscard]] inline char printf_notation(real_format_notation notation)
{
  switch (notation)
  {
    case real_format_notation::fixed:
      return 'f';
    case real_format_notation::scientific:
      return 'e';
    case real_format_notation::general:
      return 'g';
  }
  return 'g';
}

template <typename T> [[nodiscard]] bool scalar_signbit(T value)
{
  return std::signbit(static_cast<long double>(value));
}

template <typename T> [[nodiscard]] std::string fmt_real(T value, scalar_format_options const& options)
{
  int const precision = effective_precision<T>(options);
  switch (options.notation)
  {
    case real_format_notation::fixed:
      return fmt::format("{:.{}f}", value, precision);
    case real_format_notation::scientific:
      return fmt::format("{:.{}e}", value, precision);
    case real_format_notation::general:
      return fmt::format("{:.{}g}", value, precision);
  }
  return fmt::format("{}", value);
}

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
[[nodiscard]] inline std::string format_float128(uni20::float128 value, scalar_format_options const& options)
{
  int const precision = effective_precision<uni20::float128>(options);
  char const notation = printf_notation(options.notation);
  std::string const format = fmt::format("%.{}{}", precision, notation);

#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_FLOAT128)
  char fixed_buffer[128] = {};
  int const fixed_count = ::strfromf128(fixed_buffer, sizeof(fixed_buffer), format.c_str(), value);
  if (fixed_count < 0)
  {
    throw std::runtime_error("failed to format uni20::float128 value");
  }
  if (static_cast<std::size_t>(fixed_count) < sizeof(fixed_buffer))
  {
    return std::string(fixed_buffer, static_cast<std::size_t>(fixed_count));
  }

  std::string dynamic_buffer(static_cast<std::size_t>(fixed_count) + 1, '\0');
  int const dynamic_count = ::strfromf128(dynamic_buffer.data(), dynamic_buffer.size(), format.c_str(), value);
  if (dynamic_count < 0)
  {
    throw std::runtime_error("failed to format uni20::float128 value");
  }
  dynamic_buffer.resize(static_cast<std::size_t>(dynamic_count));
  return dynamic_buffer;
#else
  return fmt::format("{}", static_cast<long double>(value));
#endif
}

[[nodiscard]] inline uni20::float128 parse_float128(std::string_view text)
{
  std::string buffer(text);
  char* end = nullptr;
  errno = 0;
#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_FLOAT128)
  uni20::float128 const parsed = ::strtof128(buffer.c_str(), &end);
#else
  long double const parsed = std::strtold(buffer.c_str(), &end);
#endif
  if (end == buffer.c_str() || *end != '\0')
  {
    throw std::invalid_argument(fmt::format("invalid float128 value: {}", text));
  }
  if (errno == ERANGE)
  {
    throw std::out_of_range(fmt::format("float128 value out of range: {}", text));
  }
  return static_cast<uni20::float128>(parsed);
}
#endif

template <typename T> [[nodiscard]] T parse_builtin_real(std::string_view text)
{
  std::string buffer(text);
  char* end = nullptr;
  errno = 0;

  if constexpr (std::same_as<T, float>)
  {
    float const parsed = std::strtof(buffer.c_str(), &end);
    if (end == buffer.c_str() || *end != '\0')
    {
      throw std::invalid_argument(fmt::format("invalid real value: {}", text));
    }
    if (errno == ERANGE)
    {
      throw std::out_of_range(fmt::format("real value out of range: {}", text));
    }
    return parsed;
  }
  else if constexpr (std::same_as<T, double>)
  {
    double const parsed = std::strtod(buffer.c_str(), &end);
    if (end == buffer.c_str() || *end != '\0')
    {
      throw std::invalid_argument(fmt::format("invalid real value: {}", text));
    }
    if (errno == ERANGE)
    {
      throw std::out_of_range(fmt::format("real value out of range: {}", text));
    }
    return parsed;
  }
  else
  {
    long double const parsed = std::strtold(buffer.c_str(), &end);
    if (end == buffer.c_str() || *end != '\0')
    {
      throw std::invalid_argument(fmt::format("invalid real value: {}", text));
    }
    if (errno == ERANGE)
    {
      throw std::out_of_range(fmt::format("real value out of range: {}", text));
    }
    return static_cast<T>(parsed);
  }
}
} // namespace detail

/// \brief Format a Uni20 real scalar.
/// \tparam T Real scalar type.
/// \param value Value to format.
/// \param options Text formatting options.
/// \return Formatted scalar text.
/// \ingroup core_math
template <Real T> [[nodiscard]] std::string format_real(T value, scalar_format_options const& options = {})
{
  using real_type = std::remove_cv_t<T>;
  if (options.normalize_negative_zero && value == real_type{})
  {
    value = real_type{};
  }

#if UNI20_HAS_FLOAT128
  if constexpr (std::same_as<real_type, uni20::float128>)
  {
    return detail::format_float128(value, options);
  }
  else
#endif
  {
    return detail::fmt_real(value, options);
  }
}

/// \brief Format a Uni20 complex scalar as `real+imagi`.
/// \tparam T Complex scalar type.
/// \param value Value to format.
/// \param options Text formatting options.
/// \return Formatted scalar text.
/// \ingroup core_math
template <Complex T> [[nodiscard]] std::string format_complex(T const& value, scalar_format_options const& options = {})
{
  using real_type = make_real_t<T>;
  real_type imag = value.imag();
  bool const negative_imag = detail::scalar_signbit(imag) && !(options.normalize_negative_zero && imag == real_type{});
  if (negative_imag)
  {
    imag = -imag;
  }
  else if (options.normalize_negative_zero && imag == real_type{})
  {
    imag = real_type{};
  }

  return format_real(value.real(), options) + (negative_imag ? "-" : "+") + format_real(imag, options) +
         std::string(options.imaginary_unit);
}

/// \brief Format any Uni20 real or complex scalar.
/// \tparam T Real or complex scalar type.
/// \param value Value to format.
/// \param options Text formatting options.
/// \return Formatted scalar text.
/// \ingroup core_math
template <RealOrComplex T>
[[nodiscard]] std::string format_scalar(T const& value, scalar_format_options const& options = {})
{
  if constexpr (Real<T>)
  {
    return format_real(value, options);
  }
  else
  {
    return format_complex(value, options);
  }
}

/// \brief Parse a Uni20 real scalar from text.
/// \tparam T Real scalar type.
/// \param text Text containing one complete real literal.
/// \return Parsed value.
/// \throws std::invalid_argument if the text is not a complete real literal.
/// \throws std::out_of_range if the literal overflows or underflows the target type.
/// \ingroup core_math
template <Real T> [[nodiscard]] T parse_real(std::string_view text)
{
  using real_type = std::remove_cv_t<T>;
#if UNI20_HAS_FLOAT128
  if constexpr (std::same_as<real_type, uni20::float128>)
  {
    return detail::parse_float128(text);
  }
  else
#endif
  {
    return detail::parse_builtin_real<real_type>(text);
  }
}

/// \brief Read a Uni20 real scalar from a stream token.
/// \details This helper exists because extension real aliases such as
///          `uni20::float128` cannot portably receive ordinary overloaded stream
///          extraction operators.
/// \tparam T Real scalar type.
/// \param stream Input stream.
/// \param value Destination value.
/// \return The input stream.
/// \ingroup core_math
template <Real T> std::istream& read_real(std::istream& stream, T& value)
{
  std::string token;
  stream >> token;
  if (!stream)
  {
    return stream;
  }

  try
  {
    value = parse_real<T>(token);
  }
  catch (...)
  {
    stream.setstate(std::ios::failbit);
  }
  return stream;
}

} // namespace uni20

#if UNI20_HAS_FLOAT128
template <> struct fmt::formatter<uni20::float128>
{
    constexpr auto parse(format_parse_context& ctx)
    {
      auto it = ctx.begin();
      if (it != ctx.end() && *it != '}')
      {
        throw fmt::format_error("uni20::float128 formatter currently supports only default '{}'");
      }
      return it;
    }

    template <typename FormatContext> auto format(uni20::float128 value, FormatContext& ctx) const
    {
      return fmt::format_to(ctx.out(), "{}", uni20::format_real(value));
    }
};

template <> struct fmt::formatter<uni20::complex<uni20::float128>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
      auto it = ctx.begin();
      if (it != ctx.end() && *it != '}')
      {
        throw fmt::format_error("uni20::complex<uni20::float128> formatter currently supports only default '{}'");
      }
      return it;
    }

    template <typename FormatContext>
    auto format(uni20::complex<uni20::float128> const& value, FormatContext& ctx) const
    {
      return fmt::format_to(ctx.out(), "{}", uni20::format_complex(value));
    }
};
#endif
