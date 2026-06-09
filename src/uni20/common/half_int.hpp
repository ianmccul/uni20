/**
 * \file half_int.hpp
 * \brief Defines `basic_half_int<T>`, a signed integer type with half-integer resolution.
 */

#pragma once

#include <uni20/common/trace.hpp>

#include <charconv>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <ios>
#include <iosfwd>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace uni20
{

/// \brief Signed value type that stores multiples of one half exactly.
/// \tparam T Signed integer storage type used for the doubled representation.
template <std::signed_integral T> class basic_half_int {
  public:
    /// \brief Underlying signed storage type.
    using value_type = T;

    /// \brief Tag type used to construct directly from the doubled representation.
    struct twice_tag
    {};

    /// \brief Construct the value zero.
    constexpr basic_half_int() = default;

    /// \brief Construct from an integral value.
    /// \param value Integral value to represent exactly.
    template <std::integral U> constexpr basic_half_int(U value) : twice_(checked_double(static_cast<T>(value))) {}

    /// \brief Construct by rounding a floating-point value to the nearest half-integer.
    /// \param value Floating-point value to round.
    template <std::floating_point U> explicit basic_half_int(U value) : twice_(rounded_twice(value)) {}

    /// \brief Construct directly from the doubled representation.
    /// \param twice_value Stored doubled value.
    /// \param tag Selects the direct doubled-value constructor.
    constexpr basic_half_int(T twice_value, twice_tag tag) : twice_(twice_value)
    {
        static_cast<void>(tag);
    }

    /// \brief Return the doubled representation.
    /// \return Stored doubled value.
    constexpr auto twice() const -> T { return twice_; }

    /// \brief Return whether the represented value is integral.
    /// \return `true` if the value has no half-integer part.
    constexpr auto is_integral() const -> bool { return (twice_ & T{1}) == 0; }

    /// \brief Convert to `double`.
    /// \return Floating-point representation.
    constexpr auto to_double() const -> double { return static_cast<double>(twice_) / 2.0; }

    /// \brief Convert to the underlying integral type.
    /// \return Integral representation.
    /// \throws std::runtime_error If the value is not integral.
    constexpr auto to_int() const -> T
    {
        if (!this->is_integral())
        {
            throw std::runtime_error("basic_half_int cannot be converted to an integer exactly");
        }
        return twice_ / T{2};
    }

    /// \brief Convert to the underlying integral type assuming exact integrality.
    /// \return Integral representation.
    /// \pre `is_integral()`
    constexpr auto to_int_assert() const -> T
    {
        DEBUG_PRECONDITION(this->is_integral());
        return twice_ / T{2};
    }

    /// \brief Add another half-integer value.
    /// \param other Value to add.
    /// \return Reference to `*this`.
    constexpr auto operator+=(basic_half_int other) -> basic_half_int&
    {
        twice_ += other.twice_;
        return *this;
    }

    /// \brief Subtract another half-integer value.
    /// \param other Value to subtract.
    /// \return Reference to `*this`.
    constexpr auto operator-=(basic_half_int other) -> basic_half_int&
    {
        twice_ -= other.twice_;
        return *this;
    }

    /// \brief Prefix increment by one.
    /// \return Reference to `*this`.
    constexpr auto operator++() -> basic_half_int&
    {
        twice_ += T{2};
        return *this;
    }

    /// \brief Prefix decrement by one.
    /// \return Reference to `*this`.
    constexpr auto operator--() -> basic_half_int&
    {
        twice_ -= T{2};
        return *this;
    }

    /// \brief Postfix increment by one.
    /// \return Previous value.
    constexpr auto operator++(int) -> basic_half_int
    {
        auto const old = *this;
        ++(*this);
        return old;
    }

    /// \brief Postfix decrement by one.
    /// \return Previous value.
    constexpr auto operator--(int) -> basic_half_int
    {
        auto const old = *this;
        --(*this);
        return old;
    }

    /// \brief Compare two half-integer values.
    /// \param other Other value to compare.
    /// \return Ordering result based on the doubled representation.
    constexpr auto operator<=>(basic_half_int const& other) const = default;

    /// \brief Parse a textual half-integer representation.
    /// \param text String form such as `"3"`, `"5/2"`, or `"1.5"`.
    /// \return Parsed half-integer value.
    static auto parse(std::string_view text) -> basic_half_int
    {
        if (text.empty())
        {
            throw std::runtime_error("basic_half_int parse failed: empty string");
        }

        auto const slash = text.find('/');
        if (slash != std::string_view::npos)
        {
            auto const numerator = parse_integral<T>(text.substr(0, slash));
            auto const denominator = text.substr(slash + 1);
            if (denominator != "2")
            {
                throw std::runtime_error("basic_half_int parse failed: expected denominator 2");
            }
            return from_twice(numerator);
        }

        auto const dot = text.find('.');
        if (dot == std::string_view::npos)
        {
            return basic_half_int(parse_integral<T>(text));
        }

        auto const whole = text.substr(0, dot);
        auto const fractional = text.substr(dot + 1);
        if (fractional.empty())
        {
            return basic_half_int(parse_integral<T>(whole));
        }
        if (fractional != "0" && fractional != "5")
        {
            throw std::runtime_error("basic_half_int parse failed: expected decimal .0 or .5");
        }

        auto const sign = (!whole.empty() && whole.front() == '-') ? T{-1} : T{1};
        auto const twice_whole = checked_double(parse_integral<T>(whole));
        auto const half = (fractional == "5") ? sign : T{0};
        return from_twice(twice_whole + half);
    }

  private:
    /// \brief Construct from the doubled representation without rechecking.
    /// \param twice_value Stored doubled value.
    /// \return Half-integer value with that doubled representation.
    static constexpr auto from_twice(T twice_value) -> basic_half_int
    {
        return basic_half_int(twice_value, twice_tag{});
    }

    /// \brief Double an integral value with overflow checking.
    /// \param value Integral value to double.
    /// \return Doubled representation.
    static constexpr auto checked_double(T value) -> T
    {
        if (value > std::numeric_limits<T>::max() / T{2} || value < std::numeric_limits<T>::min() / T{2})
        {
            throw std::overflow_error("basic_half_int doubled representation overflow");
        }
        return value * T{2};
    }

    /// \brief Round a floating-point value to the nearest doubled representation.
    /// \param value Floating-point value to round.
    /// \return Rounded doubled representation.
    template <std::floating_point U> static auto rounded_twice(U value) -> T
    {
        auto const doubled = std::round(static_cast<long double>(value) * 2.0L);
        if (doubled > static_cast<long double>(std::numeric_limits<T>::max()) ||
            doubled < static_cast<long double>(std::numeric_limits<T>::min()))
        {
            throw std::overflow_error("basic_half_int rounded representation overflow");
        }
        return static_cast<T>(doubled);
    }

    /// \brief Parse an integral value from a string view.
    /// \tparam Int Integral result type.
    /// \param text Text to parse.
    /// \return Parsed integral value.
    template <std::integral Int> static auto parse_integral(std::string_view text) -> Int
    {
        if (text.empty())
        {
            throw std::runtime_error("basic_half_int parse failed: missing integer component");
        }
        auto const begin = text.data();
        auto const end = text.data() + text.size();
        Int value{};
        auto const [ptr, ec] = std::from_chars(begin, end, value);
        if (ec != std::errc{} || ptr != end)
        {
            throw std::runtime_error("basic_half_int parse failed: invalid integer component");
        }
        return value;
    }

    T twice_ = 0;
};

/// \brief Default half-integer type used by Uni20.
using half_int = basic_half_int<std::int64_t>;

/// \brief Construct a half-integer from its doubled representation.
/// \tparam T Signed storage type.
/// \param twice_value Doubled representation.
/// \return Half-integer value with that doubled representation.
template <std::signed_integral T>
constexpr auto from_twice(T twice_value) -> basic_half_int<T>
{
    return basic_half_int<T>(twice_value, typename basic_half_int<T>::twice_tag{});
}

/// \brief Add two half-integer values.
/// \tparam T Signed storage type.
/// \param lhs Left operand.
/// \param rhs Right operand.
/// \return Sum of the operands.
template <std::signed_integral T>
constexpr auto operator+(basic_half_int<T> lhs, basic_half_int<T> rhs) -> basic_half_int<T>
{
    lhs += rhs;
    return lhs;
}

/// \brief Subtract two half-integer values.
/// \tparam T Signed storage type.
/// \param lhs Left operand.
/// \param rhs Right operand.
/// \return Difference of the operands.
template <std::signed_integral T>
constexpr auto operator-(basic_half_int<T> lhs, basic_half_int<T> rhs) -> basic_half_int<T>
{
    lhs -= rhs;
    return lhs;
}

/// \brief Negate a half-integer value.
/// \tparam T Signed storage type.
/// \param value Operand to negate.
/// \return Negated value.
template <std::signed_integral T> constexpr auto operator-(basic_half_int<T> value) -> basic_half_int<T>
{
    return from_twice<T>(-value.twice());
}

/// \brief Multiply a half-integer value by an integral scalar.
/// \tparam T Signed storage type.
/// \tparam U Integral multiplier type.
/// \param lhs Half-integer value.
/// \param rhs Integral multiplier.
/// \return Product as a half-integer value.
template <std::signed_integral T, std::integral U>
constexpr auto operator*(basic_half_int<T> lhs, U rhs) -> basic_half_int<T>
{
    return from_twice<T>(lhs.twice() * static_cast<T>(rhs));
}

/// \brief Multiply an integral scalar by a half-integer value.
/// \tparam T Signed storage type.
/// \tparam U Integral multiplier type.
/// \param lhs Integral multiplier.
/// \param rhs Half-integer value.
/// \return Product as a half-integer value.
template <std::integral U, std::signed_integral T>
constexpr auto operator*(U lhs, basic_half_int<T> rhs) -> basic_half_int<T>
{
    return rhs * lhs;
}

/// \brief Multiply a half-integer value by a floating-point scalar.
/// \tparam T Signed storage type.
/// \tparam U Floating-point type.
/// \param lhs Half-integer value.
/// \param rhs Floating-point multiplier.
/// \return Product as a floating-point value.
template <std::signed_integral T, std::floating_point U>
constexpr auto operator*(basic_half_int<T> lhs, U rhs) -> U
{
    return static_cast<U>(lhs.to_double()) * rhs;
}

/// \brief Multiply a floating-point scalar by a half-integer value.
/// \tparam U Floating-point type.
/// \tparam T Signed storage type.
/// \param lhs Floating-point multiplier.
/// \param rhs Half-integer value.
/// \return Product as a floating-point value.
template <std::floating_point U, std::signed_integral T>
constexpr auto operator*(U lhs, basic_half_int<T> rhs) -> U
{
    return lhs * static_cast<U>(rhs.to_double());
}

/// \brief Multiply two half-integer values as floating-point numbers.
/// \tparam T Signed storage type.
/// \param lhs Left operand.
/// \param rhs Right operand.
/// \return Product as `double`.
template <std::signed_integral T> constexpr auto operator*(basic_half_int<T> lhs, basic_half_int<T> rhs) -> double
{
    return lhs.to_double() * rhs.to_double();
}

/// \brief Divide a half-integer value by a floating-point scalar.
/// \tparam T Signed storage type.
/// \tparam U Floating-point type.
/// \param lhs Half-integer value.
/// \param rhs Floating-point divisor.
/// \return Quotient as a floating-point value.
template <std::signed_integral T, std::floating_point U>
constexpr auto operator/(basic_half_int<T> lhs, U rhs) -> U
{
    return static_cast<U>(lhs.to_double()) / rhs;
}

/// \brief Return the absolute value of a half-integer.
/// \tparam T Signed storage type.
/// \param value Input value.
/// \return Absolute value.
template <std::signed_integral T> constexpr auto abs(basic_half_int<T> value) -> basic_half_int<T>
{
    return (value.twice() < 0) ? -value : value;
}

/// \brief Return whether a half-integer value is integral.
/// \tparam T Signed storage type.
/// \param value Value to inspect.
/// \return `true` if the value is integral.
template <std::signed_integral T> constexpr auto is_integral(basic_half_int<T> value) -> bool
{
    return value.is_integral();
}

/// \brief Convert a half-integer to its integral value.
/// \tparam T Signed storage type.
/// \param value Value to convert.
/// \return Integral representation.
template <std::signed_integral T> constexpr auto to_int(basic_half_int<T> value) -> T
{
    return value.to_int();
}

/// \brief Convert a half-integer to its integral value assuming exact integrality.
/// \tparam T Signed storage type.
/// \param value Value to convert.
/// \return Integral representation.
template <std::signed_integral T> constexpr auto to_int_assert(basic_half_int<T> value) -> T
{
    return value.to_int_assert();
}

/// \brief Format a half-integer as an integer or `n/2`.
/// \tparam T Signed storage type.
/// \param value Value to format.
/// \return Fractional string form.
template <std::signed_integral T> inline auto to_string_fraction(basic_half_int<T> value) -> std::string
{
    if (value.is_integral())
    {
        return std::to_string(value.to_int_assert());
    }
    return std::to_string(value.twice()) + "/2";
}

/// \brief Format a half-integer as an integer or decimal `.5`.
/// \tparam T Signed storage type.
/// \param value Value to format.
/// \return Decimal string form.
template <std::signed_integral T> inline auto to_string(basic_half_int<T> value) -> std::string
{
    if (value.is_integral())
    {
        return std::to_string(value.to_int_assert());
    }

    auto const whole = value.twice() / T{2};
    auto const fractional = (value.twice() < 0 && whole == 0) ? "-0.5" : std::to_string(whole) + ".5";
    return fractional;
}

/// \brief Return `(-1)^x` for an integral exponent.
/// \tparam T Integral exponent type.
/// \param value Integral exponent.
/// \return `1` for even `x`, `-1` for odd `x`.
template <std::integral T> constexpr auto minus1pow(T value) -> int
{
    return 1 - static_cast<int>((value & T{1}) << 1);
}

/// \brief Return whether three nonnegative half-integers satisfy the triangle condition.
/// \tparam T Signed storage type.
/// \param a First side.
/// \param b Second side.
/// \param c Third side.
/// \return `true` if `|a-c| <= b <= a+c` and `a+b+c` is integral.
template <std::signed_integral T>
constexpr auto is_triangle(basic_half_int<T> a, basic_half_int<T> b, basic_half_int<T> c) -> bool
{
    return (b >= abs(a - c)) && (b <= a + c) && (a + b + c).is_integral();
}

/// \brief Stream a half-integer in integer or `.5` decimal form.
/// \tparam T Signed storage type.
/// \param out Output stream.
/// \param value Value to format.
/// \return Output stream.
template <std::signed_integral T>
inline auto operator<<(std::ostream& out, basic_half_int<T> value) -> std::ostream&
{
    out << to_string(value);
    return out;
}

/// \brief Parse a half-integer from integer, `.0`, `.5`, or `n/2` form.
/// \tparam T Signed storage type.
/// \param in Input stream.
/// \param value Parsed result.
/// \return Input stream.
template <std::signed_integral T>
inline auto operator>>(std::istream& in, basic_half_int<T>& value) -> std::istream&
{
    std::string token;
    in >> token;
    if (!in)
    {
        return in;
    }

    try
    {
        value = basic_half_int<T>::parse(token);
    }
    catch (std::runtime_error const&)
    {
        in.setstate(std::ios::failbit);
    }
    return in;
}

} // namespace uni20

template <std::signed_integral T, typename CharT> struct std::formatter<uni20::basic_half_int<T>, CharT>
{
    constexpr auto parse(std::basic_format_parse_context<CharT>& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(uni20::basic_half_int<T> const& value, FormatContext& ctx) const
    {
        return std::format_to(ctx.out(), "{}", uni20::to_string(value));
    }
};

template <std::signed_integral T> struct std::hash<uni20::basic_half_int<T>>
{
    /// \brief Hash a half-integer by its doubled representation.
    /// \param value Half-integer value to hash.
    /// \return Hash of the doubled representation.
    auto operator()(uni20::basic_half_int<T> const& value) const noexcept -> std::size_t
    {
        return std::hash<T>{}(value.twice());
    }
};
