/**
 * \file u1.hpp
 * \brief Defines `U1`, the value-level quantum number type for one U(1) symmetry factor.
 */

#pragma once

#include <uni20/common/half_int.hpp>
#include <uni20/symmetry/symmetryfactor.hpp>

#include <concepts>
#include <cstdint>
#include <format>
#include <functional>
#include <iosfwd>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace uni20
{

/// \brief Value-level quantum number type representing one U(1) irrep.
class U1 {
  public:
    /// \brief Construct the scalar U(1) irrep.
    constexpr U1() = default;

    /// \brief Construct from a half-integer U(1) charge.
    /// \param value U(1) charge to store.
    constexpr explicit U1(half_int value) : value_(value) {}

    /// \brief Construct from an integral U(1) charge.
    /// \param value Integral U(1) charge.
    template <std::integral T> constexpr U1(T value) : value_(value) {}

    /// \brief Construct from a floating-point U(1) charge by rounding to the nearest half-integer.
    /// \param value Floating-point U(1) charge.
    template <std::floating_point T> explicit U1(T value) : value_(value) {}

    /// \brief Return the stored U(1) charge.
    /// \return Half-integer U(1) charge.
    constexpr auto value() const -> half_int { return value_; }

    /// \brief Compare two U(1) irreps by charge.
    /// \param other Other U(1) irrep.
    /// \return Comparison result of the stored charges.
    constexpr auto operator<=>(U1 const& other) const = default;

  private:
    half_int value_{};
};

/// \brief Return the dual U(1) irrep.
/// \param value U(1) irrep to dualize.
/// \return U(1) irrep with negated charge.
inline constexpr auto dual(U1 value) -> U1 { return U1{-value.value()}; }

/// \brief Return the quantum dimension of a U(1) irrep.
/// \param value U(1) irrep to inspect.
/// \return Quantum dimension, always `1.0`.
inline constexpr auto qdim(U1 value) -> double
{
    static_cast<void>(value);
    return 1.0;
}

/// \brief Return the integral degree of a U(1) irrep.
/// \param value U(1) irrep to inspect.
/// \return Integral degree, always `1`.
inline constexpr auto degree(U1 value) -> int
{
    static_cast<void>(value);
    return 1;
}

/// \brief Add two U(1) irreps using charge addition.
/// \param lhs Left U(1) irrep.
/// \param rhs Right U(1) irrep.
/// \return U(1) irrep with summed charge.
inline constexpr auto operator+(U1 lhs, U1 rhs) -> U1 { return U1{lhs.value() + rhs.value()}; }

/// \brief Subtract two U(1) irreps using charge subtraction.
/// \param lhs Left U(1) irrep.
/// \param rhs Right U(1) irrep.
/// \return U(1) irrep with subtracted charge.
inline constexpr auto operator-(U1 lhs, U1 rhs) -> U1 { return U1{lhs.value() - rhs.value()}; }

/// \brief Format a U(1) irrep in decimal half-integer form.
/// \param value U(1) irrep to format.
/// \return Decimal half-integer string.
inline auto to_string(U1 value) -> std::string { return uni20::to_string(value.value()); }

/// \brief Format a U(1) irrep in fractional `n/2` form.
/// \param value U(1) irrep to format.
/// \return Fractional string.
inline auto to_string_fraction(U1 value) -> std::string { return uni20::to_string_fraction(value.value()); }

/// \brief Stream a U(1) irrep in decimal half-integer form.
/// \param out Output stream.
/// \param value U(1) irrep to stream.
/// \return Output stream.
inline auto operator<<(std::ostream& out, U1 value) -> std::ostream&
{
    out << to_string(value);
    return out;
}

} // namespace uni20

namespace uni20::detail
{

/// \brief Internal codec for the value-level `U1` type.
template <> struct SymmetryFactorTraits<uni20::U1>
{
    /// \brief Return the canonical symmetry-factor type name.
    /// \return Canonical factor type string.
    static constexpr auto type_name() -> std::string_view { return "U(1)"; }

    /// \brief Encode one U(1) irrep into canonical display order.
    /// \param value U(1) irrep to encode.
    /// \return Packed factor-local code with `0` reserved for the scalar irrep.
    static constexpr auto encode(uni20::U1 value) -> std::uint64_t
    {
        auto const twice_value = value.value().twice();
        if (twice_value == 0)
        {
            return 0;
        }
        if (twice_value > 0)
        {
            auto const magnitude = static_cast<std::uint64_t>(twice_value);
            return magnitude * 2 - 1;
        }

        if (twice_value == std::numeric_limits<half_int::value_type>::min())
        {
            throw std::overflow_error("U1 encoding cannot represent the most negative half_int value");
        }

        auto const magnitude = static_cast<std::uint64_t>(-twice_value);
        return magnitude * 2;
    }

    /// \brief Decode one packed factor-local code into a U(1) irrep.
    /// \param code Packed factor-local code.
    /// \return Decoded U(1) irrep.
    static constexpr auto decode(std::uint64_t code) -> uni20::U1
    {
        if (code == 0)
        {
            return uni20::U1{};
        }
        if ((code & 1U) != 0U)
        {
            return uni20::U1{from_twice(static_cast<half_int::value_type>((code + 1) / 2))};
        }
        return uni20::U1{from_twice(-static_cast<half_int::value_type>(code / 2))};
    }
};

/// \brief Return the canonical U(1) symmetry-factor adapter instance.
/// \return Process-global U(1) factor adapter.
inline auto u1_factor() -> SymmetryFactor<uni20::U1> const&
{
    static SymmetryFactor<uni20::U1> factor;
    static auto const* registered = register_symmetry_factor(&factor);
    static_cast<void>(registered);
    return factor;
}

/// \brief Register the built-in symmetry factors needed by the first implementation.
inline void register_builtin_symmetry_factors() { static_cast<void>(u1_factor()); }

} // namespace uni20::detail

template <typename CharT> struct std::formatter<uni20::U1, CharT>
{
    constexpr auto parse(std::basic_format_parse_context<CharT>& ctx) { return ctx.begin(); }

    template <typename FormatContext> auto format(uni20::U1 const& value, FormatContext& ctx) const
    {
        return std::format_to(ctx.out(), "{}", uni20::to_string(value));
    }
};

template <> struct std::hash<uni20::U1>
{
    /// \brief Hash a U(1) irrep by its stored half-integer charge.
    /// \param value U(1) irrep to hash.
    /// \return Hash of the stored charge.
    auto operator()(uni20::U1 const& value) const noexcept -> std::size_t
    {
        return std::hash<uni20::half_int>{}(value.value());
    }
};
